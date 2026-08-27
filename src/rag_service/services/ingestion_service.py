"""
RAG Ingestion Service - PDF parsing, chunking, and embedding.
"""

import asyncio
import hashlib
import logging
import json
import os
import tempfile
from datetime import datetime
from typing import AsyncIterator, List, Optional
from uuid import UUID, uuid4


import httpx
from langchain_core.documents import Document
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.models.chunks import DocumentChunkDocling
from rag_service.clients.openai_client import get_embedding_client
from rag_service.clients.anthropic_client import get_anthropic_client
from rag_service.config import get_settings
from rag_service.cache.semantic_cache import SemanticCacheService

logger = logging.getLogger(__name__)
settings = get_settings()
print(f"[STARTUP] contextual_retrieval_enabled={settings.contextual_retrieval_enabled}")


class IngestionProgress:
    """Progress update during ingestion."""

    def __init__(self, stage: str, progress: int, message: str, result: dict = None):
        self.stage = stage
        self.progress = progress
        self.message = message
        self.result = result


class RagIngestionService:
    """
    Service for PDF ingestion and chunking using Docling + OpenAI embeddings.
    """

    CONTEXTUAL_SYSTEM_PROMPT = (
        "You are given a document and a chunk extracted from it. Give a short "
        "(1-2 sentence) context that situates the chunk within the overall "
        "document, to improve search retrieval of the chunk when read in "
        "isolation. Answer with only the context, nothing else."
    )

    def __init__(
        self,
        db: AsyncSession,
        semantic_cache_service: Optional[SemanticCacheService] = None,
    ):
        self.db = db
        self.embedding_client = get_embedding_client()
        self.semantic_cache_service = semantic_cache_service

    async def _delete_existing_chunks(self, document_id: UUID) -> int:
        """Delete existing chunks before re-ingestion."""
        stmt = delete(DocumentChunkDocling).where(DocumentChunkDocling.document_id == document_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount

    def _hash_chunk(self, content: str) -> str:
        """Hash raw (pre-overlap) chunk content for re-ingestion dedup."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def _fetch_existing_chunk_cache(self, document_id: UUID) -> dict:
        """
        Map content_hash -> (embedding, contextual_summary) for chunks
        already stored for this document, so unchanged chunks can skip the
        LLM/embedding calls on re-ingestion. Must be called BEFORE
        _delete_existing_chunks wipes the old rows.
        """
        stmt = select(
            DocumentChunkDocling.content_hash,
            DocumentChunkDocling.embedding,
            DocumentChunkDocling.contextual_summary,
        ).where(DocumentChunkDocling.document_id == document_id)
        result = await self.db.execute(stmt)
        return {
            row.content_hash: (row.embedding, row.contextual_summary)
            for row in result
            if row.content_hash
        }

    def _extract_docling_citation_metadata(self, metadata_json: dict) -> dict:
        """Returns a dict with page_number and headings for a chunk."""
        try:
            dl_meta = metadata_json.get("dl_meta", {})
            doc_items = dl_meta.get("doc_items", [])
            headings = dl_meta.get("headings", [])

            page_number = None
            if doc_items:
                prov_list = doc_items[0].get("prov", [])
                if prov_list:
                    page_number = prov_list[0].get("page_no")

            return {"page_number": page_number, "headings": headings or []}

        except Exception:
            return {"page_number": None, "headings": []}

    def _log_llm_usage(
        self,
        call_type: str,
        response,
        *,
        document_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
        model: Optional[str] = None,
    ) -> None:
        """Structured log line per LLM call, for cost tracking."""

        usage = getattr(response, "usage", None)
        logger.info(
            json.dumps(
                {
                    "event": "llm_usage",
                    "call_type": call_type,
                    "model": model or getattr(response, "model", None),
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                    "document_id": str(document_id) if document_id else None,
                    "organization_id": str(organization_id) if organization_id else None,
                },
                default=str,
            )
        )

    async def _generate_contextual_summaries(
        self,
        docs: List[Document],
        full_document_text: str,
        document_id: UUID,
        organization_id: UUID,
    ) -> List[Optional[str]]:
        """
        Phase 4: Contextual retrieval. For each chunk, ask Claude for a short
        blurb situating it within the document, to be prepended before
        embedding.

        The full document is sent as a cache_control block so N chunks cost
        ~1 full-document read, not N.

        POINT 1 FIX: the first chunk is awaited alone (cache warm-up) before
        the rest are gathered concurrently. Without this, the semaphore's
        first batch of up to 5 concurrent calls all start before call #1's
        cache write has landed, so none of them get a cache read - only
        calls after the first batch benefited. Running one call solo first
        guarantees the cache is warm before any concurrent calls fire.

        No-op (returns all-None) unless settings.contextual_retrieval_enabled.
        """
        if not settings.contextual_retrieval_enabled:
            return [None] * len(docs)

        client = get_anthropic_client()
        window = settings.contextual_context_window
        use_full_document = len(docs) <= max(window * 5, 20)

        summaries: List[Optional[str]] = [None] * len(docs)
        cache_hit_count = [0]  # mutable counter, closure-friendly
        semaphore = asyncio.Semaphore(5)  # bound concurrency against API rate limits

        async def _summarize_one(
            i: int, chunk_text: str, context_text: str, cacheable: bool
        ) -> None:
            async with semaphore:
                try:
                    system_blocks = [{"type": "text", "text": self.CONTEXTUAL_SYSTEM_PROMPT}]
                    doc_block = {"type": "text", "text": f"<document>\n{context_text}\n</document>"}
                    if cacheable:
                        doc_block["cache_control"] = {"type": "ephemeral"}
                    system_blocks.append(doc_block)

                    response = await client.messages.create(
                        model=settings.llm_model,
                        max_tokens=150,
                        system=system_blocks,
                        messages=[{"role": "user", "content": f"<chunk>\n{chunk_text}\n</chunk>"}],
                    )
                    self._log_llm_usage(
                        "ingestion_context",
                        response,
                        document_id=document_id,
                        organization_id=organization_id,
                    )
                    usage = getattr(response, "usage", None)
                    if getattr(usage, "cache_read_input_tokens", 0):
                        cache_hit_count[0] += 1
                    summaries[i] = response.content[0].text.strip()
                except Exception as e:
                    logger.warning(f"[CONTEXTUAL_RETRIEVAL] chunk {i} failed: {e}")

        tasks = []
        for i, doc in enumerate(docs):
            if use_full_document:
                context_text = full_document_text
            else:
                lo, hi = max(0, i - window), min(len(docs), i + window + 1)
                context_text = "\n\n".join(d.page_content for d in docs[lo:hi])
            tasks.append(_summarize_one(i, doc.page_content, context_text, use_full_document))

        # POINT 1 FIX: warm the cache with one solo call first.
        if use_full_document and len(tasks) > 1:
            await tasks[0]
            await asyncio.gather(*tasks[1:])
        else:
            await asyncio.gather(*tasks)

        logger.info(f"[CONTEXTUAL_RETRIEVAL] cache hits: {cache_hit_count[0]}/{len(docs)} chunks")

        failed = sum(1 for s in summaries if s is None)
        if failed:
            logger.warning(
                f"[CONTEXTUAL_RETRIEVAL] {failed}/{len(docs)} chunks got no summary; "
                f"those embed/store without one."
            )
        return summaries

    async def _insert_docling_chunks(
        self,
        document_id: UUID,
        organization_id: UUID,
        chunks: List[Document],
        embeddings: List[List[float]],
        contextual_summaries: Optional[List[str]] = None,
        content_hashes: Optional[List[str]] = None,
    ) -> None:
        """Insert Docling chunks into the database."""
        try:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                citation_meta = self._extract_docling_citation_metadata(chunk.metadata)

                contextual_summary = None
                if contextual_summaries and i < len(contextual_summaries):
                    contextual_summary = contextual_summaries[i]

                chunk_record = DocumentChunkDocling(
                    id=uuid4(),
                    document_id=document_id,
                    organization_id=organization_id,
                    content=chunk.page_content,
                    page_number=citation_meta["page_number"],
                    chunk_metadata={**chunk.metadata, "chunk_index": i},
                    embedding=embedding,
                    contextual_summary=contextual_summary,
                    content_hash=content_hashes[i] if content_hashes else None,
                    created_at=datetime.now(),
                )
                self.db.add(chunk_record)

            await self.db.commit()

        except Exception as e:
            await self.db.rollback()
            raise RuntimeError(f"Failed to insert chunks: {str(e)}")

    async def ingest_pdf(
        self,
        document_url: str,
        document_id: UUID,
        organization_id: UUID,
        chunk_size: int = 750,
    ) -> AsyncIterator[IngestionProgress]:
        """
        Complete ingestion pipeline for a PDF with progress streaming.

        Yields:
            IngestionProgress objects with stage updates.
        """
        try:
            # Stage 1: Cache invalidation
            yield IngestionProgress("INVALIDATING", 5, "Invalidating existing caches...")

            if self.semantic_cache_service:
                deleted_semantic = await self.semantic_cache_service.invalidate_document(
                    document_id
                )
                if deleted_semantic > 0:
                    logger.info(f"Invalidated {deleted_semantic} semantic cache entries")

            # Stage 1.5 (POINT 3, NEW): load existing chunk cache BEFORE
            # deletion, so unchanged chunks can reuse their embedding/summary
            # instead of reprocessing from scratch.
            existing_chunk_cache = await self._fetch_existing_chunk_cache(document_id)

            # Stage 2: Delete existing chunks
            yield IngestionProgress("PREPARING", 10, "Deleting existing chunks...")

            deleted_chunks = await self._delete_existing_chunks(document_id)
            if deleted_chunks > 0:
                logger.info(f"Deleted {deleted_chunks} existing chunks")

            # Stage 3: Download and parse PDF
            yield IngestionProgress("DOWNLOADING", 15, "Downloading PDF...")

            # Import Docling here to avoid startup delays
            from docling.chunking import HybridChunker
            from langchain_docling.loader import DoclingLoader, ExportType

            # Get tokenizer
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

            yield IngestionProgress("PARSING", 30, "Parsing PDF with Docling...")

            # Docling's source resolver (docling_core `_is_safe_url`) is an SSRF
            # guard that only permits globally-routable PUBLIC IPs, so handing it
            # an internal URL like http://backend:8080/local-files/... is
            # rejected with "URL is not allowed". Download the bytes ourselves
            # (httpx has no such restriction) and parse from a LOCAL file, which
            # skips the URL guard entirely.
            cleanup_path: Optional[str] = None
            if document_url.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    resp = await client.get(document_url)
                    resp.raise_for_status()
                    pdf_bytes = resp.content
                tmp_fd, cleanup_path = tempfile.mkstemp(suffix=".pdf")
                os.close(tmp_fd)
                with open(cleanup_path, "wb") as f:
                    f.write(pdf_bytes)
                local_path = cleanup_path
            else:
                # Already a local path — use as-is.
                local_path = document_url

            try:
                loader = DoclingLoader(
                    file_path=local_path,
                    export_type=ExportType.DOC_CHUNKS,
                    chunker=HybridChunker(tokenizer=tokenizer, chunk_size=chunk_size),
                )
                docs = loader.load()
            finally:
                if cleanup_path:
                    try:
                        os.remove(cleanup_path)
                    except OSError:
                        pass

            texts = [doc.page_content for doc in docs]

            yield IngestionProgress("CHUNKING", 50, f"Created {len(docs)} chunks...")

            # Stage 3.4 (POINT 3, NEW): hash chunks, determine which are
            # unchanged vs new relative to the previous ingestion.
            chunk_hashes = [self._hash_chunk(t) for t in texts]
            reuse_mask = [h in existing_chunk_cache for h in chunk_hashes]
            new_count = sum(1 for r in reuse_mask if not r)
            logger.info(
                f"[INGEST_DEDUP] {len(docs) - new_count}/{len(docs)} chunks unchanged, "
                f"reusing cached embedding/summary; {new_count} new"
            )

            # Stage 3.5: Contextual retrieval (Phase 4) - only for NEW chunks now
            contextual_summaries: List[Optional[str]] = [None] * len(docs)
            if settings.contextual_retrieval_enabled:
                new_docs = [d for d, r in zip(docs, reuse_mask) if not r]
                if new_docs:
                    yield IngestionProgress(
                        "CONTEXTUALIZING",
                        55,
                        f"Generating context for {len(new_docs)} new chunks...",
                    )
                    full_document_text = "\n\n".join(texts)
                    new_summaries = await self._generate_contextual_summaries(
                        new_docs, full_document_text, document_id, organization_id
                    )
                    new_iter = iter(new_summaries)
                    for i, reuse in enumerate(reuse_mask):
                        contextual_summaries[i] = (
                            existing_chunk_cache[chunk_hashes[i]][1] if reuse else next(new_iter)
                        )
                else:
                    contextual_summaries = [existing_chunk_cache[h][1] for h in chunk_hashes]

            # Stage 4: Generate embeddings - only for NEW chunks now
            yield IngestionProgress(
                "EMBEDDING", 60, f"Generating embeddings for {new_count} new chunks..."
            )

            texts_to_embed_all = [
                f"{s}\n\n{t}" if s else t for s, t in zip(contextual_summaries, texts)
            ]
            new_indices = [i for i, r in enumerate(reuse_mask) if not r]
            new_texts_to_embed = [texts_to_embed_all[i] for i in new_indices]

            new_embeddings = (
                await self.embedding_client.aembed_documents(new_texts_to_embed)
                if new_texts_to_embed
                else []
            )
            new_emb_iter = iter(new_embeddings)
            chunk_embeddings = [
                existing_chunk_cache[chunk_hashes[i]][0] if reuse_mask[i] else next(new_emb_iter)
                for i in range(len(docs))
            ]

            yield IngestionProgress("EMBEDDING", 80, "Embeddings complete...")

            # Stage 5: Store in database
            yield IngestionProgress("STORING", 85, "Storing chunks in database...")

            await self._insert_docling_chunks(
                document_id,
                organization_id,
                docs,
                chunk_embeddings,
                contextual_summaries=contextual_summaries,
                content_hashes=chunk_hashes,
            )

            # Complete
            result = {
                "success": True,
                "document_id": str(document_id),
                "status": "ready",
                "chunks_count": len(docs),
                "created_at": datetime.now().isoformat(),
            }

            logger.info(f"PDF ingestion complete: {len(docs)} chunks")

            yield IngestionProgress("COMPLETE", 100, "Ingestion complete!", result)

        except Exception as e:
            logger.error(f"PDF ingestion failed: {e}")
            error_result = {
                "success": False,
                "document_id": str(document_id),
                "status": "error",
                "chunks_count": 0,
                "error": str(e),
            }
            yield IngestionProgress("ERROR", 0, f"Ingestion failed: {e}", error_result)
