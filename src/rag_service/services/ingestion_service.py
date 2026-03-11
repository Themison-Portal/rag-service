"""
RAG Ingestion Service - PDF parsing, chunking, and embedding.
"""
import logging
from datetime import datetime
from typing import AsyncIterator, List, Optional
from uuid import UUID, uuid4

import httpx
from langchain_core.documents import Document
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.models.chunks import DocumentChunkDocling
from rag_service.clients.openai_client import get_embedding_client
from rag_service.config import get_settings
from rag_service.cache.semantic_cache import SemanticCacheService

logger = logging.getLogger(__name__)
settings = get_settings()


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
        stmt = delete(DocumentChunkDocling).where(
            DocumentChunkDocling.document_id == document_id
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount

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

    async def _insert_docling_chunks(
        self,
        document_id: UUID,
        chunks: List[Document],
        embeddings: List[List[float]],
        contextual_summaries: Optional[List[str]] = None,
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
                    content=chunk.page_content,
                    page_number=citation_meta["page_number"],
                    chunk_metadata={**chunk.metadata, "chunk_index": i},
                    embedding=embedding,
                    contextual_summary=contextual_summary,
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
                deleted_semantic = await self.semantic_cache_service.invalidate_document(document_id)
                if deleted_semantic > 0:
                    logger.info(f"Invalidated {deleted_semantic} semantic cache entries")

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

            loader = DoclingLoader(
                file_path=document_url,
                export_type=ExportType.DOC_CHUNKS,
                chunker=HybridChunker(tokenizer=tokenizer, chunk_size=chunk_size),
            )
            docs = loader.load()
            texts = [doc.page_content for doc in docs]

            yield IngestionProgress("CHUNKING", 50, f"Created {len(docs)} chunks...")

            # Stage 4: Generate embeddings
            yield IngestionProgress("EMBEDDING", 60, f"Generating embeddings for {len(texts)} chunks...")

            chunk_embeddings = await self.embedding_client.aembed_documents(texts)

            yield IngestionProgress("EMBEDDING", 80, "Embeddings complete...")

            # Stage 5: Store in database
            yield IngestionProgress("STORING", 85, "Storing chunks in database...")

            await self._insert_docling_chunks(document_id, docs, chunk_embeddings)

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
