"""
RAG Retrieval Service - Vector search and hybrid retrieval.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from rag_service.services.reranker import RerankerService

from rag_service.clients.openai_client import get_embedding_client
from rag_service.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RagRetrievalService:
    """
    Service for retrieving similar document chunks from the database.
    Supports both vector-only and hybrid (vector + BM25) search.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_client = get_embedding_client()
        self.reranker = RerankerService()

    def _embedding_to_pg_vector(self, emb: List[float]) -> str:
        """Convert Python list of floats to PostgreSQL vector format."""
        return "[" + ",".join(str(x) for x in emb) + "]"

    async def get_query_embedding(self, query_text: str) -> Tuple[List[float], dict]:
        """
        Get embedding for query text.
        Returns (embedding, timing_info).
        """
        timing_info = {"cache_hit": False, "embedding_ms": 0.0}

        embed_start = time.perf_counter()
        embedding = await self.embedding_client.aembed_query(query_text)
        timing_info["embedding_ms"] = (time.perf_counter() - embed_start) * 1000

        logger.info(f"[TIMING] Embedding generated in {timing_info['embedding_ms']:.2f}ms")
        return embedding, timing_info

    async def _search_similar_chunks_docling(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        organization_id: UUID,
        top_k: int = 20,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> Tuple[List[dict], dict]:
        """
        Retrieve top-k similar chunks using pgvector cosine similarity.
        """
        timing_info = {}

        if precomputed_embedding is not None:
            query_vector = precomputed_embedding
            timing_info["cache_hit"] = True
            timing_info["embedding_ms"] = 0.0
        else:
            query_vector, embed_timing = await self.get_query_embedding(query_text)
            timing_info.update(embed_timing)

        query_vector_str = self._embedding_to_pg_vector(query_vector)

        sql = text("""
            SELECT
                pc.content,
                pc.page_number,
                pc.chunk_metadata,
                1 - (pc.embedding <=> (:v)::vector) AS similarity
            FROM document_chunks_docling pc
            WHERE pc.document_id = :pid
            AND pc.organization_id = :org_id
            ORDER BY pc.embedding <=> (:v)::vector
            LIMIT :k
        """)

        db_start = time.perf_counter()
        result = await self.db.execute(
            sql, {"v": query_vector_str, "k": top_k, "pid": document_id, "org_id": organization_id}
        )
        rows = result.fetchall()
        timing_info["db_search_ms"] = (time.perf_counter() - db_start) * 1000

        logger.info(
            f"[TIMING] Vector search: {timing_info['db_search_ms']:.2f}ms, found {len(rows)} chunks"
        )

        docs = [
            {
                "page_content": row.content,
                "score": float(row.similarity),
                "metadata": {
                    "title": document_name,
                    "page": row.page_number,
                    "docling": row.chunk_metadata,
                },
            }
            for row in rows
        ]

        return docs, timing_info

    async def _search_bm25(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        organization_id: UUID,
        top_k: int = 20,
    ) -> List[dict]:
        """
        Full-text BM25 search using PostgreSQL tsvector.
        """
        sql = text("""
            SELECT
                pc.id,
                pc.content,
                pc.page_number,
                pc.chunk_metadata,
                ts_rank(pc.content_tsv, plainto_tsquery('english', :query)) AS bm25_score
            FROM document_chunks_docling pc
            WHERE pc.document_id = :pid
              AND pc.organization_id = :org_id
              AND pc.content_tsv @@ plainto_tsquery('english', :query)
            ORDER BY bm25_score DESC
            LIMIT :k
        """)

        db_start = time.perf_counter()
        result = await self.db.execute(
            sql, {"query": query_text, "k": top_k, "pid": document_id, "org_id": organization_id}
        )
        rows = result.fetchall()
        bm25_time = (time.perf_counter() - db_start) * 1000

        logger.info(f"[TIMING] BM25 search: {bm25_time:.2f}ms, found {len(rows)} chunks")

        docs = [
            {
                "id": str(row.id),
                "page_content": row.content,
                "score": float(row.bm25_score),
                "metadata": {
                    "title": document_name,
                    "page": row.page_number,
                    "docling": row.chunk_metadata,
                },
            }
            for row in rows
        ]
        return docs

    def _reciprocal_rank_fusion(
        self, vector_results: List[dict], bm25_results: List[dict], k: int = 60
    ) -> List[dict]:
        """
        Combine vector and BM25 results using Reciprocal Rank Fusion (RRF).
        """
        doc_map: Dict[str, dict] = {}
        rrf_scores: Dict[str, float] = {}

        # Process vector results
        for rank, doc in enumerate(vector_results):
            doc_id = doc.get("id") or hash(doc["page_content"])
            doc_id = str(doc_id)

            if doc_id not in doc_map:
                doc_map[doc_id] = doc.copy()
                doc_map[doc_id]["vector_rank"] = rank + 1
                doc_map[doc_id]["vector_score"] = doc.get("score", 0)

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.get("id") or hash(doc["page_content"])
            doc_id = str(doc_id)

            if doc_id not in doc_map:
                doc_map[doc_id] = doc.copy()

            doc_map[doc_id]["bm25_rank"] = rank + 1
            doc_map[doc_id]["bm25_score"] = doc.get("score", 0)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            doc["score"] = rrf_scores[doc_id]
            doc["rrf_score"] = rrf_scores[doc_id]
            fused_results.append(doc)

        logger.info(
            f"[HYBRID] RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 -> {len(fused_results)} merged"
        )
        return fused_results

    def _apply_confidence_filter(
        self, chunks: list, min_score: float, min_bm25_score: float = None
    ) -> tuple:
        """
        STEP 2 FIX (vector) + STEP 2b (bm25): filters out weak matches.

        Vector-sourced chunks: filtered on vector_score (real cosine similarity,
        min_score is calibrated against this scale).

        BM25-only chunks (no vector_score - keyword match the vector search
        missed): ts_rank isn't on a comparable scale to min_score, so it's
        min-max normalized within this result set first, then compared to
        min_bm25_score. If min_bm25_score is None (not yet empirically
        derived), BM25-only chunks are kept and just logged, not dropped -
        this is intentional "observe mode" until real score distributions
        are reviewed and a defensible threshold is set, the same way
        min_score was validated against real vector_score data.
        """
        # Normalize BM25 scores within this result set first.
        bm25_vals = [c.get("bm25_score") for c in chunks if c.get("bm25_score") is not None]
        if bm25_vals:
            max_v, min_v = max(bm25_vals), min(bm25_vals)
            range_v = max_v - min_v or 1e-9
            for c in chunks:
                if c.get("bm25_score") is not None:
                    c["bm25_score_normalized"] = (c["bm25_score"] - min_v) / range_v

        kept, dropped = [], []
        for c in chunks:
            vector_score = c.get("vector_score")
            bm25_norm = c.get("bm25_score_normalized")

            if vector_score is not None:
                passes = vector_score >= min_score
            elif bm25_norm is not None and min_bm25_score is not None:
                passes = bm25_norm >= min_bm25_score
            else:
                # No vector score, and either no BM25 score or no threshold
                # set yet - can't responsibly drop something we can't score.
                passes = True

            (kept if passes else dropped).append(c)

        if dropped:
            logger.info(
                f"[CONFIDENCE_FILTER] dropped {len(dropped)}/{len(chunks)} chunks "
                f"below thresholds (vector min_score={min_score}, bm25 min_score={min_bm25_score}) "
                f"(dropped vector_scores={[round(c.get('vector_score', 0), 4) for c in dropped if c.get('vector_score') is not None]}, "
                f"dropped bm25_scores_normalized={[round(c.get('bm25_score_normalized', 0), 4) for c in dropped if c.get('bm25_score_normalized') is not None]})"
            )

        bm25_only_scores = [
            round(c.get("bm25_score_normalized"), 4)
            for c in chunks
            if c.get("vector_score") is None and c.get("bm25_score_normalized") is not None
        ]
        if bm25_only_scores:
            logger.info(
                f"[BM25_SCORE_OBSERVE] normalized scores for BM25-only chunks: {bm25_only_scores}"
            )

        return kept, {
            "confidence_filter_applied": True,
            "confidence_filter_threshold": min_score,
            "confidence_filter_bm25_threshold": min_bm25_score,
            "confidence_filter_input_count": len(chunks),
            "confidence_filter_kept_count": len(kept),
            "confidence_filter_dropped_count": len(dropped),
            "confidence_filter_kept_scores": [
                round(c.get("vector_score", 0), 4)
                for c in kept
                if c.get("vector_score") is not None
            ],
            "confidence_filter_dropped_scores": [
                round(c.get("vector_score", 0), 4)
                for c in dropped
                if c.get("vector_score") is not None
            ],
            "confidence_filter_bm25_only_scores_normalized": bm25_only_scores,
        }

    async def _search_hybrid(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        organization_id: UUID,
        top_k: int = 20,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> Tuple[List[dict], dict]:
        """
        Hybrid search combining vector similarity and BM25.
        """
        timing_info = {"hybrid_search": True}
        hybrid_start = time.perf_counter()

        vector_results, vector_timing = await self._search_similar_chunks_docling(
            query_text, document_id, document_name, organization_id, top_k, precomputed_embedding
        )
        bm25_results = await self._search_bm25(
            query_text, document_id, document_name, organization_id, top_k
        )

        timing_info.update(vector_timing)
        timing_info["hybrid_parallel_ms"] = (time.perf_counter() - hybrid_start) * 1000

        rrf_k = settings.hybrid_search_rrf_k
        fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results, k=rrf_k)

        timing_info["vector_count"] = len(vector_results)
        timing_info["bm25_count"] = len(bm25_results)
        timing_info["fused_count"] = len(fused_results)

        logger.info(
            f"[HYBRID] Search complete: {len(fused_results)} results in {timing_info['hybrid_parallel_ms']:.2f}ms"
        )

        return fused_results, timing_info

    async def retrieve_similar_chunks(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        organization_id: UUID,
        top_k: int = None,
        min_score: float = None,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> Tuple[List[dict], dict]:
        """
        Retrieve and format top similar chunks for a query.

        top_k here is the FINAL number of chunks returned (post-rerank).
        Internally, fetch_k (settings.retrieval_fetch_k, wider than top_k)
        controls how many candidates are pulled before reranking trims
        them down - "retrieve wide, rerank to the top few" (Step 4).
        """
        if top_k is None:
            top_k = (
                settings.reranker_top_k if settings.reranker_enabled else settings.retrieval_top_k
            )
        if min_score is None:
            min_score = settings.retrieval_min_score

        fetch_k = max(settings.retrieval_fetch_k, top_k)

        retrieval_start = time.perf_counter()
        timing_info = {"chunk_cache_hit": False}

        # Use hybrid search if enabled
        if settings.hybrid_search_enabled:
            raw_chunks, search_timing = await self._search_hybrid(
                query_text,
                document_id,
                document_name,
                organization_id,
                fetch_k,
                precomputed_embedding,
            )
            filtered_chunks, filter_info = self._apply_confidence_filter(
                raw_chunks, min_score, min_bm25_score=settings.retrieval_min_bm25_score
            )
            timing_info.update(filter_info)
        else:
            raw_chunks, search_timing = await self._search_similar_chunks_docling(
                query_text,
                document_id,
                document_name,
                organization_id,
                fetch_k,
                precomputed_embedding,
            )
            filtered_chunks = [d for d in raw_chunks if d["score"] >= min_score]

        timing_info.update(search_timing)

        # STEP 4: rerank the confidence-filtered candidates down to top_k.
        reranked_chunks, rerank_timing = await self.reranker.rerank(
            query_text=query_text,
            chunks=filtered_chunks,
            top_n=top_k,
        )
        timing_info.update(rerank_timing)

        timing_info["retrieval_total_ms"] = (time.perf_counter() - retrieval_start) * 1000

        logger.info(
            f"[TIMING] Retrieval total: {timing_info['retrieval_total_ms']:.2f}ms, "
            f"{len(reranked_chunks)} chunks after filter+rerank"
        )

        return reranked_chunks, timing_info
