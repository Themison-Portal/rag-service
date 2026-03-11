"""
PostgreSQL-based semantic similarity cache for RAG responses.
Uses pgvector for efficient similarity search.
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.models.semantic_cache import SemanticCacheResponse

logger = logging.getLogger(__name__)


class SemanticCacheService:
    """
    Semantic similarity cache using PostgreSQL + pgvector.

    Finds cached responses where query embedding similarity >= threshold.
    Scoped by document_id to prevent cross-document cache hits.
    """

    DEFAULT_SIMILARITY_THRESHOLD = 0.90

    def __init__(
        self,
        db: AsyncSession,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ):
        self.db = db
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _embedding_to_pg_vector(embedding: List[float]) -> str:
        """Convert Python list to PostgreSQL vector literal."""
        return "[" + ",".join(str(x) for x in embedding) + "]"

    @staticmethod
    def hash_context(chunks: List[dict]) -> str:
        """
        Generate hash of chunk content for cache invalidation detection.
        """
        content = json.dumps(
            [c.get("page_content", "") for c in chunks],
            sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def get_similar_response(
        self,
        query_embedding: List[float],
        document_id: UUID,
        similarity_threshold: float = None
    ) -> Optional[Dict]:
        """
        Search for semantically similar cached response.

        Uses pgvector cosine similarity with HNSW index.
        Returns highest similarity match above threshold, or None.
        """
        threshold = similarity_threshold or self.similarity_threshold
        search_start = time.perf_counter()

        query_vector = self._embedding_to_pg_vector(query_embedding)

        sql = text("""
            SELECT
                id,
                query_text,
                response_data,
                context_hash,
                1 - (query_embedding <=> (:v)::vector) AS similarity
            FROM semantic_cache_responses
            WHERE document_id = :doc_id
              AND 1 - (query_embedding <=> (:v)::vector) >= :threshold
            ORDER BY query_embedding <=> (:v)::vector
            LIMIT 1
        """)

        try:
            result = await self.db.execute(sql, {
                "v": query_vector,
                "doc_id": document_id,
                "threshold": threshold
            })
            row = result.fetchone()

            search_ms = (time.perf_counter() - search_start) * 1000

            if row:
                logger.info(
                    f"[CACHE] Semantic [HIT] - similarity={row.similarity:.4f} in {search_ms:.2f}ms"
                )
                await self._update_cache_hit(row.id)

                return {
                    "response": row.response_data,
                    "similarity": float(row.similarity),
                    "original_query": row.query_text,
                    "context_hash": row.context_hash,
                    "cache_id": str(row.id)
                }
            else:
                logger.info(f"[CACHE] Semantic [MISS] - threshold: {threshold} in {search_ms:.2f}ms")
                return None

        except Exception as e:
            try:
                await self.db.rollback()
            except Exception:
                pass
            logger.warning(f"[SEMANTIC_CACHE] Search error: {e}")
            return None

    async def _update_cache_hit(self, cache_id: UUID) -> None:
        """Update hit count and last accessed time."""
        try:
            stmt = (
                update(SemanticCacheResponse)
                .where(SemanticCacheResponse.id == cache_id)
                .values(
                    hit_count=SemanticCacheResponse.hit_count + 1,
                    last_accessed_at=datetime.now(timezone.utc)
                )
            )
            await self.db.execute(stmt)
            await self.db.commit()
        except Exception as e:
            logger.warning(f"[SEMANTIC_CACHE] Failed to update hit count: {e}")

    async def store_response(
        self,
        query_text: str,
        query_embedding: List[float],
        document_id: UUID,
        response: Dict,
        context_hash: str
    ) -> None:
        """Store new response in semantic cache."""
        store_start = time.perf_counter()

        try:
            cache_entry = SemanticCacheResponse(
                query_text=query_text,
                query_embedding=query_embedding,
                document_id=document_id,
                response_data=response,
                context_hash=context_hash,
                hit_count=0,
                created_at=datetime.now(timezone.utc),
                last_accessed_at=datetime.now(timezone.utc)
            )

            self.db.add(cache_entry)
            await self.db.commit()

            store_ms = (time.perf_counter() - store_start) * 1000
            logger.info(f"[CACHE] Semantic [STORE] - stored in {store_ms:.2f}ms")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"[SEMANTIC_CACHE] Store error: {e}")

    async def invalidate_document(self, document_id: UUID) -> int:
        """Delete all cache entries for a document."""
        invalidate_start = time.perf_counter()

        try:
            stmt = delete(SemanticCacheResponse).where(
                SemanticCacheResponse.document_id == document_id
            )
            result = await self.db.execute(stmt)
            await self.db.commit()

            deleted_count = result.rowcount
            invalidate_ms = (time.perf_counter() - invalidate_start) * 1000

            if deleted_count > 0:
                logger.info(
                    f"[CACHE] Semantic [INVALIDATE] - {deleted_count} entries in {invalidate_ms:.2f}ms"
                )

            return deleted_count

        except Exception as e:
            await self.db.rollback()
            logger.error(f"[SEMANTIC_CACHE] Invalidation error: {e}")
            return 0
