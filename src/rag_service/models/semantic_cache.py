"""
Semantic cache model for storing RAG responses with query embeddings.
Enables similarity-based cache lookup using pgvector.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped

from .base import Base


class SemanticCacheResponse(Base):
    """
    Cached RAG response with semantic embedding.

    Enables similarity-based cache lookup where queries with
    cosine similarity >= threshold return cached responses.
    """
    __tablename__ = 'semantic_cache_responses'

    id: Mapped[UUID] = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text: Mapped[str] = Column(Text, nullable=False)
    query_embedding: Mapped[List[float]] = Column(Vector(1536), nullable=False)
    document_id: Mapped[UUID] = Column(UUID(as_uuid=True), nullable=False)
    response_data: Mapped[Dict] = Column(JSON, nullable=False)
    hit_count: Mapped[int] = Column(Integer, default=0)
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    last_accessed_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    context_hash: Mapped[str] = Column(String(32), nullable=False)

    __table_args__ = (
        Index(
            'idx_semantic_cache_embedding_hnsw',
            'query_embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'query_embedding': 'vector_cosine_ops'}
        ),
        Index('idx_semantic_cache_document_id', 'document_id'),
    )

    def __repr__(self) -> str:
        return f"<SemanticCacheResponse(id={self.id}, query='{self.query_text[:30]}...', hits={self.hit_count})>"
