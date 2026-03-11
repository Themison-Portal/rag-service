"""
Document chunk model for storing embeddings with pgvector.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, Computed, DateTime, Index, Integer, Text
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import Mapped

from .base import Base


class DocumentChunkDocling(Base):
    """
    A model representing a document chunk with embedding.
    Uses pgvector for efficient similarity search.
    """
    __tablename__ = 'document_chunks_docling'

    id: Mapped[UUID] = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[UUID] = Column(UUID(as_uuid=True), nullable=False)
    content: Mapped[str] = Column(Text, nullable=False)
    page_number: Mapped[int] = Column(Integer, nullable=True)
    chunk_metadata: Mapped[Dict] = Column("chunk_metadata", JSON)
    embedding: Mapped[List[float]] = Column(Vector(1536))
    created_at: Mapped[datetime] = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Phase 1: Hybrid search - tsvector for BM25 full-text search
    content_tsv = Column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))

    # Phase 3: Larger embedding model (2000 dimensions - HNSW index limit)
    embedding_large: Mapped[List[float]] = Column(Vector(2000), nullable=True)

    # Phase 4: Contextual retrieval
    contextual_summary: Mapped[str] = Column(Text, nullable=True)

    __table_args__ = (
        Index(
            'idx_chunks_embedding_hnsw',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_content_gin', 'content_tsv', postgresql_using='gin'),
        Index(
            'idx_chunks_embedding_large_hnsw',
            'embedding_large',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding_large': 'vector_cosine_ops'}
        ),
    )
