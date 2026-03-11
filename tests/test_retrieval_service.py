"""
Unit tests for RAG retrieval service.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from rag_service.services.retrieval_service import RagRetrievalService


class TestRagRetrievalService:
    """Tests for RagRetrievalService."""

    @pytest.fixture
    def retrieval_service(self, mock_db_session, mock_embedding_client):
        """Create retrieval service with mocks."""
        with patch("rag_service.services.retrieval_service.get_embedding_client", return_value=mock_embedding_client):
            service = RagRetrievalService(db=mock_db_session)
            service.embedding_client = mock_embedding_client
            return service

    def test_embedding_to_pg_vector(self, retrieval_service):
        """Test conversion of embedding list to PostgreSQL vector format."""
        embedding = [0.1, 0.2, 0.3]
        result = retrieval_service._embedding_to_pg_vector(embedding)
        assert result == "[0.1,0.2,0.3]"

    def test_embedding_to_pg_vector_empty(self, retrieval_service):
        """Test conversion of empty embedding."""
        embedding = []
        result = retrieval_service._embedding_to_pg_vector(embedding)
        assert result == "[]"

    @pytest.mark.asyncio
    async def test_get_query_embedding(self, retrieval_service, mock_embedding_client):
        """Test query embedding generation."""
        query = "What are clinical trials?"

        embedding, timing = await retrieval_service.get_query_embedding(query)

        assert len(embedding) == 1536
        assert "embedding_ms" in timing
        mock_embedding_client.aembed_query.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_search_similar_chunks_docling(self, retrieval_service, mock_db_session, sample_document_id):
        """Test vector similarity search."""
        # Mock database response
        mock_row = MagicMock()
        mock_row.content = "Test content"
        mock_row.page_number = 1
        mock_row.chunk_metadata = {"test": "data"}
        mock_row.similarity = 0.85

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_db_session.execute.return_value = mock_result

        docs, timing = await retrieval_service._search_similar_chunks_docling(
            query_text="test query",
            document_id=sample_document_id,
            document_name="Test Document",
            top_k=10,
            precomputed_embedding=[0.1] * 1536
        )

        assert len(docs) == 1
        assert docs[0]["page_content"] == "Test content"
        assert docs[0]["score"] == 0.85
        assert docs[0]["metadata"]["title"] == "Test Document"
        assert "db_search_ms" in timing

    @pytest.mark.asyncio
    async def test_search_bm25(self, retrieval_service, mock_db_session, sample_document_id):
        """Test BM25 full-text search."""
        # Mock database response
        mock_row = MagicMock()
        mock_row.id = uuid4()
        mock_row.content = "BM25 content"
        mock_row.page_number = 2
        mock_row.chunk_metadata = {"test": "bm25"}
        mock_row.bm25_score = 0.9

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_db_session.execute.return_value = mock_result

        docs = await retrieval_service._search_bm25(
            query_text="test query",
            document_id=sample_document_id,
            document_name="Test Document",
            top_k=10
        )

        assert len(docs) == 1
        assert docs[0]["page_content"] == "BM25 content"
        assert docs[0]["score"] == 0.9

    def test_reciprocal_rank_fusion(self, retrieval_service):
        """Test RRF fusion of vector and BM25 results."""
        vector_results = [
            {"page_content": "doc1", "score": 0.9, "id": "1"},
            {"page_content": "doc2", "score": 0.8, "id": "2"},
        ]
        bm25_results = [
            {"page_content": "doc2", "score": 0.95, "id": "2"},
            {"page_content": "doc3", "score": 0.7, "id": "3"},
        ]

        fused = retrieval_service._reciprocal_rank_fusion(vector_results, bm25_results, k=60)

        # doc2 should be ranked higher (appears in both)
        assert len(fused) == 3
        assert fused[0]["page_content"] == "doc2"  # Highest RRF score
        assert "rrf_score" in fused[0]

    def test_reciprocal_rank_fusion_empty(self, retrieval_service):
        """Test RRF with empty results."""
        fused = retrieval_service._reciprocal_rank_fusion([], [], k=60)
        assert fused == []

    @pytest.mark.asyncio
    async def test_retrieve_similar_chunks_hybrid(self, retrieval_service, mock_db_session, sample_document_id):
        """Test hybrid retrieval with both vector and BM25."""
        # Mock vector search response
        mock_row = MagicMock()
        mock_row.content = "Test content"
        mock_row.page_number = 1
        mock_row.chunk_metadata = {}
        mock_row.similarity = 0.85

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_db_session.execute.return_value = mock_result

        with patch.object(retrieval_service, "_search_bm25", return_value=[]):
            chunks, timing = await retrieval_service.retrieve_similar_chunks(
                query_text="test query",
                document_id=sample_document_id,
                document_name="Test Document",
                precomputed_embedding=[0.1] * 1536
            )

        assert "retrieval_total_ms" in timing

    @pytest.mark.asyncio
    async def test_retrieve_similar_chunks_filters_by_score(self, retrieval_service, mock_db_session, sample_document_id):
        """Test that chunks are filtered by minimum score."""
        # Create chunks with different scores
        mock_rows = [
            MagicMock(content="High score", page_number=1, chunk_metadata={}, similarity=0.9),
            MagicMock(content="Low score", page_number=2, chunk_metadata={}, similarity=0.01),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_db_session.execute.return_value = mock_result

        # Disable hybrid search to test vector-only filtering
        with patch("rag_service.services.retrieval_service.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = False
            mock_settings.retrieval_top_k = 20
            mock_settings.retrieval_min_score = 0.04

            chunks, _ = await retrieval_service.retrieve_similar_chunks(
                query_text="test",
                document_id=sample_document_id,
                document_name="Test",
                min_score=0.5,
                precomputed_embedding=[0.1] * 1536
            )

        # Only high score chunk should pass
        assert len(chunks) == 1
        assert chunks[0]["page_content"] == "High score"
