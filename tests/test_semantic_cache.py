"""
Unit tests for semantic cache service.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from rag_service.cache.semantic_cache import SemanticCacheService


class TestSemanticCacheService:
    """Tests for SemanticCacheService."""

    @pytest.fixture
    def cache_service(self, mock_db_session):
        """Create cache service with mock session."""
        return SemanticCacheService(db=mock_db_session, similarity_threshold=0.90)

    def test_embedding_to_pg_vector(self, cache_service):
        """Test embedding conversion to PostgreSQL vector format."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        result = cache_service._embedding_to_pg_vector(embedding)
        assert result == "[0.1,0.2,0.3,0.4]"

    def test_hash_context(self):
        """Test context hashing for cache invalidation."""
        chunks = [
            {"page_content": "chunk 1"},
            {"page_content": "chunk 2"}
        ]

        hash1 = SemanticCacheService.hash_context(chunks)
        hash2 = SemanticCacheService.hash_context(chunks)

        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # SHA256 truncated to 32 chars

    def test_hash_context_different_content(self):
        """Test that different content produces different hashes."""
        chunks1 = [{"page_content": "chunk 1"}]
        chunks2 = [{"page_content": "chunk 2"}]

        hash1 = SemanticCacheService.hash_context(chunks1)
        hash2 = SemanticCacheService.hash_context(chunks2)

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_get_similar_response_hit(self, cache_service, mock_db_session):
        """Test semantic cache hit returns cached response."""
        document_id = uuid4()
        query_embedding = [0.1] * 1536

        # Mock database response for cache hit
        mock_row = MagicMock()
        mock_row.id = uuid4()
        mock_row.query_text = "similar query"
        mock_row.response_data = {"response": "cached answer", "sources": []}
        mock_row.context_hash = "abc123"
        mock_row.similarity = 0.95

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_db_session.execute.return_value = mock_result

        result = await cache_service.get_similar_response(
            query_embedding=query_embedding,
            document_id=document_id
        )

        assert result is not None
        assert result["similarity"] == 0.95
        assert result["response"]["response"] == "cached answer"
        assert result["original_query"] == "similar query"

    @pytest.mark.asyncio
    async def test_get_similar_response_miss(self, cache_service, mock_db_session):
        """Test semantic cache miss returns None."""
        document_id = uuid4()
        query_embedding = [0.1] * 1536

        # Mock database response for cache miss
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await cache_service.get_similar_response(
            query_embedding=query_embedding,
            document_id=document_id
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_similar_response_custom_threshold(self, cache_service, mock_db_session):
        """Test semantic cache with custom similarity threshold."""
        document_id = uuid4()
        query_embedding = [0.1] * 1536

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        await cache_service.get_similar_response(
            query_embedding=query_embedding,
            document_id=document_id,
            similarity_threshold=0.95
        )

        # Verify the query was executed with custom threshold
        mock_db_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_response(self, cache_service, mock_db_session):
        """Test storing a response in semantic cache."""
        document_id = uuid4()
        query_text = "test query"
        query_embedding = [0.1] * 1536
        response = {"response": "test answer", "sources": []}
        context_hash = "abc123"

        await cache_service.store_response(
            query_text=query_text,
            query_embedding=query_embedding,
            document_id=document_id,
            response=response,
            context_hash=context_hash
        )

        # Verify add and commit were called
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_response_handles_error(self, cache_service, mock_db_session):
        """Test that store errors are handled gracefully."""
        mock_db_session.commit.side_effect = Exception("DB Error")

        # Should not raise
        await cache_service.store_response(
            query_text="test",
            query_embedding=[0.1] * 1536,
            document_id=uuid4(),
            response={},
            context_hash="hash"
        )

        mock_db_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_document(self, cache_service, mock_db_session):
        """Test cache invalidation for a document."""
        document_id = uuid4()

        # Mock deletion result
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_db_session.execute.return_value = mock_result

        deleted = await cache_service.invalidate_document(document_id)

        assert deleted == 5
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_document_no_entries(self, cache_service, mock_db_session):
        """Test invalidation when no cache entries exist."""
        document_id = uuid4()

        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db_session.execute.return_value = mock_result

        deleted = await cache_service.invalidate_document(document_id)

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_invalidate_document_handles_error(self, cache_service, mock_db_session):
        """Test that invalidation errors are handled gracefully."""
        mock_db_session.execute.side_effect = Exception("DB Error")

        deleted = await cache_service.invalidate_document(uuid4())

        assert deleted == 0
        mock_db_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_cache_hit(self, cache_service, mock_db_session):
        """Test that cache hit updates hit count."""
        cache_id = uuid4()

        await cache_service._update_cache_hit(cache_id)

        mock_db_session.execute.assert_called_once()
        mock_db_session.commit.assert_called_once()
