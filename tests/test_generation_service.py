"""
Unit tests for RAG generation service.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from rag_service.services.generation_service import RagGenerationService


class TestRagGenerationService:
    """Tests for RagGenerationService."""

    @pytest.fixture
    def mock_retrieval_service(self, sample_chunks):
        """Create mock retrieval service."""
        service = AsyncMock()
        service.get_query_embedding = AsyncMock(return_value=([0.1] * 1536, {"embedding_ms": 10}))
        service.retrieve_similar_chunks = AsyncMock(return_value=(sample_chunks, {"retrieval_total_ms": 50}))
        return service

    @pytest.fixture
    def mock_semantic_cache(self):
        """Create mock semantic cache service."""
        cache = AsyncMock()
        cache.get_similar_response = AsyncMock(return_value=None)
        cache.store_response = AsyncMock()
        cache.hash_context = MagicMock(return_value="test_hash")
        return cache

    @pytest.fixture
    def generation_service(self, mock_retrieval_service, mock_semantic_cache):
        """Create generation service with mocks."""
        return RagGenerationService(
            retrieval_service=mock_retrieval_service,
            semantic_cache_service=mock_semantic_cache
        )

    def test_extract_chunk_metadata(self, generation_service, sample_chunks):
        """Test metadata extraction from chunks."""
        meta = generation_service._extract_chunk_metadata(sample_chunks[0])

        assert meta["title"] == "Test Document"
        assert meta["page"] == 1
        assert meta["section"] == "Introduction"
        assert meta["bbox"] == [10, 20, 100, 50]
        assert "clinical trials" in meta["content"]

    def test_extract_chunk_metadata_missing_fields(self, generation_service):
        """Test metadata extraction with missing fields."""
        chunk = {
            "page_content": "Content",
            "metadata": {}
        }

        meta = generation_service._extract_chunk_metadata(chunk)

        assert meta["title"] == "Unknown"
        assert meta["page"] == 0
        assert meta["section"] is None
        assert meta["bbox"] is None

    def test_compress_chunks_merges_same_page(self, generation_service):
        """Test that chunks from the same page are merged."""
        chunks = [
            {
                "page_content": "First chunk",
                "metadata": {"title": "Doc", "page": 1, "docling": {"dl_meta": {}}}
            },
            {
                "page_content": "Second chunk",
                "metadata": {"title": "Doc", "page": 1, "docling": {"dl_meta": {}}}
            },
            {
                "page_content": "Third chunk",
                "metadata": {"title": "Doc", "page": 2, "docling": {"dl_meta": {}}}
            }
        ]

        compressed = generation_service._compress_chunks(chunks)

        assert len(compressed) == 2  # Two unique pages
        # Find the merged chunk (page 1)
        page1_chunk = next(c for c in compressed if c["page"] == 1)
        assert "merged_count" in page1_chunk
        assert page1_chunk["merged_count"] == 2

    def test_compress_chunks_empty(self, generation_service):
        """Test compression of empty chunks list."""
        assert generation_service._compress_chunks([]) == []

    def test_format_context_compact(self, generation_service):
        """Test compact context formatting."""
        chunk_meta = {
            "title": "Test Doc",
            "page": 5,
            "bbox": [10, 20, 100, 50],
            "content": "Sample content"
        }

        formatted = generation_service._format_context_compact(chunk_meta)

        assert "[Test Doc|p5|bbox:" in formatted
        assert "Sample content" in formatted

    def test_repair_json_trailing_comma(self, generation_service):
        """Test JSON repair removes trailing commas."""
        broken_json = '{"key": "value",}'
        repaired = generation_service._repair_json(broken_json)
        assert repaired == '{"key": "value"}'

    def test_parse_llm_json_valid(self, generation_service):
        """Test parsing valid JSON response."""
        json_str = '{"response": "Answer", "sources": []}'
        result = generation_service._parse_llm_json(json_str)

        assert result["response"] == "Answer"
        assert result["sources"] == []

    def test_parse_llm_json_with_extra_text(self, generation_service):
        """Test parsing JSON with surrounding text."""
        json_str = 'Here is my response:\n{"response": "Answer", "sources": []}\nDone.'
        result = generation_service._parse_llm_json(json_str)

        assert result["response"] == "Answer"

    def test_parse_llm_json_malformed(self, generation_service):
        """Test parsing malformed JSON falls back gracefully."""
        json_str = '{"response": "Answer with missing quote}'
        result = generation_service._parse_llm_json(json_str)

        # Should return some response, not raise
        assert "response" in result
        assert "sources" in result

    @pytest.mark.asyncio
    async def test_generate_answer_semantic_cache_hit(
        self, mock_retrieval_service, mock_semantic_cache, sample_query_response
    ):
        """Test that semantic cache hit returns cached response."""
        # Configure cache to return a hit
        mock_semantic_cache.get_similar_response = AsyncMock(return_value={
            "response": sample_query_response,
            "similarity": 0.95,
            "original_query": "cached query"
        })

        service = RagGenerationService(
            retrieval_service=mock_retrieval_service,
            semantic_cache_service=mock_semantic_cache
        )

        result = await service.generate_answer(
            query_text="test query",
            document_id=uuid4(),
            document_name="Test Doc"
        )

        assert result["timing"]["semantic_cache_hit"] is True
        assert result["timing"]["semantic_cache_similarity"] == 0.95
        # Retrieval should not be called on cache hit
        mock_retrieval_service.retrieve_similar_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_answer_no_chunks(self, mock_retrieval_service, mock_semantic_cache):
        """Test response when no relevant chunks are found."""
        mock_retrieval_service.retrieve_similar_chunks = AsyncMock(
            return_value=([], {"retrieval_total_ms": 10})
        )

        service = RagGenerationService(
            retrieval_service=mock_retrieval_service,
            semantic_cache_service=mock_semantic_cache
        )

        result = await service.generate_answer(
            query_text="irrelevant query",
            document_id=uuid4(),
            document_name="Test Doc"
        )

        assert "do not contain" in result["result"]["response"].lower()
        assert result["result"]["sources"] == []

    @pytest.mark.asyncio
    async def test_generate_answer_calls_llm(
        self, mock_retrieval_service, mock_semantic_cache, sample_chunks
    ):
        """Test that LLM is called when cache misses."""
        mock_retrieval_service.retrieve_similar_chunks = AsyncMock(
            return_value=(sample_chunks, {"retrieval_total_ms": 50})
        )

        service = RagGenerationService(
            retrieval_service=mock_retrieval_service,
            semantic_cache_service=mock_semantic_cache
        )

        # Mock the Anthropic client
        with patch("rag_service.services.generation_service.get_anthropic_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(
                text='{"response": "Generated answer", "sources": [{"name": "Doc", "page": 1, "exactText": "text", "bboxes": [[1,2,3,4]], "relevance": "high"}]}'
            )]
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await service.generate_answer(
                query_text="test query",
                document_id=uuid4(),
                document_name="Test Doc"
            )

        assert result["result"]["response"] == "Generated answer"
        assert len(result["result"]["sources"]) == 1
        assert "llm_call_ms" in result["timing"]

    @pytest.mark.asyncio
    async def test_generate_answer_stores_in_cache(
        self, mock_retrieval_service, mock_semantic_cache, sample_chunks
    ):
        """Test that generated answers are stored in semantic cache."""
        mock_retrieval_service.retrieve_similar_chunks = AsyncMock(
            return_value=(sample_chunks, {"retrieval_total_ms": 50})
        )

        service = RagGenerationService(
            retrieval_service=mock_retrieval_service,
            semantic_cache_service=mock_semantic_cache
        )

        with patch("rag_service.services.generation_service.get_anthropic_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"response": "Answer", "sources": []}')]
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await service.generate_answer(
                query_text="test query",
                document_id=uuid4(),
                document_name="Test Doc"
            )

        # Verify cache store was called
        mock_semantic_cache.store_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_answer_handles_llm_error(
        self, mock_retrieval_service, mock_semantic_cache, sample_chunks
    ):
        """Test graceful handling of LLM errors."""
        mock_retrieval_service.retrieve_similar_chunks = AsyncMock(
            return_value=(sample_chunks, {"retrieval_total_ms": 50})
        )

        service = RagGenerationService(
            retrieval_service=mock_retrieval_service,
            semantic_cache_service=mock_semantic_cache
        )

        with patch("rag_service.services.generation_service.get_anthropic_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
            mock_get_client.return_value = mock_client

            result = await service.generate_answer(
                query_text="test query",
                document_id=uuid4(),
                document_name="Test Doc"
            )

        assert "error" in result["result"]["response"].lower()
        assert "error" in result["timing"]
