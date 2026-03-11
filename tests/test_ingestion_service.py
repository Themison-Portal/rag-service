"""
Unit tests for RAG ingestion service.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from rag_service.services.ingestion_service import RagIngestionService, IngestionProgress


class TestIngestionProgress:
    """Tests for IngestionProgress dataclass."""

    def test_progress_creation(self):
        """Test creating an ingestion progress object."""
        progress = IngestionProgress(
            stage="EMBEDDING",
            progress=50,
            message="Processing embeddings"
        )

        assert progress.stage == "EMBEDDING"
        assert progress.progress == 50
        assert progress.message == "Processing embeddings"
        assert progress.result is None

    def test_progress_with_result(self):
        """Test progress with result dict."""
        result = {"success": True, "chunks_count": 10}
        progress = IngestionProgress(
            stage="COMPLETE",
            progress=100,
            message="Done",
            result=result
        )

        assert progress.result == result


class TestRagIngestionService:
    """Tests for RagIngestionService."""

    @pytest.fixture
    def mock_semantic_cache(self):
        """Create mock semantic cache service."""
        cache = AsyncMock()
        cache.invalidate_document = AsyncMock(return_value=5)
        return cache

    @pytest.fixture
    def ingestion_service(self, mock_db_session, mock_semantic_cache, mock_embedding_client):
        """Create ingestion service with mocks."""
        with patch("rag_service.services.ingestion_service.get_embedding_client", return_value=mock_embedding_client):
            service = RagIngestionService(
                db=mock_db_session,
                semantic_cache_service=mock_semantic_cache
            )
            service.embedding_client = mock_embedding_client
            return service

    def test_extract_docling_citation_metadata(self, ingestion_service):
        """Test extraction of citation metadata from Docling format."""
        metadata = {
            "dl_meta": {
                "doc_items": [{
                    "prov": [{"page_no": 5}]
                }],
                "headings": ["Chapter 1", "Section 1.1"]
            }
        }

        result = ingestion_service._extract_docling_citation_metadata(metadata)

        assert result["page_number"] == 5
        assert result["headings"] == ["Chapter 1", "Section 1.1"]

    def test_extract_docling_citation_metadata_empty(self, ingestion_service):
        """Test extraction with empty metadata."""
        result = ingestion_service._extract_docling_citation_metadata({})

        assert result["page_number"] is None
        assert result["headings"] == []

    def test_extract_docling_citation_metadata_malformed(self, ingestion_service):
        """Test extraction with malformed metadata."""
        metadata = {"dl_meta": {"doc_items": []}}  # Empty doc_items

        result = ingestion_service._extract_docling_citation_metadata(metadata)

        assert result["page_number"] is None

    @pytest.mark.asyncio
    async def test_delete_existing_chunks(self, ingestion_service, mock_db_session):
        """Test deletion of existing chunks."""
        document_id = uuid4()

        mock_result = MagicMock()
        mock_result.rowcount = 10
        mock_db_session.execute.return_value = mock_result

        deleted = await ingestion_service._delete_existing_chunks(document_id)

        assert deleted == 10
        mock_db_session.execute.assert_called_once()
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_docling_chunks(self, ingestion_service, mock_db_session):
        """Test insertion of document chunks."""
        document_id = uuid4()

        # Create mock Document objects
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"dl_meta": {"doc_items": [{"prov": [{"page_no": 1}]}]}}

        chunks = [mock_doc]
        embeddings = [[0.1] * 1536]

        await ingestion_service._insert_docling_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings
        )

        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_docling_chunks_with_contextual(self, ingestion_service, mock_db_session):
        """Test insertion with contextual summaries."""
        document_id = uuid4()

        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {}

        chunks = [mock_doc]
        embeddings = [[0.1] * 1536]
        summaries = ["This chunk discusses clinical trial methodology."]

        await ingestion_service._insert_docling_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            contextual_summaries=summaries
        )

        mock_db_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_docling_chunks_handles_error(self, ingestion_service, mock_db_session):
        """Test that insertion errors are handled."""
        mock_db_session.commit.side_effect = Exception("DB Error")

        with pytest.raises(RuntimeError, match="Failed to insert chunks"):
            await ingestion_service._insert_docling_chunks(
                document_id=uuid4(),
                chunks=[MagicMock(page_content="test", metadata={})],
                embeddings=[[0.1] * 1536]
            )

        mock_db_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_pdf_progress_stages(self, ingestion_service, mock_db_session, mock_semantic_cache):
        """Test that ingestion yields correct progress stages."""
        document_id = uuid4()

        # Mock the delete operation
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db_session.execute.return_value = mock_result

        # Mock Docling loader
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {}

        with patch("langchain_docling.loader.DoclingLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [mock_doc]
            mock_loader_class.return_value = mock_loader

            with patch("docling.chunking.HybridChunker"):
                with patch("transformers.AutoTokenizer"):
                    stages = []
                    async for progress in ingestion_service.ingest_pdf(
                        document_url="https://example.com/test.pdf",
                        document_id=document_id,
                        chunk_size=750
                    ):
                        stages.append(progress.stage)

        # Verify we get the expected stages
        assert "INVALIDATING" in stages
        assert "PREPARING" in stages
        assert "DOWNLOADING" in stages
        assert "PARSING" in stages
        assert "EMBEDDING" in stages
        assert "STORING" in stages
        assert "COMPLETE" in stages

    @pytest.mark.asyncio
    async def test_ingest_pdf_error_handling(self, ingestion_service, mock_semantic_cache):
        """Test that ingestion errors yield ERROR stage."""
        document_id = uuid4()

        # Make cache invalidation fail
        mock_semantic_cache.invalidate_document.side_effect = Exception("Cache error")

        stages = []
        async for progress in ingestion_service.ingest_pdf(
            document_url="https://example.com/test.pdf",
            document_id=document_id
        ):
            stages.append(progress.stage)
            if progress.result and not progress.result.get("success"):
                assert "error" in progress.result

        assert "ERROR" in stages

    @pytest.mark.asyncio
    async def test_ingest_pdf_returns_result(self, ingestion_service, mock_db_session, mock_semantic_cache):
        """Test that successful ingestion returns proper result."""
        document_id = uuid4()

        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db_session.execute.return_value = mock_result

        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {}

        with patch("langchain_docling.loader.DoclingLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.load.return_value = [mock_doc]
            mock_loader_class.return_value = mock_loader

            with patch("docling.chunking.HybridChunker"):
                with patch("transformers.AutoTokenizer"):
                    final_result = None
                    async for progress in ingestion_service.ingest_pdf(
                        document_url="https://example.com/test.pdf",
                        document_id=document_id
                    ):
                        if progress.stage == "COMPLETE":
                            final_result = progress.result

        assert final_result is not None
        assert final_result["success"] is True
        assert final_result["document_id"] == str(document_id)
        assert final_result["chunks_count"] == 1
