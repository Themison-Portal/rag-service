"""
Unit tests for PDF highlighting service.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_service.services.highlighting_service import PDFHighlightService


class TestPDFHighlightService:
    """Tests for PDFHighlightService."""

    @pytest.fixture
    def highlight_service(self):
        """Create highlighting service."""
        return PDFHighlightService()

    @pytest.mark.asyncio
    async def test_get_pdf_from_url(self, highlight_service):
        """Test downloading PDF from URL."""
        pdf_content = b"%PDF-1.4 test content"

        with patch("rag_service.services.highlighting_service.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = pdf_content
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch("rag_service.services.highlighting_service.fitz.open") as mock_fitz_open:
                mock_doc = MagicMock()
                mock_fitz_open.return_value = mock_doc

                result = await highlight_service._get_pdf_from_url("https://example.com/test.pdf")

                mock_client.get.assert_called_once_with("https://example.com/test.pdf")
                mock_fitz_open.assert_called_once_with(stream=pdf_content, filetype="pdf")
                assert result == mock_doc

    @pytest.mark.asyncio
    async def test_get_highlighted_pdf_success(self, highlight_service):
        """Test successful PDF highlighting."""
        bboxes = [[10, 20, 100, 50], [15, 60, 90, 80]]

        with patch.object(highlight_service, "_get_pdf_from_url") as mock_get_pdf:
            # Create mock PDF document
            mock_page = MagicMock()
            mock_page.rect.height = 800

            # Mock text blocks
            mock_page.get_text.return_value = [
                (10, 700, 100, 750, "text", 0, 0),  # Intersects with bbox
            ]

            # Mock annotation
            mock_annot = MagicMock()
            mock_page.add_highlight_annot.return_value = mock_annot

            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=5)
            mock_doc.__getitem__ = MagicMock(return_value=mock_page)
            mock_doc.tobytes.return_value = b"highlighted PDF"
            mock_doc.close = MagicMock()

            mock_get_pdf.return_value = mock_doc

            result = await highlight_service.get_highlighted_pdf(
                doc_url="https://example.com/test.pdf",
                page=1,
                bboxes=bboxes
            )

            assert result == b"highlighted PDF"
            mock_page.add_highlight_annot.assert_called()
            mock_annot.update.assert_called()
            mock_doc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_highlighted_pdf_invalid_page(self, highlight_service):
        """Test highlighting with invalid page number."""
        with patch.object(highlight_service, "_get_pdf_from_url") as mock_get_pdf:
            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=3)
            mock_doc.close = MagicMock()
            mock_get_pdf.return_value = mock_doc

            with pytest.raises(ValueError, match="out of range"):
                await highlight_service.get_highlighted_pdf(
                    doc_url="https://example.com/test.pdf",
                    page=10,  # Page doesn't exist
                    bboxes=[[10, 20, 100, 50]]
                )

            mock_doc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_highlighted_pdf_no_bboxes(self, highlight_service):
        """Test highlighting with empty bboxes."""
        with patch.object(highlight_service, "_get_pdf_from_url") as mock_get_pdf:
            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=3)
            mock_doc.close = MagicMock()
            mock_get_pdf.return_value = mock_doc

            with pytest.raises(ValueError, match="No bboxes"):
                await highlight_service.get_highlighted_pdf(
                    doc_url="https://example.com/test.pdf",
                    page=1,
                    bboxes=[]
                )

    @pytest.mark.asyncio
    async def test_get_highlighted_pdf_skips_invalid_bbox(self, highlight_service):
        """Test that invalid bboxes are skipped."""
        bboxes = [
            [10, 20, 100, 50],  # Valid
            [1, 2],  # Invalid - only 2 values
            None,  # Invalid - None
        ]

        with patch.object(highlight_service, "_get_pdf_from_url") as mock_get_pdf:
            mock_page = MagicMock()
            mock_page.rect.height = 800
            mock_page.get_text.return_value = []

            mock_annot = MagicMock()
            mock_page.add_highlight_annot.return_value = mock_annot

            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=5)
            mock_doc.__getitem__ = MagicMock(return_value=mock_page)
            mock_doc.tobytes.return_value = b"PDF"
            mock_doc.close = MagicMock()

            mock_get_pdf.return_value = mock_doc

            # Should not raise, should skip invalid bboxes
            result = await highlight_service.get_highlighted_pdf(
                doc_url="https://example.com/test.pdf",
                page=1,
                bboxes=bboxes
            )

            assert result == b"PDF"

    @pytest.mark.asyncio
    async def test_get_highlighted_pdf_coordinate_conversion(self, highlight_service):
        """Test that coordinates are properly converted from Docling to PDF format."""
        # Docling uses top-left origin, PDF uses bottom-left origin
        bboxes = [[10, 20, 100, 50]]  # x0, y0, x1, y1

        with patch.object(highlight_service, "_get_pdf_from_url") as mock_get_pdf:
            mock_page = MagicMock()
            mock_page.rect.height = 800  # Page height
            mock_page.get_text.return_value = []

            # Capture the Rect that's created
            captured_rects = []

            def capture_rect(rect):
                captured_rects.append(rect)
                mock_annot = MagicMock()
                return mock_annot

            mock_page.add_highlight_annot.side_effect = capture_rect

            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=5)
            mock_doc.__getitem__ = MagicMock(return_value=mock_page)
            mock_doc.tobytes.return_value = b"PDF"
            mock_doc.close = MagicMock()

            mock_get_pdf.return_value = mock_doc

            with patch("rag_service.services.highlighting_service.fitz.Rect") as mock_rect:
                mock_rect.return_value = MagicMock(is_empty=False, is_infinite=False)

                await highlight_service.get_highlighted_pdf(
                    doc_url="https://example.com/test.pdf",
                    page=1,
                    bboxes=bboxes
                )

                # Verify Rect was called with converted coordinates
                # y coordinates should be: page_height - y
                mock_rect.assert_called()
                call_args = mock_rect.call_args[0]
                assert call_args[0] == 10  # x0
                assert call_args[2] == 100  # x1
                # y coordinates are flipped
                assert call_args[1] == 800 - 50  # page_height - y1
                assert call_args[3] == 800 - 20  # page_height - y0

    @pytest.mark.asyncio
    async def test_get_highlighted_pdf_closes_doc_on_error(self, highlight_service):
        """Test that PDF document is closed even on error."""
        with patch.object(highlight_service, "_get_pdf_from_url") as mock_get_pdf:
            mock_page = MagicMock()
            mock_page.rect.height = 800
            mock_page.get_text.side_effect = Exception("PDF error")

            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=5)
            mock_doc.__getitem__ = MagicMock(return_value=mock_page)
            mock_doc.close = MagicMock()

            mock_get_pdf.return_value = mock_doc

            with pytest.raises(Exception):
                await highlight_service.get_highlighted_pdf(
                    doc_url="https://example.com/test.pdf",
                    page=1,
                    bboxes=[[10, 20, 100, 50]]
                )

            # Document should still be closed
            mock_doc.close.assert_called_once()
