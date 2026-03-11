"""
PDF Highlighting Service - Generate highlighted PDF pages.
"""
import logging
from typing import List

import fitz  # PyMuPDF
import httpx

logger = logging.getLogger(__name__)


class PDFHighlightService:
    """Service for generating highlighted PDF pages with bboxes."""

    async def _get_pdf_from_url(self, url: str) -> fitz.Document:
        """Download PDF from URL."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            pdf_bytes = resp.content
            return fitz.open(stream=pdf_bytes, filetype="pdf")

    async def get_highlighted_pdf(
        self,
        doc_url: str,
        page: int,
        bboxes: List[List[float]],
    ) -> bytes:
        """
        Generate a highlighted PDF page.

        Args:
            doc_url: URL of the PDF document.
            page: Page number (1-indexed).
            bboxes: List of bounding boxes [[x0, y0, x1, y1], ...].

        Returns:
            PDF bytes with highlights.
        """
        doc_pdf = await self._get_pdf_from_url(doc_url)

        try:
            if page < 1 or page > len(doc_pdf):
                raise ValueError(f"Page {page} out of range")

            page_obj = doc_pdf[page - 1]
            page_height = page_obj.rect.height

            if not bboxes:
                raise ValueError("No bboxes provided for highlighting")

            for bbox in bboxes:
                if not bbox or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = map(float, bbox)

                # Normalize bbox
                x0, x1 = sorted([x0, x1])
                y0, y1 = sorted([y0, y1])

                # Convert Docling (top-left) to PDF (bottom-left) coordinates
                target_rect = fitz.Rect(
                    x0,
                    page_height - y1,
                    x1,
                    page_height - y0,
                )

                if target_rect.is_empty or target_rect.is_infinite:
                    continue

                # Smart highlight: expand to text blocks if overlapping
                blocks = page_obj.get_text("blocks")
                intersecting_blocks = [
                    fitz.Rect(b[:4])
                    for b in blocks
                    if target_rect.intersects(fitz.Rect(b[:4]))
                ]

                if intersecting_blocks:
                    for block_rect in intersecting_blocks:
                        annot = page_obj.add_highlight_annot(block_rect)
                        annot.update()
                else:
                    annot = page_obj.add_highlight_annot(target_rect)
                    annot.update()

            # Serialize PDF
            pdf_bytes = doc_pdf.tobytes(garbage=3, clean=True, deflate=True)
            return pdf_bytes

        finally:
            doc_pdf.close()
