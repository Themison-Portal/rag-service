"""
gRPC Server implementation for RAG Service.
"""
import logging
from uuid import UUID

import grpc

# Import generated protobuf code
from gen.python.rag.v1 import rag_service_pb2 as pb2
from gen.python.rag.v1 import rag_service_pb2_grpc as pb2_grpc

from rag_service.db.session import get_session, check_database_connection
from rag_service.cache.semantic_cache import SemanticCacheService
from rag_service.services.ingestion_service import RagIngestionService
from rag_service.services.retrieval_service import RagRetrievalService
from rag_service.services.generation_service import RagGenerationService
from rag_service.services.highlighting_service import PDFHighlightService
from rag_service.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Map stage strings to protobuf enums
STAGE_MAP = {
    "INVALIDATING": pb2.INGEST_STAGE_STORING,
    "PREPARING": pb2.INGEST_STAGE_STORING,
    "DOWNLOADING": pb2.INGEST_STAGE_DOWNLOADING,
    "PARSING": pb2.INGEST_STAGE_PARSING,
    "CHUNKING": pb2.INGEST_STAGE_CHUNKING,
    "EMBEDDING": pb2.INGEST_STAGE_EMBEDDING,
    "STORING": pb2.INGEST_STAGE_STORING,
    "COMPLETE": pb2.INGEST_STAGE_COMPLETE,
    "ERROR": pb2.INGEST_STAGE_ERROR,
}

# Map relevance strings to protobuf enums
RELEVANCE_MAP = {
    "high": pb2.RELEVANCE_HIGH,
    "medium": pb2.RELEVANCE_MEDIUM,
    "low": pb2.RELEVANCE_LOW,
}


class RagServicer(pb2_grpc.RagServiceServicer):
    """
    gRPC servicer implementing the RagService.
    """

    def __init__(self):
        self.highlight_service = PDFHighlightService()

    async def IngestPdf(self, request, context):
        """
        Ingest PDF with streaming progress updates.
        """
        logger.info(f"IngestPdf called for document_id={request.document_id}")

        try:
            document_id = UUID(request.document_id)
            chunk_size = request.chunk_size if request.chunk_size > 0 else 750

            async with get_session() as session:
                semantic_cache = SemanticCacheService(session)
                ingestion_service = RagIngestionService(
                    db=session,
                    semantic_cache_service=semantic_cache,
                )

                async for progress in ingestion_service.ingest_pdf(
                    document_url=request.document_url,
                    document_id=document_id,
                    chunk_size=chunk_size,
                ):
                    stage = STAGE_MAP.get(progress.stage, pb2.INGEST_STAGE_UNSPECIFIED)

                    response = pb2.IngestPdfProgress(
                        stage=stage,
                        progress_percent=progress.progress,
                        message=progress.message,
                    )

                    if progress.result:
                        response.result.CopyFrom(pb2.IngestResult(
                            success=progress.result.get("success", False),
                            document_id=progress.result.get("document_id", ""),
                            status=progress.result.get("status", ""),
                            chunks_count=progress.result.get("chunks_count", 0),
                            created_at=progress.result.get("created_at", ""),
                            error=progress.result.get("error", ""),
                        ))

                    yield response

        except Exception as e:
            logger.error(f"IngestPdf error: {e}")
            yield pb2.IngestPdfProgress(
                stage=pb2.INGEST_STAGE_ERROR,
                progress_percent=0,
                message=f"Ingestion failed: {str(e)}",
                result=pb2.IngestResult(
                    success=False,
                    document_id=request.document_id,
                    status="error",
                    error=str(e),
                ),
            )

    async def Query(self, request, context):
        """
        RAG query - retrieve chunks and generate answer.
        """
        logger.info(f"Query called for document_id={request.document_id}")

        try:
            document_id = UUID(request.document_id)
            top_k = request.top_k if request.top_k > 0 else settings.retrieval_top_k
            min_score = request.min_score if request.min_score > 0 else settings.retrieval_min_score

            async with get_session() as session:
                semantic_cache = SemanticCacheService(session)
                retrieval_service = RagRetrievalService(db=session)
                generation_service = RagGenerationService(
                    retrieval_service=retrieval_service,
                    semantic_cache_service=semantic_cache,
                )

                result = await generation_service.generate_answer(
                    query_text=request.query,
                    document_id=document_id,
                    document_name=request.document_name,
                    top_k=top_k,
                    min_score=min_score,
                )

            # Build response
            answer_data = result["result"]
            timing_data = result["timing"]

            # Convert sources
            sources = []
            for s in answer_data.get("sources", []):
                bboxes = []
                for bbox in s.get("bboxes", []):
                    if len(bbox) == 4:
                        bboxes.append(pb2.BBox(
                            x0=float(bbox[0]),
                            y0=float(bbox[1]),
                            x1=float(bbox[2]),
                            y1=float(bbox[3]),
                        ))

                relevance = RELEVANCE_MAP.get(
                    s.get("relevance", "high"),
                    pb2.RELEVANCE_HIGH
                )

                sources.append(pb2.RagSource(
                    name=s.get("name", ""),
                    page=s.get("page", 0),
                    section=s.get("section", ""),
                    exact_text=s.get("exactText", ""),
                    bboxes=bboxes,
                    relevance=relevance,
                ))

            answer = pb2.RagAnswer(
                response=answer_data.get("response", ""),
                sources=sources,
            )

            retrieval = timing_data.get("retrieval", {})
            timing = pb2.QueryTiming(
                embedding_ms=timing_data.get("embedding_ms", 0),
                retrieval_ms=retrieval.get("retrieval_total_ms", 0),
                generation_ms=timing_data.get("llm_call_ms", 0),
                total_ms=timing_data.get("generation_total_ms", 0),
                chunks_retrieved=timing_data.get("original_chunk_count", 0),
                chunks_compressed=timing_data.get("compressed_chunk_count", 0),
            )

            cache_info = pb2.CacheInfo(
                embedding_cache_hit=timing_data.get("embedding_cache_hit", False),
                semantic_cache_hit=timing_data.get("semantic_cache_hit", False),
                chunk_cache_hit=retrieval.get("chunk_cache_hit", False),
                response_cache_hit=timing_data.get("response_cache_hit", False),
                semantic_similarity=timing_data.get("semantic_cache_similarity", 0),
            )

            return pb2.QueryResponse(
                answer=answer,
                timing=timing,
                cache_info=cache_info,
            )

        except Exception as e:
            logger.error(f"Query error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.QueryResponse()

    async def GetHighlightedPdf(self, request, context):
        """
        Get highlighted PDF page with bboxes.
        """
        logger.info(f"GetHighlightedPdf called for page={request.page}")

        try:
            bboxes = [
                [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
                for bbox in request.bboxes
            ]

            pdf_bytes = await self.highlight_service.get_highlighted_pdf(
                doc_url=request.document_url,
                page=request.page,
                bboxes=bboxes,
            )

            return pb2.HighlightedPdfResponse(
                pdf_content=pdf_bytes,
                content_type="application/pdf",
            )

        except Exception as e:
            logger.error(f"GetHighlightedPdf error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.HighlightedPdfResponse()

    async def InvalidateDocument(self, request, context):
        """
        Invalidate cache for a document.
        """
        logger.info(f"InvalidateDocument called for document_id={request.document_id}")

        try:
            document_id = UUID(request.document_id)

            async with get_session() as session:
                # Delete chunks
                from sqlalchemy import delete
                from rag_service.models.chunks import DocumentChunkDocling

                stmt = delete(DocumentChunkDocling).where(
                    DocumentChunkDocling.document_id == document_id
                )
                result = await session.execute(stmt)
                chunks_deleted = result.rowcount

                # Delete semantic cache entries
                semantic_cache = SemanticCacheService(session)
                cache_deleted = await semantic_cache.invalidate_document(document_id)

                await session.commit()

            return pb2.InvalidateDocumentResponse(
                success=True,
                chunks_deleted=chunks_deleted,
                cache_entries_deleted=cache_deleted,
            )

        except Exception as e:
            logger.error(f"InvalidateDocument error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.InvalidateDocumentResponse(success=False)

    async def HealthCheck(self, request, context):
        """
        Health check endpoint.
        """
        components = []

        # Check database
        db_healthy = await check_database_connection()
        components.append(pb2.ComponentHealth(
            name="database",
            healthy=db_healthy,
            message="Connected" if db_healthy else "Connection failed",
        ))

        # Overall status
        all_healthy = all(c.healthy for c in components)
        status = pb2.SERVICE_STATUS_SERVING if all_healthy else pb2.SERVICE_STATUS_NOT_SERVING

        return pb2.HealthCheckResponse(
            status=status,
            version="0.1.0",
            components=components,
        )
