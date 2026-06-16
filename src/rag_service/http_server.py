"""
FastAPI HTTP server for RAG Service — wire-compatible alternative to gRPC.

Each endpoint mirrors one RPC on `RagServicer` (server.py). Both transports
share the same internal services (ingestion_service, retrieval_service,
generation_service, highlighting_service) so business logic lives in exactly
one place.

JSON wire format matches the dict shape the BE's `RagClient` already returns
to its callers — so the BE HTTP client can deserialize each response straight
into the same dict the gRPC client yields.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_service.cache.semantic_cache import SemanticCacheService
from rag_service.clients.anthropic_client import generate_text
from rag_service.config import get_settings
from rag_service.db.session import check_database_connection, get_session
from rag_service.services.generation_service import RagGenerationService
from rag_service.services.highlighting_service import PDFHighlightService
from rag_service.services.ingestion_service import RagIngestionService
from rag_service.services.retrieval_service import RagRetrievalService

logger = logging.getLogger(__name__)

# Serialize PDF ingestion to one document at a time. Each docling pipeline loads
# large layout + OCR + table torch models; running several concurrently exceeds
# the instance memory and OOM-kills the service. Concurrent/duplicate uploads
# queue on this semaphore instead of running in parallel. This caps concurrency
# only — no docling feature is disabled.
_INGEST_SEMAPHORE = asyncio.Semaphore(1)


class IngestPdfBody(BaseModel):
    document_url: str
    document_id: str
    chunk_size: int = 750


class GenerateMessage(BaseModel):
    role: str
    content: str


class GenerateBody(BaseModel):
    messages: List[GenerateMessage]
    model_hint: Optional[str] = ""
    temperature: float = 0.0
    max_tokens: Optional[int] = 0
    feature_tag: str = ""
    response_schema_json: str = ""
    response_schema_name: str = "structured_output"


class QueryBody(BaseModel):
    query: str
    document_id: str
    document_name: str
    top_k: int = 0
    min_score: float = 0.0


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class HighlightBody(BaseModel):
    document_url: str
    page: int
    bboxes: List[BBox] = Field(default_factory=list)


class InvalidateBody(BaseModel):
    document_id: str


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="RAG Service (HTTP)", version="0.1.0")
    highlight_service = PDFHighlightService()

    @app.get("/healthz")
    async def health():
        db_healthy = await check_database_connection()
        components = [
            {
                "name": "database",
                "healthy": db_healthy,
                "message": "Connected" if db_healthy else "Connection failed",
            }
        ]
        all_healthy = all(c["healthy"] for c in components)
        return {
            "status": "SERVING" if all_healthy else "NOT_SERVING",
            "version": "0.1.0",
            "components": components,
        }

    @app.get("/")
    async def root():
        return {"service": "rag-service", "transport": "http"}

    @app.post("/v1/generate")
    async def generate(body: GenerateBody):
        feature_tag = body.feature_tag or "(none)"
        logger.info(
            "HTTP /generate feature=%s model_hint=%s msgs=%d structured=%s",
            feature_tag,
            body.model_hint or "(default)",
            len(body.messages),
            bool(body.response_schema_json),
        )

        response_schema = None
        if body.response_schema_json:
            try:
                response_schema = json.loads(body.response_schema_json)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"response_schema_json is not valid JSON: {e}",
                )

        try:
            result = await generate_text(
                messages=[m.model_dump() for m in body.messages],
                model_hint=body.model_hint or None,
                temperature=float(body.temperature) if body.temperature else 0.0,
                max_tokens=int(body.max_tokens) if body.max_tokens else None,
                response_schema=response_schema,
                response_schema_name=body.response_schema_name or "structured_output",
                feature_tag=feature_tag,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("HTTP /generate error (feature=%s)", feature_tag)
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "content": result["content"],
            "model": result["model"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
        }

    @app.post("/v1/query")
    async def query(body: QueryBody):
        logger.info("HTTP /query document_id=%s", body.document_id)
        try:
            document_id = UUID(body.document_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid document_id: {e}")

        top_k = body.top_k if body.top_k > 0 else settings.retrieval_top_k
        min_score = (
            body.min_score if body.min_score > 0 else settings.retrieval_min_score
        )

        try:
            async with get_session() as session:
                semantic_cache = SemanticCacheService(session)
                retrieval_service = RagRetrievalService(db=session)
                generation_service = RagGenerationService(
                    retrieval_service=retrieval_service,
                    semantic_cache_service=semantic_cache,
                )
                result = await generation_service.generate_answer(
                    query_text=body.query,
                    document_id=document_id,
                    document_name=body.document_name,
                    top_k=top_k,
                    min_score=min_score,
                )
        except Exception as e:
            logger.exception("HTTP /query error")
            raise HTTPException(status_code=500, detail=str(e))

        answer_data = result.get("result", {})
        timing_data = result.get("timing", {}) or {}
        retrieval = timing_data.get("retrieval", {}) or {}

        sources_out: List[Dict[str, Any]] = []
        for s in answer_data.get("sources", []) or []:
            bboxes = [
                [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                for b in (s.get("bboxes") or [])
                if isinstance(b, (list, tuple)) and len(b) == 4
            ]
            sources_out.append(
                {
                    "name": s.get("name", ""),
                    "page": s.get("page", 0),
                    "section": s.get("section") or None,
                    "exactText": s.get("exactText", ""),
                    "bboxes": bboxes,
                    "relevance": s.get("relevance", "high"),
                }
            )

        return {
            "result": {
                "response": answer_data.get("response", ""),
                "sources": sources_out,
            },
            "timing": {
                "embedding_ms": timing_data.get("embedding_ms", 0),
                "retrieval_total_ms": retrieval.get("retrieval_total_ms", 0),
                "llm_call_ms": timing_data.get("llm_call_ms", 0),
                "generation_total_ms": timing_data.get("generation_total_ms", 0),
                "original_chunk_count": timing_data.get("original_chunk_count", 0),
                "compressed_chunk_count": timing_data.get("compressed_chunk_count", 0),
                "embedding_cache_hit": timing_data.get("embedding_cache_hit", False),
                "semantic_cache_hit": timing_data.get("semantic_cache_hit", False),
                "chunk_cache_hit": retrieval.get("chunk_cache_hit", False),
                "response_cache_hit": timing_data.get("response_cache_hit", False),
                "semantic_cache_similarity": timing_data.get(
                    "semantic_cache_similarity", 0
                ),
            },
        }

    @app.post("/v1/ingest_pdf")
    async def ingest_pdf(body: IngestPdfBody):
        logger.info("HTTP /ingest_pdf document_id=%s", body.document_id)
        try:
            document_id = UUID(body.document_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid document_id: {e}")

        chunk_size = body.chunk_size if body.chunk_size > 0 else 750

        async def event_stream():
            # Serialize ingestion: only one docling pipeline runs at a time, so
            # concurrent/duplicate uploads queue instead of OOM-ing the service.
            await _INGEST_SEMAPHORE.acquire()
            try:
                async with get_session() as session:
                    semantic_cache = SemanticCacheService(session)
                    ingestion_service = RagIngestionService(
                        db=session,
                        semantic_cache_service=semantic_cache,
                    )
                    async for progress in ingestion_service.ingest_pdf(
                        document_url=body.document_url,
                        document_id=document_id,
                        chunk_size=chunk_size,
                    ):
                        payload: Dict[str, Any] = {
                            "stage": progress.stage,
                            "progress_percent": progress.progress,
                            "message": progress.message,
                            "result": None,
                        }
                        if progress.stage in ("COMPLETE", "ERROR") and progress.result:
                            payload["result"] = {
                                "success": progress.result.get("success", False),
                                "document_id": progress.result.get("document_id", ""),
                                "status": progress.result.get("status", ""),
                                "chunks_count": progress.result.get("chunks_count", 0),
                                "created_at": progress.result.get("created_at", ""),
                                "error": progress.result.get("error") or None,
                            }
                        yield f"data: {json.dumps(payload)}\n\n"
            except Exception as e:
                logger.exception("HTTP /ingest_pdf error")
                err_payload = {
                    "stage": "ERROR",
                    "progress_percent": 0,
                    "message": f"Ingestion failed: {e}",
                    "result": {
                        "success": False,
                        "document_id": body.document_id,
                        "status": "error",
                        "chunks_count": 0,
                        "created_at": "",
                        "error": str(e),
                    },
                }
                yield f"data: {json.dumps(err_payload)}\n\n"
            finally:
                _INGEST_SEMAPHORE.release()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/get_highlighted_pdf")
    async def get_highlighted_pdf(body: HighlightBody):
        logger.info("HTTP /get_highlighted_pdf page=%d", body.page)
        try:
            bboxes = [[b.x0, b.y0, b.x1, b.y1] for b in body.bboxes]
            pdf_bytes = await highlight_service.get_highlighted_pdf(
                doc_url=body.document_url,
                page=body.page,
                bboxes=bboxes,
            )
        except Exception as e:
            logger.exception("HTTP /get_highlighted_pdf error")
            raise HTTPException(status_code=500, detail=str(e))

        return Response(content=pdf_bytes, media_type="application/pdf")

    @app.post("/v1/invalidate_document")
    async def invalidate_document(body: InvalidateBody):
        logger.info("HTTP /invalidate_document document_id=%s", body.document_id)
        try:
            document_id = UUID(body.document_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid document_id: {e}")

        try:
            async with get_session() as session:
                from sqlalchemy import delete

                from rag_service.models.chunks import DocumentChunkDocling

                stmt = delete(DocumentChunkDocling).where(
                    DocumentChunkDocling.document_id == document_id
                )
                res = await session.execute(stmt)
                chunks_deleted = res.rowcount

                semantic_cache = SemanticCacheService(session)
                cache_deleted = await semantic_cache.invalidate_document(document_id)
                await session.commit()
        except Exception as e:
            logger.exception("HTTP /invalidate_document error")
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "success": True,
            "chunks_deleted": chunks_deleted,
            "cache_entries_deleted": cache_deleted,
        }

    return app


app = create_app()
