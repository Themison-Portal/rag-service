"""
RAG Service entry point.

Picks the transport at startup based on `TRANSPORT` (default "http"):
  - "http" → run uvicorn against `rag_service.http_server:app`
  - "grpc" → run the legacy gRPC server (kept for revert)
"""
import asyncio
import logging
import os
import signal
import sys
from concurrent import futures

from rag_service.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _self_heal_pgvector_sync() -> None:
    """Synchronously ensure pgvector is enabled before serving traffic."""
    from sqlalchemy import text

    from rag_service.db.session import get_session

    async def _run():
        try:
            async with get_session() as session:
                logger.info("Self-healing: Ensuring pgvector extension is enabled...")
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await session.commit()
                logger.info("Self-healing: pgvector extension verified.")
        except Exception as e:
            logger.warning(f"Self-healing: Could not enable pgvector: {e}")

    asyncio.run(_run())


def run_http() -> None:
    """Run the FastAPI HTTP server via uvicorn."""
    import uvicorn

    settings = get_settings()
    # Cloud Run / Render set $PORT — honour it if present.
    port = int(os.environ.get("PORT") or settings.http_port)
    _self_heal_pgvector_sync()
    logger.info(f"Starting RAG Service (HTTP) on 0.0.0.0:{port}")
    uvicorn.run(
        "rag_service.http_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )


async def _serve_grpc() -> None:
    import grpc
    from grpc_reflection.v1alpha import reflection

    from gen.python.rag.v1 import rag_service_pb2 as pb2
    from gen.python.rag.v1 import rag_service_pb2_grpc as pb2_grpc
    from rag_service.server import RagServicer

    settings = get_settings()
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )

    # Self-heal pgvector before serving
    from sqlalchemy import text

    from rag_service.db.session import get_session

    try:
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await session.commit()
    except Exception as e:
        logger.warning(f"Self-healing: Could not enable pgvector: {e}")

    pb2_grpc.add_RagServiceServicer_to_server(RagServicer(), server)
    SERVICE_NAMES = (
        pb2.DESCRIPTOR.services_by_name["RagService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    listen_addr = f"[::]:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting RAG Service (gRPC) on {listen_addr}")
    await server.start()

    async def shutdown():
        logger.info("Shutting down RAG Service...")
        await server.stop(grace=5)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        except NotImplementedError:
            pass

    await server.wait_for_termination()


def run_grpc() -> None:
    asyncio.run(_serve_grpc())


def main() -> None:
    transport = (os.environ.get("TRANSPORT") or get_settings().transport or "http").lower()
    try:
        if transport == "grpc":
            run_grpc()
        else:
            run_http()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
