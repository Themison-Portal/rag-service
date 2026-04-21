"""
RAG Service - gRPC server entry point.
"""
import asyncio
import logging
import signal
import sys
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

# Import generated protobuf code
from gen.python.rag.v1 import rag_service_pb2 as pb2
from gen.python.rag.v1 import rag_service_pb2_grpc as pb2_grpc

from rag_service.server import RagServicer
from rag_service.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def serve():
    """Start the gRPC server."""
    settings = get_settings()

    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
        ],
    )

    # --- Self-Healing: Ensure pgvector is enabled ---
    from rag_service.db.session import get_session
    from sqlalchemy import text
    try:
        async with get_session() as session:
            logger.info("Self-healing: Ensuring pgvector extension is enabled...")
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await session.commit()
            logger.info("Self-healing: pgvector extension verified.")
    except Exception as e:
        logger.warning(f"Self-healing: Could not enable pgvector: {e}")

    # Add servicer
    pb2_grpc.add_RagServiceServicer_to_server(RagServicer(), server)

    # Add reflection for debugging
    SERVICE_NAMES = (
        pb2.DESCRIPTOR.services_by_name["RagService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Bind to port
    listen_addr = f"[::]:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting RAG Service on {listen_addr}")
    await server.start()

    # Graceful shutdown handling
    async def shutdown():
        logger.info("Shutting down RAG Service...")
        await server.stop(grace=5)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    logger.info("RAG Service is ready to serve requests")
    await server.wait_for_termination()


def main():
    """Main entry point."""
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
