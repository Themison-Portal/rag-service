"""
evals/rag_client.py

Adapter between the eval suite and the real RagGenerationService.
"""

import asyncio
import os
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_service.services.retrieval_service import RagRetrievalService
from rag_service.services.generation_service import RagGenerationService
from rag_service.config import get_settings

settings = get_settings()

# Fixed test fixtures — the UC protocol doc already ingested in your eval/staging DB.
EVAL_DOCUMENT_ID = UUID(os.environ["EVAL_DOCUMENT_ID"])
EVAL_DOCUMENT_NAME = os.environ.get("EVAL_DOCUMENT_NAME", "UC Protocol v1.1")
EVAL_ORGANIZATION_ID = UUID(os.environ["EVAL_ORGANIZATION_ID"])


async def _run_rag_pipeline_async(question: str, history: list | None = None):
    # Engine/session are created fresh INSIDE this coroutine, not at module
    # level. asyncio.run() (see run_rag_pipeline below) spins up a new event
    # loop per call, and asyncpg connections are bound to the loop that
    # created them - a module-level engine gets reused across loops and
    # breaks with "another operation is in progress" / "attached to a
    # different loop" after the first call. Creating + disposing per call
    # is wasteful (new connection pool every question) but correct, and
    # eval-suite runs aren't performance-sensitive.
    engine = create_async_engine(settings.database_url)
    session_local = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with session_local() as db:
            retrieval_service = RagRetrievalService(db)
            generation_service = RagGenerationService(retrieval_service)

            conversation_history = None
            if history:
                conversation_history = []
                for prev_q, prev_a in history:
                    conversation_history.append({"role": "user", "content": prev_q})
                    conversation_history.append({"role": "assistant", "content": prev_a})

            output = await generation_service.generate_answer(
                query_text=question,
                document_id=EVAL_DOCUMENT_ID,
                document_name=EVAL_DOCUMENT_NAME,
                organization_id=EVAL_ORGANIZATION_ID,
                conversation_history=conversation_history,
            )

            result = output["result"]
            answer_text = result.get("response", "")
            retrieved_context = result.get("debug_context", [])

            return answer_text, retrieved_context
    finally:
        await engine.dispose()


def run_rag_pipeline(question: str, history: list | None = None):
    """Sync wrapper for deepeval/pytest."""
    return asyncio.run(_run_rag_pipeline_async(question, history))
