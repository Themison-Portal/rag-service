"""
Database session management for the RAG service.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from typing import AsyncGenerator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_service.config import get_settings

logger = logging.getLogger(__name__)

# Create engine lazily
_engine = None
_async_session = None


def _get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={
                "command_timeout": 60,
                "server_settings": {
                    "statement_timeout": "60000",
                    "idle_in_transaction_session_timeout": "60000"
                }
            }
        )
    return _engine


def _get_session_factory():
    """Get or create the session factory."""
    global _async_session
    if _async_session is None:
        _async_session = async_sessionmaker(
            _get_engine(),
            class_=AsyncSession,
            expire_on_commit=False
        )
    return _async_session

def _reset_engine() -> None:
    """
    Drop cached engine/session-factory references. asyncpg connections are
    bound to the event loop they were created in — if the engine was built
    inside a different loop than the one now running (e.g. the self-heal
    step's asyncio.run() before uvicorn starts its own loop), every query
    fails with "Event loop is closed". Call this after any startup code
    that touches the DB in a throwaway loop, so the engine gets lazily
    rebuilt in whichever loop is actually serving requests.
    """
    global _engine, _async_session
    _engine = None
    _async_session = None


@asynccontextmanager
async def get_session(
    organization_id: Optional[str] = None,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    organization_id, when provided, sets a per-transaction session variable
    (app.current_org_id) that RLS policies can check.

    Usage:
        async with get_session(organization_id=org_id) as session:
            result = await session.execute(...)
    """
    session_factory = _get_session_factory()
    session = session_factory()
    try:
        if organization_id:
            await session.execute(
                text("SELECT set_config('app.current_org_id', :org_id, true)"),
                {"org_id": str(organization_id)},
            )
        yield session
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()


# Export engine for direct access if needed
engine = property(lambda: _get_engine())


async def check_database_connection() -> bool:
    """Check if database connection is healthy."""
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
