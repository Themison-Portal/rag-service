"""
Pytest configuration and fixtures for RAG service tests.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Set test environment variables before importing app modules
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:54322/postgres")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def mock_db_session() -> AsyncGenerator[AsyncMock, None]:
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    session.close = AsyncMock()
    yield session


@pytest.fixture
def mock_embedding_client():
    """Create a mock embedding client."""
    client = MagicMock()
    # Return 1536-dimensional embeddings
    client.aembed_query = AsyncMock(return_value=[0.1] * 1536)
    client.aembed_documents = AsyncMock(return_value=[[0.1] * 1536])
    return client


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = AsyncMock()
    response = MagicMock()
    response.content = [MagicMock(text='{"response": "Test answer", "sources": []}')]
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.fixture
def sample_document_id():
    """Generate a sample document ID."""
    return uuid4()


@pytest.fixture
def sample_chunks():
    """Create sample document chunks."""
    return [
        {
            "page_content": "This is a test chunk about clinical trials.",
            "score": 0.85,
            "metadata": {
                "title": "Test Document",
                "page": 1,
                "docling": {
                    "dl_meta": {
                        "page_no": 1,
                        "headings": ["Introduction"],
                        "doc_items": [{
                            "prov": [{
                                "page_no": 1,
                                "bbox": {"l": 10, "t": 20, "r": 100, "b": 50}
                            }]
                        }]
                    }
                }
            }
        },
        {
            "page_content": "Another chunk with protocol information.",
            "score": 0.75,
            "metadata": {
                "title": "Test Document",
                "page": 2,
                "docling": {
                    "dl_meta": {
                        "page_no": 2,
                        "headings": ["Methods"],
                        "doc_items": [{
                            "prov": [{
                                "page_no": 2,
                                "bbox": {"l": 15, "t": 25, "r": 105, "b": 55}
                            }]
                        }]
                    }
                }
            }
        }
    ]


@pytest.fixture
def sample_query_response():
    """Create a sample query response."""
    return {
        "response": "Based on the document, clinical trials require specific protocols.",
        "sources": [
            {
                "name": "Test Document",
                "page": 1,
                "section": "Introduction",
                "exactText": "clinical trials",
                "bboxes": [[10, 20, 100, 50]],
                "relevance": "high"
            }
        ]
    }
