"""
OpenAI client for embeddings.
"""
import logging
from typing import List

from langchain_openai import OpenAIEmbeddings

from rag_service.config import get_settings

logger = logging.getLogger(__name__)

_embedding_client = None


def get_embedding_client() -> OpenAIEmbeddings:
    """Get or create the OpenAI embedding client."""
    global _embedding_client
    if _embedding_client is None:
        settings = get_settings()
        _embedding_client = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            dimensions=settings.embedding_dimensions,
        )
        logger.info(f"Initialized OpenAI embedding client with model: {settings.embedding_model}")
    return _embedding_client


async def embed_query(text: str) -> List[float]:
    """
    Generate embedding for a single query.

    Args:
        text: The text to embed.

    Returns:
        List of floats representing the embedding vector.
    """
    client = get_embedding_client()
    return await client.aembed_query(text)


async def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple documents.

    Args:
        texts: List of texts to embed.

    Returns:
        List of embedding vectors.
    """
    client = get_embedding_client()
    return await client.aembed_documents(texts)
