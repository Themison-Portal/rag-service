"""
Configuration for the RAG microservice.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    RAG Service settings loaded from environment variables.
    """

    # API Keys
    openai_api_key: str
    anthropic_api_key: str

    # Database
    database_url: str  # PostgreSQL connection (asyncpg format)

    # Transport: "http" (default) or "grpc". main.py picks the server to
    # start based on this. Both implementations stay in the repo so a
    # revert is one env-var flip.
    transport: str = "http"

    # HTTP server (when transport == "http")
    http_port: int = 8000

    # gRPC Server (when transport == "grpc")
    grpc_port: int = 50051
    grpc_max_workers: int = 10

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Retrieval configuration
    retrieval_top_k: int = 20
    retrieval_min_score: float = 0.04
    retrieval_min_bm25_score: float | None = (
        None  # observe-mode until derived empirically from real bm25_score_normalized data
    )

    # Hybrid search configuration
    hybrid_search_enabled: bool = True
    hybrid_search_rrf_k: int = 60

    # Semantic cache configuration
    semantic_cache_similarity_threshold: float = 0.90

    # Reranking configuration
    reranker_enabled: bool = True
    reranker_provider: str = "cohere"
    reranker_model: str = "rerank-english-v3.0"
    reranker_top_k: int = 6
    retrieval_fetch_k: int = 45  # wide enough to catch scattered/multi-section answers on this doc
    cohere_api_key: str = "yLkBSYMAAUMR3VxOFzeY7O4a0WYjiRF6XuZKBvt3"

    # Contextual retrieval configuration
    contextual_retrieval_enabled: bool = False
    contextual_context_window: int = 3

    # LLM configuration
    llm_model: str = "claude-sonnet-4-6"
    # llm_max_tokens: int = 2000
    llm_max_tokens: int = 6000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

    from pydantic import field_validator

    @field_validator("openai_api_key", "anthropic_api_key", "database_url")
    @classmethod
    def clean_secrets(cls, v):
        if isinstance(v, str):
            # Also handle potential quotes or escaped newlines
            return v.strip().replace('"', "").replace("'", "")
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
