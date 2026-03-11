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

    # gRPC Server
    grpc_port: int = 50051
    grpc_max_workers: int = 10

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Retrieval configuration
    retrieval_top_k: int = 20
    retrieval_min_score: float = 0.04

    # Hybrid search configuration
    hybrid_search_enabled: bool = True
    hybrid_search_rrf_k: int = 60

    # Semantic cache configuration
    semantic_cache_similarity_threshold: float = 0.90

    # Reranking configuration
    reranker_enabled: bool = False
    reranker_provider: str = "cohere"
    reranker_model: str = "rerank-english-v3.0"
    reranker_top_k: int = 5
    cohere_api_key: str = ""

    # Contextual retrieval configuration
    contextual_retrieval_enabled: bool = False
    contextual_context_window: int = 3

    # LLM configuration
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 2000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
