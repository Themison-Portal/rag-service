"""
RAG Reranking Service - Cross-encoder reranking via Cohere.

STEP 4 of the RAG Development Plan: "The Cohere reranker is configured
but never called in the code. Finish wiring it so the best chunks are
reordered to the top before the model sees them."

Config for this already existed in config.py (reranker_enabled,
reranker_provider, reranker_model, reranker_top_k, cohere_api_key) -
this file is what actually reads and uses it. Retrieval should
over-fetch (see retrieval_fetch_k) and hand everything to rerank(),
which trims down to reranker_top_k.
"""

import logging
import time
from typing import List, Tuple

import cohere

from rag_service.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_cohere_client: cohere.AsyncClient | None = None


def get_cohere_client() -> cohere.AsyncClient:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.AsyncClient(api_key=settings.cohere_api_key)
    return _cohere_client


class RerankerService:
    """
    Reranks a candidate set of chunks against the query using a
    cross-encoder model, so the chunks most relevant to THIS specific
    question end up first - not just the ones with the best raw
    vector/BM25 similarity.
    """

    def __init__(self):
        self.provider = settings.reranker_provider
        self.model = settings.reranker_model
        self.top_k = settings.reranker_top_k
        self.enabled = settings.reranker_enabled
        self.client = get_cohere_client() if self.provider == "cohere" else None

    async def rerank(
        self,
        query_text: str,
        chunks: List[dict],
        top_n: int = None,
    ) -> Tuple[List[dict], dict]:
        """
        Rerank chunks by relevance to query_text.

        Args:
            query_text: the user's question.
            chunks: candidate chunks, each with a "page_content" key.
            top_n: how many chunks to keep after reranking. Defaults to
                   settings.reranker_top_k if not given.

        Returns:
            (reranked_chunks, timing_info). If reranking is disabled,
            fails, or there are no candidates, falls back to the
            original order truncated to top_n - a Cohere outage
            degrades quality rather than breaking retrieval entirely.
        """
        effective_top_n = top_n if top_n is not None else self.top_k
        timing_info = {"reranker_enabled": self.enabled, "reranker_provider": self.provider}

        if not chunks:
            timing_info["reranker_skipped"] = "no_candidates"
            return chunks, timing_info

        if not self.enabled:
            timing_info["reranker_skipped"] = "disabled"
            return chunks[:effective_top_n], timing_info

        if self.provider != "cohere" or self.client is None:
            timing_info["reranker_skipped"] = f"unsupported_provider:{self.provider}"
            return chunks[:effective_top_n], timing_info

        rerank_start = time.perf_counter()
        try:
            documents = [c.get("page_content", "") for c in chunks]

            response = await self.client.rerank(
                model=self.model,
                query=query_text,
                documents=documents,
                top_n=min(effective_top_n, len(documents)),
            )

            reranked = []
            for result in response.results:
                chunk = chunks[result.index].copy()
                chunk["rerank_score"] = result.relevance_score
                # Keep the pre-rerank score too, so the trace log (Step 1)
                # can show both and it's visible whether reranking actually
                # changed the ordering vs. just confirming it.
                chunk["pre_rerank_score"] = chunk.get("score")
                chunk["score"] = result.relevance_score
                reranked.append(chunk)

            timing_info["reranker_ms"] = (time.perf_counter() - rerank_start) * 1000
            timing_info["reranker_input_count"] = len(chunks)
            timing_info["reranker_output_count"] = len(reranked)
            timing_info["reranker_model"] = self.model

            logger.info(
                f"[RERANK] {len(chunks)} -> {len(reranked)} chunks in "
                f"{timing_info['reranker_ms']:.2f}ms using {self.model}"
            )
            return reranked, timing_info

        except Exception as e:
            logger.error(f"[RERANK_ERROR] Falling back to pre-rerank order: {e}")
            timing_info["reranker_error"] = str(e)
            timing_info["reranker_ms"] = (time.perf_counter() - rerank_start) * 1000
            return chunks[:effective_top_n], timing_info
