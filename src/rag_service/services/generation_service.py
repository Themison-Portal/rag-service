"""
RAG Generation Service - LLM answer generation with caching.
"""
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from rag_service.clients.anthropic_client import get_anthropic_client
from rag_service.cache.semantic_cache import SemanticCacheService
from rag_service.services.retrieval_service import RagRetrievalService
from rag_service.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# System prompt optimized for prompt caching
SYSTEM_PROMPT = """You are an expert clinical Document assistant. You MUST respond with valid JSON only.

RULES:
- Use ONLY the provided context
- Every fact MUST have an inline citation: (Document_Title, p. X)
- Include bbox coordinates from context in your sources
- If multiple chunks from same page, include ALL their bboxes

RESPOND WITH THIS EXACT JSON STRUCTURE (no other text):
{"response": "markdown answer with citations", "sources": [{"name": "doc title", "page": 1, "section": "section or null", "exactText": "verbatim quote", "bboxes": [[x0,y0,x1,y1]], "relevance": "high"}]}"""


class RagGenerationService:
    """
    RAG generation service that combines retrieval and LLM generation.
    """

    def __init__(
        self,
        retrieval_service: RagRetrievalService,
        semantic_cache_service: Optional[SemanticCacheService] = None,
    ):
        self.retrieval_service = retrieval_service
        self.semantic_cache_service = semantic_cache_service

    def _extract_chunk_metadata(self, doc: dict) -> dict:
        """Extract metadata from a chunk."""
        meta = doc.get("metadata", {})
        dl_meta = meta.get("docling", {}).get("dl_meta", {})
        doc_items = dl_meta.get("doc_items", [])

        bbox = None
        if doc_items:
            prov = doc_items[0].get("prov", [])
            if prov:
                raw_bbox = prov[0].get("bbox")
                if isinstance(raw_bbox, dict):
                    bbox = [raw_bbox.get("l"), raw_bbox.get("t"), raw_bbox.get("r"), raw_bbox.get("b")]
                else:
                    bbox = raw_bbox

        title = meta.get("title", "Unknown")
        page = dl_meta.get("page_no") or meta.get("page") or 0
        headings = dl_meta.get("headings", [])
        section = headings[-1] if headings else None

        return {
            "title": title,
            "page": page,
            "section": section,
            "bbox": bbox,
            "content": doc.get("page_content", ""),
        }

    def _compress_chunks(self, chunks: List[dict]) -> List[dict]:
        """Compress chunks by merging those from the same page."""
        if not chunks:
            return []

        page_groups: Dict[tuple, List[dict]] = {}
        for chunk in chunks:
            meta = self._extract_chunk_metadata(chunk)
            key = (meta["title"], meta["page"])
            if key not in page_groups:
                page_groups[key] = []
            page_groups[key].append(meta)

        compressed = []
        for (title, page), group in page_groups.items():
            if len(group) == 1:
                compressed.append(group[0])
            else:
                all_bboxes = [m["bbox"] for m in group if m["bbox"]]
                all_content = "\n...\n".join(m["content"] for m in group)
                section = next((m["section"] for m in group if m["section"]), None)

                compressed.append({
                    "title": title,
                    "page": page,
                    "section": section,
                    "bboxes": all_bboxes,
                    "content": all_content[:2000],
                    "merged_count": len(group),
                })

        logger.info(f"[COMPRESSION] {len(chunks)} chunks -> {len(compressed)} compressed")
        return compressed

    def _format_context_compact(self, chunk_meta: dict) -> str:
        """Compact context format for reduced token usage."""
        title = chunk_meta.get("title", "Unknown")
        page = chunk_meta.get("page", 0)
        content = chunk_meta.get("content", "")

        if "bboxes" in chunk_meta:
            bbox_str = str(chunk_meta["bboxes"])
        else:
            bbox_str = str(chunk_meta.get("bbox"))

        return f"[{title}|p{page}|bbox:{bbox_str}]\n{content}"

    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON formatting issues."""
        repaired = json_str
        repaired = re.sub(r'(?<!\\)\n(?=(?:[^"]*"[^"]*")*[^"]*"[^"]*$)', '\\n', repaired)
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        repaired = re.sub(r'(\})\s*(")', r'\1,\2', repaired)
        repaired = re.sub(r'(\])\s*(")', r'\1,\2', repaired)
        repaired = re.sub(r'(")\s+(")', r'\1,\2', repaired)
        repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)
        return repaired

    def _parse_llm_json(self, raw_content: str) -> dict:
        """Parse JSON from LLM response with fallback strategies."""
        # Strategy 1: Direct parse
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON with regex
        json_match = re.search(r'\{[\s\S]*\}', raw_content)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            # Strategy 3: Repair and parse
            try:
                repaired = self._repair_json(json_str)
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Extract response field only
        response_match = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]', raw_content)
        if response_match:
            return {
                "response": response_match.group(1).replace('\\"', '"').replace('\\n', '\n'),
                "sources": []
            }

        # Strategy 5: Return raw content
        clean_content = raw_content
        clean_content = re.sub(r'^\s*\{?\s*"response"\s*:\s*"?', '', clean_content)
        clean_content = re.sub(r'"?\s*,?\s*"sources"\s*:.*$', '', clean_content, flags=re.DOTALL)
        clean_content = clean_content.strip().strip('"').strip()

        return {
            "response": clean_content[:3000] if clean_content else "Unable to parse response from AI.",
            "sources": []
        }

    async def generate_answer(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        top_k: int = 15,
        min_score: float = 0.04
    ) -> dict:
        """
        Generate answer with timing information.

        Returns dict with 'result' and 'timing' info.
        """
        generation_start = time.perf_counter()
        timing_info = {
            "response_cache_hit": False,
            "semantic_cache_hit": False,
            "chunks_compressed": False,
        }

        # 1. Get query embedding
        query_embedding, embed_timing = await self.retrieval_service.get_query_embedding(query_text)
        timing_info["embedding_ms"] = embed_timing.get("embedding_ms", 0)
        timing_info["embedding_cache_hit"] = embed_timing.get("cache_hit", False)

        # 2. Check semantic cache
        if self.semantic_cache_service:
            semantic_start = time.perf_counter()
            cached = await self.semantic_cache_service.get_similar_response(
                query_embedding=query_embedding,
                document_id=document_id
            )
            timing_info["semantic_cache_search_ms"] = (time.perf_counter() - semantic_start) * 1000

            if cached:
                timing_info["semantic_cache_hit"] = True
                timing_info["semantic_cache_similarity"] = cached["similarity"]
                timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000

                logger.info(f"[TIMING] Semantic cache HIT: {timing_info['generation_total_ms']:.2f}ms")

                return {
                    "result": cached["response"],
                    "timing": timing_info
                }

        # 3. Retrieve chunks
        filtered_chunks, retrieval_timing = await self.retrieval_service.retrieve_similar_chunks(
            query_text=query_text,
            document_id=document_id,
            document_name=document_name,
            top_k=top_k,
            min_score=min_score,
            precomputed_embedding=query_embedding
        )
        timing_info["retrieval"] = retrieval_timing
        timing_info["original_chunk_count"] = len(filtered_chunks)

        if not filtered_chunks:
            timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000
            return {
                "result": {
                    "response": "The provided documents do not contain this information.",
                    "sources": []
                },
                "timing": timing_info
            }

        # 4. Compress chunks
        compression_start = time.perf_counter()
        compressed_chunks = self._compress_chunks(filtered_chunks)
        timing_info["compression_ms"] = (time.perf_counter() - compression_start) * 1000
        timing_info["compressed_chunk_count"] = len(compressed_chunks)
        timing_info["chunks_compressed"] = len(compressed_chunks) < len(filtered_chunks)

        # 5. Format context
        formatted_context = "\n\n".join([
            self._format_context_compact(chunk) for chunk in compressed_chunks
        ])

        # 6. Call Claude
        llm_start = time.perf_counter()
        user_message = f"CONTEXT:\n{formatted_context}\n\nQUESTION: {query_text}"

        try:
            client = get_anthropic_client()
            response = await client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            timing_info["llm_call_ms"] = (time.perf_counter() - llm_start) * 1000
            logger.info(f"[TIMING] LLM call: {timing_info['llm_call_ms']:.2f}ms")

            raw_content = response.content[0].text
            parsed = self._parse_llm_json(raw_content)

            # Convert sources
            sources = []
            for s in parsed.get("sources", []):
                bboxes = s.get("bboxes", [])
                if bboxes and not isinstance(bboxes[0], list):
                    bboxes = [bboxes]

                sources.append({
                    "name": s.get("name", s.get("protocol", "Unknown")),
                    "page": s.get("page", 0),
                    "section": s.get("section"),
                    "exactText": s.get("exactText", ""),
                    "bboxes": bboxes,
                    "relevance": s.get("relevance", "high"),
                })

            result = {
                "response": parsed.get("response", ""),
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"[ERROR] Claude API call failed: {e}")
            timing_info["llm_call_ms"] = (time.perf_counter() - llm_start) * 1000
            timing_info["error"] = str(e)
            return {
                "result": {
                    "response": f"Error generating response: {str(e)}",
                    "sources": []
                },
                "timing": timing_info
            }

        # 7. Store in semantic cache
        if self.semantic_cache_service:
            context_hash = SemanticCacheService.hash_context(filtered_chunks)
            await self.semantic_cache_service.store_response(
                query_text=query_text,
                query_embedding=query_embedding,
                document_id=document_id,
                response=result,
                context_hash=context_hash
            )

        timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000
        logger.info(f"[TIMING] Generation complete: {timing_info['generation_total_ms']:.2f}ms")

        return {
            "result": result,
            "timing": timing_info
        }
