"""
RAG Generation Service - LLM answer generation with caching.

Changes in this revision (Step 5 of the RAG Development Plan - contextual
retrieval, on top of the Step 2 grounding/logging work):
  - _extract_chunk_metadata now carries contextual_summary through from the
    retrieved chunk dict (populated at ingest time, see
    ingestion_service._generate_contextual_summaries; NULL for chunks
    ingested before the feature was enabled, or where summary generation
    failed - both are handled as "no context to prepend").
  - _compress_chunks carries contextual_summary through both the
    single-chunk and merged-page branches (picks the first non-null summary
    among merged chunks, since same-page chunks share a neighborhood).
  - _format_context_compact prepends the contextual summary ahead of the
    chunk content when present, so the LLM sees it as part of the chunk's
    context rather than as a separate signal.
  - _log_query_trace now records whether each retrieved chunk had context,
    to make the contextual-retrieval eval (measuring effect vs baseline)
    checkable against production trace logs, not just the offline eval
    script.

Changes carried over from Step 2:
  - Added explicit grounding rule to SYSTEM_PROMPT: don't guess/fill gaps
    with general knowledge when context is insufficient.
  - Added explicit relevance-scoring instruction to SYSTEM_PROMPT: the
    "relevance" field was previously just an unexplained example value
    in the schema ("relevance": "high"), which the model was echoing
    verbatim on every source regardless of actual relevance. Now it's
    told what the levels mean and to vary it accordingly.
  - Added SYSTEM_PROMPT_VERSION (hash of the prompt text) so trace logs
    can be tied to exactly which prompt version produced them.
  - Chunk scores now flow through _extract_chunk_metadata and
    _compress_chunks so _log_query_trace can surface per-chunk score
    (or scores, when chunks were merged) - needed to verify the
    confidence filter fix in retrieval_service.py is dropping the
    right chunks.
"""

import hashlib
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
- AMBIGUITY: If the question is short, generic, or could reasonably refer to more than one
  thing in the document (e.g. "what is the dose?" when multiple doses/arms exist), do NOT
  pick one interpretation and answer as if it were the only one. Either present all the
  relevant options the context supports (e.g. all treatment arms and their doses), or ask
  a clarifying question in the response text. Do not default to whichever retrieved chunk
  happens to be most detailed if the question itself doesn't specify which thing it means.
- COMPLETENESS FOR LISTS: When the question asks for "all," "main," or a full set of items
  (e.g. all exclusion/inclusion criteria, all endpoints), and the context contains a numbered
  or lettered list, you MUST include every single numbered/lettered item present in the
  context verbatim in your response — do not summarize, merge, or silently omit any item,
  even if some seem redundant or minor. If the context shows items numbered 1 through N,
  your response must account for all N.
- CRITICAL - EXACT REFUSAL PHRASE: If the context does not contain enough information to answer
  confidently, your response text MUST contain the literal substring "I don't have this
  information" - this exact wording, character for character. Do not paraphrase it, do not use a
  synonym or reworded version (do NOT write "the document does not specify", "there is no
  mention of", "is not addressed", etc. instead of it). This is checked programmatically, so the
  exact phrase must appear even if you also explain further. Example: "I don't have this
  information regarding [the specific thing asked]. However, the context does show [whatever
  related information is available]." If the context partially answers the question, answer what
  it does support and use this exact phrase only for the part that isn't covered.
- Do NOT write inline citations like "(Document_Title, p. X)" inside the response text -
  no doc name, no page number, no parenthetical citation mixed into sentences.
- DOCUMENT NAME PLACEMENT: If all cited content comes from a single document, state
  the document name ONCE, as the very first line of the response, in this exact format:
  "Source: {Document_Title}" - then a blank line, then the rest of the answer as normal.
  Do NOT repeat the document name anywhere else in the response - not in the reference
  tags, not inline in the prose. If content is cited from more than one distinct document,
  omit this header line entirely and rely on the sources array for attribution instead.
- After each point's text, add a reference tag on its own line in this EXACT format:
  "[Section {full section heading as given in context} · p.{page}]" - section and page
  ONLY. Never include the document name inside this tag - it belongs only in the single
  header line above, not per-point.
- Include bbox coordinates from context in your sources
- If multiple chunks from same page, include ALL their bboxes
- Set "relevance" on each source based on how directly it answers the question: "high" if it
  directly answers the question, "medium" if it provides supporting or related context, "low"
  if it is only tangentially related. Do not default every source to "high".

RESPOND WITH THIS EXACT JSON STRUCTURE (no other text):
{"response": "Source: Document_Title.pdf\n\nAnswer text here.\n\n1. Point text.\n[Section X · p.Y]\n\n2. Point text.\n[Section X · p.Y]", "sources": [{"name": "doc title", "page": 1, "section": "section or null", "exactText": "verbatim quote", "bboxes": [[x0,y0,x1,y1]], "relevance": "high"}]}"""

SYSTEM_PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]

# Separate, lightweight prompt for query condensation (Step 3 - conversation
# history). Deliberately NOT part of SYSTEM_PROMPT: condensation is a small,
# fast rewrite task, not a grounded-answer task, and keeping it isolated
# means a change to one prompt can't accidentally affect the other.
CONDENSE_PROMPT = """Given a conversation history and a new question, rewrite the \
new question as a standalone question that can be understood without the \
conversation history.

RULES:
- Only resolve ambiguous references (pronouns like "it"/"that", phrases like \
"the other one", "what about X instead") using the conversation history. Do \
NOT add facts, assumptions, numbers, or clinical details that were not \
explicitly stated in the conversation or the new question.
- If you cannot confidently resolve what the new question refers to, return \
it UNCHANGED rather than guessing - an unresolved but honest question is \
better than a confidently wrong rewrite in this context.
- If the new question is already standalone (doesn't depend on anything in \
the prior turns), return it unchanged.
- Do not attempt to answer the question. Only rewrite it.

Answer with ONLY the rewritten (or unchanged) question, nothing else - no \
preamble, no explanation, no quotation marks."""


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

    async def _condense_query(
        self,
        conversation_history: List[dict],
        new_question: str,
    ) -> str:
        """
        Step 3: rewrite a follow-up into a standalone query using recent
        conversation turns, so retrieval has something meaningful to embed
        and search on (previously only the latest message ever reached
        retrieval - "can you be more specific?" embedded and searched on
        its own, with no reference to what it was a follow-up to).

        Runs BEFORE retrieval. The original, unmodified new_question is
        still what's shown to the LLM for generation and to the user -
        only the retrieval-stage query changes, so the visible answer
        reads naturally rather than like it's answering a paraphrase.

        Returns new_question unchanged if there's no history, or on any
        failure - condensation is a retrieval-quality enhancement, not a
        hard dependency; a bad/missing rewrite should degrade to today's
        known-safe behavior, not break the query.
        """
        if not conversation_history or not settings.conversation_history_enabled:
            return new_question

        try:
            max_turns = settings.conversation_history_max_turns
            history_text = "\n".join(
                f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                for turn in conversation_history[-max_turns:]
            )
            client = get_anthropic_client()
            response = await client.messages.create(
                model=settings.llm_model,
                max_tokens=150,
                system=CONDENSE_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"<conversation>\n{history_text}\n</conversation>\n\n"
                            f"<new_question>\n{new_question}\n</new_question>"
                        ),
                    }
                ],
            )
            condensed = response.content[0].text.strip()
            self._log_llm_usage("query_condense", response)
            return condensed if condensed else new_question
        except Exception as e:
            logger.warning(f"[CONDENSE_QUERY] failed, using original question: {e}")
            return new_question

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
                    bbox = [
                        raw_bbox.get("l"),
                        raw_bbox.get("t"),
                        raw_bbox.get("r"),
                        raw_bbox.get("b"),
                    ]
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
            "score": doc.get("score"),
            "contextual_summary": doc.get("contextual_summary"),
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
                all_scores = [m["score"] for m in group if m.get("score") is not None]
                contextual_summary = next(
                    (m["contextual_summary"] for m in group if m.get("contextual_summary")),
                    None,
                )

                MAX_MERGED_CONTENT_CHARS = (
                    6000  # was 2000 - silently truncated mid-list on dense pages
                )
                merged_content = all_content[:MAX_MERGED_CONTENT_CHARS]
                if len(all_content) > MAX_MERGED_CONTENT_CHARS:
                    logger.warning(
                        f"[COMPRESSION_TRUNCATED] {title} p.{page}: {len(all_content)} chars "
                        f"truncated to {MAX_MERGED_CONTENT_CHARS} - content may be lost"
                    )

                compressed.append(
                    {
                        "title": title,
                        "page": page,
                        "section": section,
                        "bboxes": all_bboxes,
                        "content": merged_content,
                        "merged_count": len(group),
                        "scores": all_scores,
                        "contextual_summary": contextual_summary,
                    }
                )
        logger.info(f"[COMPRESSION] {len(chunks)} chunks -> {len(compressed)} compressed")
        return compressed

    def _format_context_compact(self, chunk_meta: dict) -> str:
        """Compact context format for reduced token usage."""
        title = chunk_meta.get("title", "Unknown")
        page = chunk_meta.get("page", 0)
        section = chunk_meta.get("section")
        content = chunk_meta.get("content", "")
        contextual_summary = chunk_meta.get("contextual_summary")

        if "bboxes" in chunk_meta:
            bbox_str = str(chunk_meta["bboxes"])
        else:
            bbox_str = str(chunk_meta.get("bbox"))

        section_str = f"|section:{section}" if section else ""

        body = f"{contextual_summary}\n{content}" if contextual_summary else content
        return f"[{title}|p{page}{section_str}|bbox:{bbox_str}]\n{body}"

    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON formatting issues."""
        repaired = json_str
        repaired = re.sub(r'(?<!\\)\n(?=(?:[^"]*"[^"]*")*[^"]*"[^"]*$)', "\\n", repaired)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = re.sub(r'(\})\s*(")', r"\1,\2", repaired)
        repaired = re.sub(r'(\])\s*(")', r"\1,\2", repaired)
        repaired = re.sub(r'(")\s+(")', r"\1,\2", repaired)
        repaired = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", repaired)
        return repaired

    def _parse_llm_json(self, raw_content: str) -> dict:
        """Parse JSON from LLM response with fallback strategies."""
        # Strategy 1: Direct parse
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON with regex
        json_match = re.search(r"\{[\s\S]*\}", raw_content)
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
                "response": response_match.group(1).replace('\\"', '"').replace("\\n", "\n"),
                "sources": [],
            }

        # Strategy 5: Return raw content
        clean_content = raw_content
        clean_content = re.sub(r'^\s*\{?\s*"response"\s*:\s*"?', "", clean_content)
        clean_content = re.sub(r'"?\s*,?\s*"sources"\s*:.*$', "", clean_content, flags=re.DOTALL)
        clean_content = clean_content.strip().strip('"').strip()

        return {
            "response": (
                clean_content[:3000] if clean_content else "Unable to parse response from AI."
            ),
            "sources": [],
        }

    def _log_query_trace(
        self,
        query_text: str,
        compressed_chunks: List[dict],
        formatted_context: str,
        result: dict,
        timing_info: dict,
        retrieval_query: Optional[str] = None,
    ) -> None:
        """Single structured log line per query - full trace for debugging/eval."""
        trace = {
            "event": "rag_query_trace",
            "query": query_text,
            "retrieval_query": retrieval_query if retrieval_query != query_text else None,
            "system_prompt_version": SYSTEM_PROMPT_VERSION,
            "retrieved_chunks": [
                {
                    "title": c.get("title"),
                    "page": c.get("page"),
                    "section": c.get("section"),
                    "score": c.get("score"),
                    "scores": c.get("scores"),
                    "had_context": bool(c.get("contextual_summary")),
                }
                for c in compressed_chunks
            ],
            "context_char_count": len(formatted_context),
            "response": result.get("response", ""),
            "sources": result.get("sources", []),
            "timing": timing_info,
        }
        logger.info(json.dumps(trace, default=str))

    def _log_llm_usage(
        self,
        call_type: str,
        response: Any,
        *,
        document_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
        model: Optional[str] = None,
    ) -> None:
        """Structured log line per LLM call, for cost tracking. Separate from
        _log_query_trace (which covers one full query) since usage needs to be
        logged per LLM call - a single query can trigger more than one call
        (condense + generate), and ingestion calls have no query trace at all."""
        usage = getattr(response, "usage", None)
        logger.info(
            json.dumps(
                {
                    "event": "llm_usage",
                    "call_type": call_type,
                    "model": model or getattr(response, "model", None),
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                    "document_id": str(document_id) if document_id else None,
                    "organization_id": str(organization_id) if organization_id else None,
                },
                default=str,
            )
        )

    async def generate_answer(
        self,
        query_text: str,
        document_id: UUID,
        document_name: str,
        organization_id: UUID,
        top_k: int = None,
        min_score: float = 0.04,
        conversation_history: Optional[List[dict]] = None,
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

        # 0. Condense follow-up questions into a standalone query BEFORE
        # anything touches retrieval. query_text (the original question) is
        # still used for generation/display below - only retrieval_query is
        # used for embedding/search.
        condense_start = time.perf_counter()
        retrieval_query = await self._condense_query(conversation_history or [], query_text)
        timing_info["condense_ms"] = (time.perf_counter() - condense_start) * 1000
        timing_info["query_condensed"] = retrieval_query != query_text

        # Follow-up answers are entangled with their specific conversation's
        # context - caching/serving them across different conversations
        # risks returning an answer phrased for someone else's follow-up.
        # Only standalone queries (no history) use the semantic cache.
        skip_cache = bool(conversation_history)

        # 1. Get query embedding
        query_embedding, embed_timing = await self.retrieval_service.get_query_embedding(
            retrieval_query
        )
        timing_info["embedding_ms"] = embed_timing.get("embedding_ms", 0)
        timing_info["embedding_cache_hit"] = embed_timing.get("cache_hit", False)

        # 2. Check semantic cache
        if self.semantic_cache_service and not skip_cache:
            semantic_start = time.perf_counter()
            cached = await self.semantic_cache_service.get_similar_response(
                query_embedding=query_embedding,
                document_id=document_id,
                organization_id=organization_id,
            )
            timing_info["semantic_cache_search_ms"] = (time.perf_counter() - semantic_start) * 1000

            if cached:
                timing_info["semantic_cache_hit"] = True
                timing_info["semantic_cache_similarity"] = cached["similarity"]
                timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000

                logger.info(
                    f"[TIMING] Semantic cache HIT: {timing_info['generation_total_ms']:.2f}ms"
                )

                self._log_query_trace(
                    query_text, [], "", cached["response"], timing_info, retrieval_query
                )

                return {"result": cached["response"], "timing": timing_info}
        # 3. Retrieve chunks
        filtered_chunks, retrieval_timing = await self.retrieval_service.retrieve_similar_chunks(
            query_text=retrieval_query,
            document_id=document_id,
            document_name=document_name,
            organization_id=organization_id,
            top_k=top_k,
            min_score=min_score,
            precomputed_embedding=query_embedding,
        )
        timing_info["retrieval"] = retrieval_timing
        timing_info["original_chunk_count"] = len(filtered_chunks)

        if not filtered_chunks:
            timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000
            empty_result = {
                "response": "The provided documents do not contain this information.",
                "sources": [],
            }
            self._log_query_trace(query_text, [], "", empty_result, timing_info, retrieval_query)
            return {"result": empty_result, "timing": timing_info}

        # 4. Compress chunks
        compression_start = time.perf_counter()
        compressed_chunks = self._compress_chunks(filtered_chunks)
        timing_info["compression_ms"] = (time.perf_counter() - compression_start) * 1000
        timing_info["compressed_chunk_count"] = len(compressed_chunks)
        timing_info["chunks_compressed"] = len(compressed_chunks) < len(filtered_chunks)

        # 5. Format context
        formatted_context = "\n\n".join(
            [self._format_context_compact(chunk) for chunk in compressed_chunks]
        )

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
            self._log_llm_usage(
                "query_generation",
                response,
                document_id=document_id,
                organization_id=organization_id,
            )
            logger.info(f"[TIMING] LLM call: {timing_info['llm_call_ms']:.2f}ms")

            raw_content = response.content[0].text

            # Diagnostic logging: capture everything needed to root-cause an
            # empty sources array without needing to reproduce the query.
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "max_tokens":
                logger.warning(
                    f"[LLM_TRUNCATED] query={query_text!r} stop_reason={stop_reason} "
                    f"raw_len={len(raw_content)}"
                )

            parsed = self._parse_llm_json(raw_content)

            if not parsed.get("sources"):
                logger.warning(
                    f"[EMPTY_SOURCES] query={query_text!r} stop_reason={stop_reason} "
                    f"chunk_count={len(compressed_chunks)} "
                    f"chunk_pages={[c.get('page') for c in compressed_chunks]} "
                    f"raw_content_tail={raw_content[-500:]!r}"
                )

            # Convert sources
            sources = []
            for s in parsed.get("sources", []):
                bboxes = s.get("bboxes", [])
                if bboxes and not isinstance(bboxes[0], list):
                    bboxes = [bboxes]

                sources.append(
                    {
                        "name": s.get("name", s.get("protocol", "Unknown")),
                        "page": s.get("page", 0),
                        "section": s.get("section"),
                        "exactText": s.get("exactText", ""),
                        "bboxes": bboxes,
                        "relevance": s.get("relevance", "high"),
                    }
                )

                # Safety net: if the LLM produced an answer but the sources array
            # came back empty (JSON truncation, parse fallback, formatting
            # slip), fall back to the chunks we actually sent it. Guarantees
            # the Evidence Viewer always has something to show whenever real
            # context was used.
            if not sources and compressed_chunks:
                logger.warning(
                    f"[SOURCES_FALLBACK] Reconstructing sources for query={query_text!r}"
                )
                for chunk in compressed_chunks:
                    bboxes = chunk.get("bboxes") or ([chunk["bbox"]] if chunk.get("bbox") else [])
                    sources.append(
                        {
                            "name": chunk.get("title", "Unknown"),
                            "page": chunk.get("page", 0),
                            "section": chunk.get("section"),
                            "exactText": "",
                            "bboxes": bboxes,
                            "relevance": "high",
                        }
                    )

            result = {
                "response": parsed.get("response", ""),
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"[ERROR] Claude API call failed: {e}")
            timing_info["llm_call_ms"] = (time.perf_counter() - llm_start) * 1000
            timing_info["error"] = str(e)
            return {
                "result": {"response": f"Error generating response: {str(e)}", "sources": []},
                "timing": timing_info,
            }

        # 7. Store in semantic cache
        if self.semantic_cache_service and not skip_cache:
            context_hash = SemanticCacheService.hash_context(filtered_chunks)
            await self.semantic_cache_service.store_response(
                query_text=query_text,
                query_embedding=query_embedding,
                organization_id=organization_id,
                document_id=document_id,
                response=result,
                context_hash=context_hash,
            )

        timing_info["generation_total_ms"] = (time.perf_counter() - generation_start) * 1000
        logger.info(f"[TIMING] Generation complete: {timing_info['generation_total_ms']:.2f}ms")

        self._log_query_trace(
            query_text, compressed_chunks, formatted_context, result, timing_info, retrieval_query
        )

        return {"result": result, "timing": timing_info}
