"""
LLM client — Anthropic (baseline) and OpenAI (eval candidate), switched via
settings.generation_provider. Kept in one file since both back the same
generate_response/generate_text entry points; rag_service.clients.openai_client
is a separate, unrelated file for embeddings (langchain_openai.OpenAIEmbeddings).
"""

import json
import logging
from typing import Any, Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from rag_service.config import get_settings

logger = logging.getLogger(__name__)

_anthropic_client = None
_openai_llm_client = None  # distinct from openai_client.py's embedding client


def get_anthropic_client() -> AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        settings = get_settings()
        _anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        logger.info("Initialized Anthropic client")
    return _anthropic_client


def get_openai_llm_client() -> AsyncOpenAI:
    global _openai_llm_client
    if _openai_llm_client is None:
        settings = get_settings()
        _openai_llm_client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("Initialized OpenAI LLM client")
    return _openai_llm_client


def resolve_model(model_hint: Optional[str]) -> str:
    settings = get_settings()
    provider = settings.generation_provider
    if provider == "openai":
        if model_hint == "fast":
            return settings.llm_model_openai_fast
        return settings.llm_model_openai_smart
    else:
        if model_hint == "fast":
            return settings.llm_model_fast
        if not model_hint or model_hint == "smart":
            return settings.llm_model
        return model_hint  # escape hatch


async def generate_response(
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    settings = get_settings()
    provider = settings.generation_provider
    effective_max_tokens = max_tokens or settings.llm_max_tokens

    if provider == "openai":
        client = get_openai_llm_client()
        resolved_model = model or settings.llm_model_openai_smart
        response = await client.chat.completions.create(
            model=resolved_model,
            reasoning_effort="low",
            max_completion_tokens=effective_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""
    else:
        client = get_anthropic_client()
        response = await client.messages.create(
            model=model or settings.llm_model,
            max_tokens=effective_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


async def generate_text(
    messages: list[dict[str, str]],
    *,
    model_hint: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_schema: Optional[dict[str, Any]] = None,
    response_schema_name: str = "structured_output",
    feature_tag: str = "",
) -> dict[str, Any]:
    settings = get_settings()
    provider = settings.generation_provider
    model = resolve_model(model_hint)
    effective_max_tokens = max_tokens or settings.llm_max_tokens

    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    convo = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]
    if not convo:
        raise ValueError("generate_text: messages must include at least one user message")

    system_arg = "\n\n".join(p for p in system_parts if p) or None

    logger.info(
        "[generate_text] feature=%s provider=%s model=%s structured=%s msgs=%d",
        feature_tag or "(none)",
        provider,
        model,
        bool(response_schema),
        len(convo),
    )

    if provider == "openai":
        client = get_openai_llm_client()
        openai_messages = (
            [{"role": "system", "content": system_arg}] if system_arg else []
        ) + convo
        kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": effective_max_tokens,
            "reasoning_effort": "low",
            "messages": openai_messages,
        }
        if response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema_name,
                    "schema": response_schema,
                    "strict": False,
                },
            }
        response = await client.chat.completions.create(**kwargs)
        content_str = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

    else:
        client = get_anthropic_client()
        call_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_arg:
            call_kwargs["system"] = system_arg

        if response_schema:
            tool_name = response_schema_name or "structured_output"
            call_kwargs["tools"] = [
                {
                    "name": tool_name,
                    "description": "Return the answer in the required JSON shape.",
                    "input_schema": response_schema,
                }
            ]
            call_kwargs["tool_choice"] = {"type": "tool", "name": tool_name}
            response = await client.messages.create(**call_kwargs)
            content_str = ""
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    content_str = json.dumps(block.input)
                    break
            if not content_str:
                content_str = "".join(
                    getattr(b, "text", "")
                    for b in response.content
                    if getattr(b, "type", None) == "text"
                )
        else:
            response = await client.messages.create(**call_kwargs)
            content_str = "".join(
                getattr(b, "text", "")
                for b in response.content
                if getattr(b, "type", None) == "text"
            )

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    return {
        "content": content_str,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
