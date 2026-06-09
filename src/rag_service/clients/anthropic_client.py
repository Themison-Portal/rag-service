"""
Anthropic client for Claude LLM generation.
"""
import json
import logging
from typing import Any, Optional

from anthropic import AsyncAnthropic

from rag_service.config import get_settings

logger = logging.getLogger(__name__)

_anthropic_client = None

# Default models per model_hint. Sonnet is the high-quality default; Haiku
# is the fast/cheap alternative for triage-style tasks.
MODEL_FAST = "claude-haiku-4-5-20251001"
# "smart" falls back to settings.llm_model so deployments can override.


def get_anthropic_client() -> AsyncAnthropic:
    """Get or create the Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        settings = get_settings()
        _anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        logger.info("Initialized Anthropic client")
    return _anthropic_client


def resolve_model(model_hint: Optional[str]) -> str:
    """Map model_hint ("fast" | "smart" | "" | None) to a concrete model id."""
    settings = get_settings()
    if not model_hint or model_hint == "smart":
        return settings.llm_model
    if model_hint == "fast":
        return MODEL_FAST
    # Treat anything else as a direct model id (escape hatch).
    return model_hint


async def generate_response(
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Generate a response using Claude.

    Args:
        system_prompt: The system prompt.
        user_message: The user message.
        model: Model to use (defaults to config).
        max_tokens: Max tokens (defaults to config).

    Returns:
        The generated response text.
    """
    settings = get_settings()
    client = get_anthropic_client()

    response = await client.messages.create(
        model=model or settings.llm_model,
        max_tokens=max_tokens or settings.llm_max_tokens,
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
    """
    Generic LLM completion entry point used by the Generate gRPC.

    Args:
        messages: List of {role, content} dicts. Roles: system | user | assistant.
                  System messages are merged into Anthropic's `system` param;
                  user/assistant messages become the `messages` payload.
        model_hint: "fast" | "smart" | "" — see resolve_model().
        temperature: 0.0..1.0
        max_tokens: cap on output tokens
        response_schema: When provided, the model is forced (via tool calling)
                         to return JSON matching this JSON Schema. The returned
                         `content` is a serialized JSON string.
        response_schema_name: Identifier passed as the tool name when
                              response_schema is set.
        feature_tag: For logging / cost attribution.

    Returns:
        Dict with keys: content (str), model (str), prompt_tokens (int),
        completion_tokens (int).
    """
    settings = get_settings()
    client = get_anthropic_client()
    model = resolve_model(model_hint)
    effective_max_tokens = max_tokens or settings.llm_max_tokens

    # Split system messages from the conversation. Anthropic expects `system`
    # as a separate top-level argument.
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
        "[generate_text] feature=%s model=%s structured=%s msgs=%d",
        feature_tag or "(none)",
        model,
        bool(response_schema),
        len(convo),
    )

    # Build call kwargs; Anthropic rejects `system=None`, so omit when empty.
    call_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": effective_max_tokens,
        "temperature": temperature,
        "messages": convo,
    }
    if system_arg:
        call_kwargs["system"] = system_arg

    if response_schema:
        # Force structured output via tool_use. We give the model exactly one
        # tool whose input_schema is the caller's response_schema and instruct
        # it to call that tool with its answer.
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
        # Extract the tool_use block.
        content_str = ""
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                content_str = json.dumps(block.input)
                break
        if not content_str:
            # Fall back to any text content the model produced.
            content_str = "".join(
                getattr(b, "text", "") for b in response.content if getattr(b, "type", None) == "text"
            )
    else:
        response = await client.messages.create(**call_kwargs)
        content_str = "".join(
            getattr(b, "text", "") for b in response.content if getattr(b, "type", None) == "text"
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
