"""
Anthropic client for Claude LLM generation.
"""
import logging
from typing import Optional

from anthropic import AsyncAnthropic

from rag_service.config import get_settings

logger = logging.getLogger(__name__)

_anthropic_client = None


def get_anthropic_client() -> AsyncAnthropic:
    """Get or create the Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        settings = get_settings()
        _anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        logger.info("Initialized Anthropic client")
    return _anthropic_client


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
