"""
Live API-key validation for the configured ANTHROPIC_API_KEY.

This test makes a real HTTP call to Anthropic with a minimal request
(just enough to validate the key + account access) and asserts a 200.
It is deliberately separated from the rest of the test suite because:

- It needs network access.
- It needs a real, working key in the environment.
- It costs a few cents per run (negligible — sub-cent for Haiku 3.5).

Skipped automatically when the key is missing or looks like a test stub
(the conftest.py stub-sets ``ANTHROPIC_API_KEY=test-anthropic-key`` for
all other tests, so this test only fires when a real key was injected
via the environment before pytest started).

Run inside the rag-service container so the secret never leaves the
environment that already has it::

    docker compose exec rag-service pytest tests/test_anthropic_key.py -v

Or with a specific marker::

    pytest -m live tests/test_anthropic_key.py
"""

import os

import pytest


# Cheapest current Claude model — keeps the validation call sub-cent.
# Updated 2026-05-18: claude-3-5-haiku-20241022 reached EOL 2026-02-19.
_LIVE_MODEL = "claude-haiku-4-5-20251001"

# Production model the rag-service actually uses (rag_service/config.py:50).
# A separate test verifies the configured key has access to *this* model,
# not just any model — a key that authenticates fine but can't call the
# prod model still breaks the live RAG flow.
_PRODUCTION_MODEL = "claude-sonnet-4-20250514"

# Sentinel values set by the conftest stub or commonly used in CI
# fixtures. If the key matches any of these, the test is skipped rather
# than sending an obviously-bogus key to Anthropic and producing a
# misleading 401 in the test output.
_STUB_KEYS = {"", "test-anthropic-key", "test", "fake", "dummy"}


def _redact(secret: str) -> str:
    """Render a secret as ``sk-ant-a…XXXX`` so test output never leaks it."""
    if len(secret) < 16:
        return f"<len={len(secret)}>"
    return f"{secret[:8]}…{secret[-4:]}"


def _looks_like_stub(key: str) -> bool:
    return key in _STUB_KEYS or key.startswith("test-")


@pytest.fixture
def live_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if _looks_like_stub(key):
        pytest.skip(
            "ANTHROPIC_API_KEY is unset or a stub; set a real key in the "
            "environment to exercise this test."
        )
    return key


@pytest.mark.live
@pytest.mark.asyncio
async def test_anthropic_key_is_accepted(live_anthropic_key: str) -> None:
    """
    Real call to Anthropic to verify the configured key is accepted.

    On success the API returns HTTP 200 with a message body. On any
    auth failure the SDK raises ``anthropic.AuthenticationError`` (HTTP
    401) — we surface a redacted key + the upstream error so the
    operator can act on it without leaking the secret.
    """
    from anthropic import AsyncAnthropic, AuthenticationError, APIStatusError

    client = AsyncAnthropic(api_key=live_anthropic_key)

    try:
        response = await client.messages.create(
            model=_LIVE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
    except AuthenticationError as e:
        pytest.fail(
            f"Anthropic rejected key {_redact(live_anthropic_key)} "
            f"(HTTP 401): {e.message}. Check: key still active in console, "
            "billing/credits available, correct workspace."
        )
    except APIStatusError as e:
        pytest.fail(
            f"Anthropic returned non-200 for key {_redact(live_anthropic_key)}: "
            f"HTTP {e.status_code} — {e.message}"
        )

    # Response shape sanity — content is a list of blocks, first is text.
    assert response.content, "Empty response.content from Anthropic"
    assert response.content[0].type == "text", (
        f"Expected text content block, got {response.content[0].type}"
    )
    # Don't assert on the actual text — Claude can return anything to "hi".


@pytest.mark.live
@pytest.mark.asyncio
async def test_production_model_is_accessible(live_anthropic_key: str) -> None:
    """
    The key may authenticate but the configured production model might
    require a plan tier the account doesn't have. This test catches
    that case explicitly so the operator sees "plan upgrade needed"
    rather than chasing a phantom auth bug.
    """
    from anthropic import AsyncAnthropic, AuthenticationError, APIStatusError, NotFoundError

    client = AsyncAnthropic(api_key=live_anthropic_key)

    try:
        await client.messages.create(
            model=_PRODUCTION_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
    except AuthenticationError as e:
        pytest.fail(
            f"Anthropic rejected key {_redact(live_anthropic_key)} for the "
            f"production model {_PRODUCTION_MODEL}: {e.message}. The key may be "
            "scoped to a workspace without access to this model — check the "
            "Anthropic console → Workspaces & Plan."
        )
    except NotFoundError as e:
        pytest.fail(
            f"Production model {_PRODUCTION_MODEL} not available on this account "
            f"(HTTP 404): {e.message}. Likely the plan doesn't include access "
            "to Sonnet 4, or the model identifier is outdated. Update "
            "rag_service/config.py llm_model to a model the account has access to."
        )
    except APIStatusError as e:
        pytest.fail(
            f"Anthropic returned non-200 for {_PRODUCTION_MODEL}: "
            f"HTTP {e.status_code} — {e.message}"
        )


@pytest.mark.live
def test_anthropic_key_format(live_anthropic_key: str) -> None:
    """
    Cheap shape check — runs before the network call so a typo in the
    key is caught without an API round-trip.

    Anthropic keys are 108 characters: ``sk-ant-api03-`` (13 chars) +
    a 95-char base64-ish secret.
    """
    assert live_anthropic_key.startswith("sk-ant-api03-"), (
        f"Key {_redact(live_anthropic_key)} doesn't start with sk-ant-api03-"
    )
    assert len(live_anthropic_key) == 108, (
        f"Key {_redact(live_anthropic_key)} is {len(live_anthropic_key)} chars; "
        "expected 108. Check for stray whitespace or truncation when copying."
    )
