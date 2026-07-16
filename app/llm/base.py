"""The seam: what a model provider must be able to do, and how it fails.

This file defines an INTERFACE and nothing else. No Azure, no HTTP, no FastAPI.
"""

from dataclasses import dataclass      # stdlib — @dataclass for the wire types (Usage, TokenChunk)
from typing import AsyncIterator, Protocol, runtime_checkable  # stdlib —
                                        #   Protocol = structural typing (the seam);
                                        #   AsyncIterator = the stream's return type;
                                        #   runtime_checkable = allow isinstance() checks


# =============================================================================
# Errors — a taxonomy, not a list
#
# The retry question is never "which status codes do I retry?" It is:
#   "Will doing this again produce a different answer?"
#
#   429, 500, 503, timeout  -> the system is busy or broken. Later may differ. RETRY.
#   400, 401, content filter -> the system understood you and said no.
#                               Later will not differ. DO NOT RETRY.
#
# Retrying a deterministic refusal burns quota to be told no again.
# =============================================================================
class LLMError(Exception):
    """Base class. Every provider failure is normalised into one of these.

    The route handler must never see an `openai.RateLimitError` or an
    `httpx.ConnectTimeout`. If it does, the provider has leaked through the
    seam and swapping providers becomes a refactor instead of a config change.
    """

    code: str = "llm_error"
    retryable: bool = False

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        # Azure sends a `Retry-After` header on 429. It knows more about when
        # capacity frees up than any backoff formula we invent. Honour it.
        self.retry_after = retry_after


class RateLimited(LLMError):
    code = "rate_limited"
    retryable = True


class ProviderUnavailable(LLMError):
    """5xx, connection errors, timeouts."""

    code = "provider_unavailable"
    retryable = True


class ContentFiltered(LLMError):
    """The provider's safety system refused. Deterministic. Never retry."""

    code = "content_filtered"
    retryable = False


class BadRequest(LLMError):
    """400/422 from the provider. Our bug. Retrying reproduces our bug."""

    code = "bad_request"
    retryable = False


class SchemaRepairFailed(LLMError):
    """The model returned JSON that failed validation twice. Fail closed.

    In a document-intelligence API, NO answer is strictly better than a
    fabricated invoice total.
    """

    code = "schema_repair_failed"
    retryable = False


# =============================================================================
# Wire types
# =============================================================================
@dataclass(frozen=True)
class Usage:
    """Token accounting. Captured from the FINAL chunk of a stream.

    In streaming mode Azure does not send usage by default — you must ask for it
    with stream_options={"include_usage": True}. Skip this and you cannot bill,
    cannot alert on cost, and cannot answer "what does one request cost you?",
    which is the follow-up question roughly every time.
    """

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class TokenChunk:
    """One increment of a streamed completion.

    `usage` is None on every chunk except the last.
    """

    text: str
    usage: Usage | None = None


# =============================================================================
# The Protocol
#
# `Protocol` is Python's structural typing ("if it has these methods, it IS one").
# Unlike an ABC, StubLLMClient does NOT inherit from LLMClient — it simply has
# the right shape. mypy checks the shape statically. Nothing is enforced at
# runtime, which is why @runtime_checkable's isinstance() only checks method
# NAMES, not signatures. Use it for assertions, never for dispatch.
#
# Why a Protocol rather than an abstract base class: it lets us type a third-party
# object we don't control as an LLMClient without touching its class.
# =============================================================================
@runtime_checkable
class LLMClient(Protocol):
    # NOTE: `def`, not `async def` — and this is a real Python subtlety.
    #
    # An async generator function (one with `yield` in an `async def` body) is
    # NOT a coroutine function. Calling it returns an AsyncIterator immediately;
    # you do not await the call, you `async for` over its result:
    #
    #     async for chunk in client.stream_chat(prompt):   # no await here
    #
    # Had we typed this `async def ... -> AsyncIterator[...]`, the signature would
    # mean "await this to GET an iterator" — a different, incompatible shape.
    # Getting this wrong produces `TypeError: 'async_generator' object is not
    # awaitable`, which is a rite of passage.
    def stream_chat(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[TokenChunk]:
        """Yield chunks as they arrive. Must not buffer.

        The return type is AsyncIterator, NOT Awaitable[list[...]].
        The signature itself forbids "collect everything, then return" —
        the naive implementation that passes a demo and fails the interview.
        """
        ...

    async def extract(self, text: str, schema: dict, *, max_tokens: int = 512) -> str:
        """Return the RAW JSON STRING the model produced.

        Deliberately `str`, not `dict`, and certainly not a parsed model.

        Structured outputs guarantee the response is schema-valid. They guarantee
        nothing about whether the values are TRUE. So the client's job ends at
        "here is what the model said"; validation is a separate, deterministic
        gate that runs afterwards, in code the model cannot influence.

        Enforcement belongs outside the model.
        """
        ...

    async def aclose(self) -> None:
        """Release connections. Called from the app's lifespan shutdown."""
        ...
