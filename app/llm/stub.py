"""A fake provider that streams tokens and can be told to fail on purpose.

This is not a cop-out for lacking an Azure key. It is how you would test this
code anyway: you cannot ask the real Azure to return a 429 on demand, or to die
exactly 200 tokens into a stream. A fault-injecting stub is the only way to
actually EXERCISE the resilience you claim to have built.

Faults are selected per-call via `FaultMode`, which the endpoint reads from a
query parameter in dev.
"""

import asyncio                          # stdlib — asyncio.sleep (NEVER time.sleep here)
import json                             # stdlib — build the extract() JSON payloads
import random                           # stdlib — seeded RNG for deterministic stub output
from enum import Enum                   # stdlib — FaultMode is an Enum (of str, so it's a query param)
from typing import AsyncIterator        # stdlib — stream_chat's return type

from app.llm.base import (              # local — app/llm/base.py: our error taxonomy + wire types
    ContentFiltered,
    LLMClient,
    ProviderUnavailable,
    RateLimited,
    TokenChunk,
    Usage,
)


class FaultMode(str, Enum):
    """Subclassing `str` as well as `Enum` means FastAPI can accept it directly
    as a query param and Pydantic will validate it, while `FaultMode.NONE == "none"`
    is still True. A very common Python idiom worth recognising."""

    NONE = "none"
    RATE_LIMIT = "rate_limit"          # 429 before any tokens flow
    MID_STREAM_ERROR = "mid_stream"    # 200 OK, tokens flowing, THEN it dies
    CONTENT_FILTER = "content_filter"  # deterministic refusal, never retryable
    HANG = "hang"                      # never responds — proves your timeout works
    BAD_JSON = "bad_json"              # extract() returns schema-invalid JSON once


_LOREM = (
    "A service boundary absorbs what the model cannot promise . "
    "Validation is enforcement and enforcement lives in deterministic code . "
    "Async buys occupancy not latency . "
    "Structured output guarantees shape never truth ."
).split()


class StubLLMClient:
    """Implements the LLMClient Protocol structurally — note it does NOT inherit
    from it. If you delete a method, mypy fails; Python does not."""

    def __init__(
        self,
        *,
        token_delay: float = 0.03,
        seed: int | None = None,
        default_fault: "FaultMode" = None,  # type: ignore[assignment]
    ) -> None:
        self._delay = token_delay
        self._rng = random.Random(seed)
        # A fault applied to every call, so the stub can be injected via
        # `app.dependency_overrides[get_llm]` in tests without the endpoint
        # needing to know faults exist. The Protocol stays clean.
        self._default_fault = default_fault or FaultMode.NONE
        # Counts tokens actually pulled from us.
        self.tokens_generated = 0
        # Set True when our generator is TORN DOWN (aclose() -> GeneratorExit).
        #
        # This flag exists because "we stopped reading" and "we cancelled the
        # provider" look identical from the outside — a suspended generator
        # produces nothing either way. Against real Azure the difference is that
        # one of them is still billing you. So the stub reports, from the inside,
        # whether it was actually closed.
        self.upstream_closed = False
        # How many times extract() has been called with BAD_JSON. Lets us return
        # bad JSON once and good JSON on the repair retry — so we can prove the
        # repair path actually executes, which is a required lab artifact.
        self._bad_json_calls = 0

    # -- streaming -------------------------------------------------------------
    # `async def` + `yield` in the body == an ASYNC GENERATOR FUNCTION.
    # Calling it does not run any of it; it hands back an async iterator.
    # Nothing executes until someone `async for`s over it. That laziness is
    # exactly what "streaming without buffering" means.
    async def stream_chat(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        fault: FaultMode | None = None,
    ) -> AsyncIterator[TokenChunk]:
        fault = fault or self._default_fault
        if fault is FaultMode.RATE_LIMIT:
            # retry_after mimics Azure's header. Our retry policy must honour it.
            raise RateLimited("stub: 429", retry_after=1.5)
        if fault is FaultMode.CONTENT_FILTER:
            raise ContentFiltered("stub: prompt rejected by content filter")
        if fault is FaultMode.HANG:
            # asyncio.sleep, never time.sleep — a blocking sleep here would freeze
            # the event loop and the "hang" fault would hang the whole server,
            # not just this request. The stub practises what the course preaches.
            await asyncio.sleep(3600)

        n_tokens = min(max_tokens, len(_LOREM))
        prompt_tokens = max(1, len(prompt) // 4)  # ~4 chars/token, a rough industry rule

        try:
            for i in range(n_tokens):
                await asyncio.sleep(self._delay)  # yields the loop; other requests proceed

                if fault is FaultMode.MID_STREAM_ERROR and i == n_tokens // 2:
                    # The nasty one. We have ALREADY sent HTTP 200 and half a response.
                    # We cannot now send a 500. The endpoint must translate this into
                    # an in-band SSE error event. That is section 1.3.
                    raise ProviderUnavailable("stub: provider died mid-stream")

                self.tokens_generated += 1
                yield TokenChunk(text=_LOREM[i] + " ")

            # The final chunk carries usage and no text — exactly how Azure behaves
            # when you pass stream_options={"include_usage": True}.
            yield TokenChunk(
                text="",
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=n_tokens),
            )
        except GeneratorExit:
            # Raised INTO this generator, at whichever `yield` it is suspended on,
            # when someone calls `agen.aclose()`. It is how Python says
            # "you are being torn down; clean up now."
            #
            # In AzureLLMClient this is where the underlying HTTP connection to
            # Azure gets closed — which is what actually stops generation, and
            # therefore stops the billing.
            self.upstream_closed = True
            raise  # never swallow GeneratorExit; Python requires it propagate

    # -- structured extraction -------------------------------------------------
    async def extract(
        self,
        text: str,
        schema: dict,
        *,
        max_tokens: int = 512,
        fault: FaultMode | None = None,
    ) -> str:
        fault = fault or self._default_fault
        await asyncio.sleep(self._delay * 5)

        if fault is FaultMode.RATE_LIMIT:
            raise RateLimited("stub: 429", retry_after=1.5)

        if fault is FaultMode.BAD_JSON and self._bad_json_calls == 0:
            self._bad_json_calls += 1
            # Schema-VALID JSON is not the same as CORRECT JSON. Here we return
            # something a naive `json.loads(raw)["invoice_total"]` would happily
            # accept a KeyError on, and that strict Pydantic will reject:
            #   - total as a string (strict=True forbids coercion)
            #   - a currency symbol instead of an ISO code (Literal forbids it)
            #   - page 0 (ge=1 forbids it)
            return json.dumps(
                {
                    "invoice_total": "1,240.50",
                    "currency": "₹",
                    "invoice_number": "INV-001",
                    "source_page": 0,
                }
            )

        return json.dumps(
            {
                "invoice_total": 1240.50,
                "currency": "INR",
                "invoice_number": "INV-001",
                "source_page": 1,
            }
        )

    async def aclose(self) -> None:
        return None


# A no-op assertion that documents intent and fails loudly if the shape drifts.
# runtime_checkable only compares method NAMES, so this is a smoke check, not proof.
assert isinstance(StubLLMClient(), LLMClient)
