"""Section 1.4 — the four controls the boundary absorbs so business logic doesn't.

    breaker.before_call()      should we call AT ALL?   (Azure's health)
      -> semaphore             how many AT ONCE?        (our in-flight count)
        -> retry + jitter      when do we try again?    (honour Retry-After)
          -> timeout           how long do we wait?     (connect + total)

WHY THIS IS ITS OWN FILE, AND NOT IN THE ENDPOINT
-------------------------------------------------
Retry, timeout, concurrency and breaking are properties of the PROVIDER
RELATIONSHIP, not of an endpoint. Smear them across route handlers and:
  - adding a second provider means finding and duplicating them everywhere
  - you cannot apply per-provider quota, because "per-provider" exists nowhere
  - you cannot fail over, because nothing knows what a provider IS
  - your metrics are per-endpoint, when the thing that fails is the PROVIDER

Put them here and a route handler reads like a paragraph of business logic.
That is the answer to "where does this live, and why does that matter when we
add a second model provider next quarter?"

NOTHING HERE IMPORTS FastAPI OR openai. It works on our own error taxonomy, so
it is provider-agnostic by construction.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import asyncio                          # stdlib — Semaphore, sleep, wait_for. NEVER time.sleep.
import logging                          # stdlib — the breaker's state transitions are ops events
import random                           # stdlib — the jitter that de-synchronises retries
import time                             # stdlib — monotonic(), for the breaker's cooldown
from contextlib import asynccontextmanager  # stdlib — slot(): hold a slot across a stream
from dataclasses import dataclass       # stdlib — CallStats
from typing import AsyncIterator, Awaitable, Callable, TypeVar  # stdlib — generic over the call

from app.llm.base import LLMError, ProviderUnavailable, RateLimited  # local — app/llm/base.py

log = logging.getLogger("doc_intel.resilience")

T = TypeVar("T")


class CapacityShed(RateLimited):
    """WE rejected this at OUR door. The provider never saw it.

    Subclasses RateLimited on purpose: to the CLIENT this is a 429 with a
    Retry-After, and it should be handled identically — back off, come back.

    But it is a distinct TYPE because it is a distinct EVENT, and conflating the
    two makes your own metrics lie to you. "429s" that are really "we shed" tells
    you to ask Azure for more quota when the actual answer is "raise your
    concurrency cap" — or the reverse. A metric that can't tell you which system
    said no is not a metric.
    """

    code = "capacity_shed"


# =============================================================================
# Circuit breaker — a fuse box, not a counter
# =============================================================================
class CircuitBreaker:
    """Stops calling a provider that is repeatedly failing.

    The naming trips everyone up, so anchor it to electricity:
      CLOSED    — circuit complete, current flows, calls go through.  HEALTHY.
      OPEN      — circuit broken, no current, calls fail instantly.   TRIPPED.
      HALF_OPEN — let ONE probe through. Succeeds -> CLOSED. Fails -> OPEN.

    HALF_OPEN is the clever part. Without it you either stay open forever, or
    slam 1000 requests at a still-broken provider the instant cooldown ends.

    What it buys: during an outage, "hang 60s then fail" becomes "fail in ~1us".
    Your concurrency slots are then FREE — for cached answers, non-LLM routes, a
    fallback provider. It converts a total outage into a degraded service.
    It also protects the provider: hammering a struggling region with retries is
    how a partial outage becomes a total one.
    """

    def __init__(self, *, threshold: int = 5, cooldown: float = 30.0) -> None:
        self._threshold = threshold
        self._cooldown = cooldown
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self._cooldown:
            return "half_open"
        return "open"

    def before_call(self) -> None:
        """Raise INSTEAD of calling, when the circuit is open.

        Note this is a plain `def`, not `async` — it does no I/O. It's a couple
        of comparisons. Making it async would imply it might yield, which would
        be a lie about what it does.
        """
        state = self.state
        if state == "closed":
            return
        if state == "open":
            raise ProviderUnavailable(
                f"circuit open — {self._failures} consecutive failures, "
                f"retrying in {self._cooldown:.0f}s"
            )
        # HALF_OPEN: allow exactly ONE probe. Everyone else still fails fast.
        if self._half_open_probe_in_flight:
            raise ProviderUnavailable("circuit half-open — a probe is already in flight")
        self._half_open_probe_in_flight = True

    def on_success(self) -> None:
        if self._opened_at is not None:
            log.info("circuit breaker CLOSED — provider recovered")
        self._failures = 0
        self._opened_at = None
        self._half_open_probe_in_flight = False

    def on_failure(self) -> None:
        """Record a failure that indicates the PROVIDER IS UNHEALTHY.

        See `is_health_signal` — the caller decides what counts, and it is a much
        narrower set than "the call failed".
        """
        self._failures += 1
        self._half_open_probe_in_flight = False
        if self._failures >= self._threshold and self._opened_at is None:
            log.warning("circuit breaker OPEN after %d consecutive failures", self._failures)
            self._opened_at = time.monotonic()
        elif self._opened_at is not None:
            # A half-open probe failed. Restart the cooldown — do NOT let a
            # failing provider get probed every request.
            self._opened_at = time.monotonic()


def is_health_signal(exc: LLMError) -> bool:
    """Does this failure mean THE PROVIDER IS SICK?

    ⚠️ THIS FUNCTION EXISTS BECAUSE A TEST CAUGHT ME GETTING IT WRONG.

    The first version tripped the breaker on ANY failure. Then a test sent 5
    BadRequests and the circuit opened — meaning OUR malformed requests had
    disabled OUR OWN access to a provider that was working perfectly. A 400 is
    the provider telling you, correctly and healthily, that you sent rubbish.
    Counting it as ill-health is a self-inflicted outage caused by a client bug.

    So the breaker measures PROVIDER HEALTH, not request failure:

      ProviderUnavailable  ✅  5xx / timeout / connection refused.
                               The provider is genuinely sick. Stop calling.
      RateLimited          ❌  429 is NORMAL OPERATION on a quota'd service —
                               "Azure will return 429. Not might. Will." A breaker
                               that trips on normal operation is a bug. 429 is
                               handled by retry + jitter + Retry-After + the
                               semaphore, which are the right tools for "busy".
      BadRequest           ❌  Our bug. The provider is fine.
      ContentFiltered      ❌  A correct, healthy refusal. The provider is fine.

    Different failures need different machinery. Lumping them together means one
    control fires for reasons it cannot fix.
    """
    return isinstance(exc, ProviderUnavailable)


# =============================================================================
# Retry policy — the taxonomy, applied
# =============================================================================
def compute_delay(attempt: int, exc: LLMError, *, base: float = 1.0) -> float:
    """How long to wait before retry `attempt` (0-indexed).

    TWO RULES, and the first beats the second every time:

    1. HONOUR Retry-After. Azure sends it on a 429. It KNOWS when capacity frees
       up; your backoff formula is a guess in the dark next to a system telling
       you the answer.

    2. Otherwise: exponential + FULL JITTER.

       Jitter is not a nicety. Fifty pods that 429 together and retry after
       exactly 2s produce a synchronised spike every 2s — forever. They were
       DESYNCHRONISED before the failure; the shared 429 aligned them, and an
       identical delay PRESERVES that alignment. Jitter doesn't reduce load,
       it de-synchronises it.
    """
    if exc.retry_after is not None:
        return exc.retry_after
    return (base * (2 ** attempt)) + random.uniform(0, base)


@dataclass
class CallStats:
    """What the boundary is doing. Emitted as metrics in a real system.

    Exists because "add resilience" is unfalsifiable without numbers — these are
    exactly the columns loadtest.py prints.
    """

    attempts: int = 0
    retries: int = 0
    rate_limited: int = 0       # 429s absorbed (retried and eventually succeeded)
    shed: int = 0               # rejected at OUR door — never reached the provider
    breaker_rejections: int = 0
    failures: int = 0


# =============================================================================
# The wrapper that composes all four
# =============================================================================
class ResilientCaller:
    """Wraps any async provider call with the four controls.

    Generic over the call, deliberately: it knows nothing about chat, embeddings
    or reranking. It knows about OUR error taxonomy and nothing else — which is
    why the same instance protects any provider we swap in.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 8,
        max_retries: int = 2,
        acquire_timeout: float = 0.5,
        breaker_threshold: int = 5,
        breaker_cooldown: float = 30.0,
    ) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._acquire_timeout = acquire_timeout
        self.breaker = CircuitBreaker(threshold=breaker_threshold, cooldown=breaker_cooldown)
        self.stats = CallStats()

    async def _acquire_or_shed(self) -> None:
        """Take a semaphore slot, or SHED the request.

        Waiting at OUR door beats waiting at the provider's. But waiting forever
        is still bad: with a 60s timeout, request #992 waits 60s and fails anyway.
        So past `acquire_timeout` we reject with our own 429 + Retry-After.

        Rejecting in 5ms is kinder than timing out in 60s.
        """
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self._acquire_timeout)
        except asyncio.TimeoutError:
            self.stats.shed += 1
            raise CapacityShed(
                "service at capacity — all concurrency slots busy", retry_after=2.0
            ) from None

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold breaker-check + one semaphore slot for the duration of the block.

        Separated from `call()` because STREAMING needs the slot held across the
        whole stream, not just the initial connect. See stream_chat in azure.py.

        ORDER MATTERS: breaker FIRST. No point taking a semaphore slot for a call
        we have already decided not to make. Check health, then capacity.
        """
        try:
            self.breaker.before_call()
        except ProviderUnavailable:
            self.stats.breaker_rejections += 1
            raise

        await self._acquire_or_shed()
        try:
            yield
        finally:
            self._sem.release()          # ALWAYS — even if the body raised.

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """A UNARY call: breaker → semaphore → retry+jitter. Safe to retry whole."""
        async with self.slot():
            return await self.attempt(fn)

    def _record(self, exc: LLMError) -> None:
        """Tell the breaker about a failure — but ONLY if it signals ill-health.

        A BadRequest is our bug; a ContentFiltered is a healthy refusal; a 429 is
        normal operation. None of them mean "stop calling this provider".
        """
        if is_health_signal(exc):
            self.breaker.on_failure()

    async def attempt(self, fn: Callable[[], Awaitable[T]]) -> T:
        """The retry loop ONLY. Assumes you already hold a slot (see `slot()`).

        ⚠️ WHAT YOU MAY AND MAY NOT RETRY
        A unary call (extract) is safe to retry whole — nobody saw the failed one.

        A STREAM is not. Once you have yielded tokens to the client, retrying
        would RE-EMIT tokens they already have. So streaming retries only the
        initial CONNECT, and a mid-stream failure becomes an in-band error frame
        (section 1.3) — never a retry.

        Retry safety is a property of the OPERATION, not of the error. That
        distinction is why this method is public and separate from `call()`.
        """
        last: LLMError | None = None
        for attempt in range(self._max_retries + 1):     # attempts = retries + 1
            self.stats.attempts += 1
            try:
                result = await fn()
                self.breaker.on_success()
                return result

            except LLMError as exc:
                # THE PRINCIPLE, not a list of status codes:
                #   "Will doing this again produce a different answer?"
                #   busy/broken (429, 5xx, timeout) -> later may differ -> RETRY
                #   understood-you-and-said-no (400, 401, content filter)
                #       -> deterministic -> NEVER. You'd burn quota to be told
                #          no again, forever.
                if not exc.retryable:
                    self._record(exc)
                    self.stats.failures += 1
                    raise

                last = exc
                if isinstance(exc, RateLimited):
                    self.stats.rate_limited += 1

                if attempt == self._max_retries:
                    self._record(exc)
                    self.stats.failures += 1
                    raise

                self.stats.retries += 1
                delay = compute_delay(attempt, exc)
                log.info(
                    "retrying after %s: %.2fs (attempt %d)",
                    exc.code, delay, attempt + 1,
                )
                await asyncio.sleep(delay)   # asyncio, NEVER time.sleep (1.1)

        raise last or ProviderUnavailable("retries exhausted")  # pragma: no cover
