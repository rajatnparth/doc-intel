"""Section 1.4 — the four controls, as executable claims.

"We added resilience" is unfalsifiable. Each test here names one control and
fails if that control is removed.
"""

import asyncio                          # stdlib — drive the async wrapper, measure timing
import time                             # stdlib — monotonic, for concurrency/timing assertions

import pytest                           # 3rd-party: pytest — raises, fixtures

from app.llm.base import (              # local — app/llm/base.py: our error taxonomy
    BadRequest,
    ContentFiltered,
    ProviderUnavailable,
    RateLimited,
)
from app.llm.resilience import (        # local — app/llm/resilience.py
    CapacityShed,
    CircuitBreaker,
    ResilientCaller,
    compute_delay,
    is_health_signal,
)


def _fail_with(exc, times: int, then=None):
    """A fake provider call: raise `exc` for the first `times` calls, then succeed."""
    state = {"n": 0}

    async def fn():
        state["n"] += 1
        if state["n"] <= times:
            raise exc
        return then if then is not None else "ok"

    fn.calls = state                    # type: ignore[attr-defined]
    return fn


# =============================================================================
# CONTROL 2a — the retry TAXONOMY. The principle, not the list.
# =============================================================================
@pytest.mark.parametrize("exc", [
    BadRequest("your bug"),
    ContentFiltered("deterministic refusal"),
])
async def test_deterministic_refusals_are_never_retried(exc) -> None:
    """"Will doing this again produce a different answer?" No -> don't.

    The provider understood you and said no. Retrying burns quota to be told no
    again, forever. This is the single most expensive mistake in the section.
    """
    r = ResilientCaller(max_retries=3)
    fn = _fail_with(exc, times=99)

    with pytest.raises(type(exc)):
        await r.call(fn)

    assert fn.calls["n"] == 1, "called ONCE — no retries on a deterministic refusal"
    assert r.stats.retries == 0


async def test_transient_failures_are_retried_then_succeed() -> None:
    """Busy or broken -> later may differ -> retry."""
    r = ResilientCaller(max_retries=3)
    fn = _fail_with(RateLimited("429", retry_after=0.0), times=2)

    assert await r.call(fn) == "ok"
    assert fn.calls["n"] == 3, "2 failures + 1 success"
    assert r.stats.retries == 2
    assert r.stats.rate_limited == 2


async def test_retries_are_bounded() -> None:
    """attempts = retries + 1. Retrying forever is not resilience, it's a loop."""
    r = ResilientCaller(max_retries=2)
    fn = _fail_with(RateLimited("429", retry_after=0.0), times=99)

    with pytest.raises(RateLimited):
        await r.call(fn)
    assert fn.calls["n"] == 3, "1 initial + 2 retries"


# =============================================================================
# CONTROL 2b — Retry-After beats your formula. And jitter DE-SYNCHRONISES.
# =============================================================================
def test_retry_after_overrides_our_backoff() -> None:
    """Azure KNOWS when capacity frees up. Our formula guesses."""
    exc = RateLimited("429", retry_after=7.5)
    assert compute_delay(attempt=0, exc=exc) == 7.5
    assert compute_delay(attempt=5, exc=exc) == 7.5, "even at attempt 5 — it still knows better"


def test_delay_grows_exponentially_when_no_retry_after() -> None:
    exc = ProviderUnavailable("503")            # no retry_after header
    lows = [compute_delay(a, exc, base=1.0) for a in range(4)]
    assert lows[0] < lows[3], "backoff must grow"
    for a, d in enumerate(lows):
        assert 2**a <= d <= 2**a + 1.0, "exponential + full jitter within [2^a, 2^a + base]"


def test_jitter_desynchronises_a_thundering_herd() -> None:
    """THE point of jitter, and it is not "reduce load".

    Fifty pods 429 together. They were DESYNCHRONISED before the failure — the
    shared 429 aligned them. A deterministic delay PRESERVES that alignment, so
    they spike together every cycle, forever. Jitter breaks the alignment.
    """
    exc = ProviderUnavailable("503")
    delays = [compute_delay(attempt=1, exc=exc, base=1.0) for _ in range(50)]

    assert len(set(delays)) > 45, "50 pods must NOT pick the same delay"
    assert max(delays) - min(delays) > 0.5, "and they must be genuinely spread"

    # Contrast: this is what deterministic backoff does. Kept as the control.
    deterministic = [2 ** 1 for _ in range(50)]
    assert len(set(deterministic)) == 1, "…all fifty fire at the same instant"


# =============================================================================
# CONTROL 3 — the semaphore caps concurrency and SHEDS the overflow
# =============================================================================
async def test_semaphore_caps_in_flight_calls() -> None:
    """8 wristbands means at most 8 inside, however many are queued."""
    r = ResilientCaller(max_concurrency=3, acquire_timeout=5.0)
    in_flight = 0
    peak = 0

    async def slow():
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return "ok"

    await asyncio.gather(*(r.call(slow) for _ in range(12)))
    assert peak <= 3, f"cap was 3, but {peak} were in flight at once"


async def test_overflow_is_shed_fast_not_queued_to_death() -> None:
    """Rejecting in 5ms is kinder than timing out in 60s.

    The client can back off sensibly on a fast 429 + Retry-After. A 60-second
    timeout burns the user's patience, your socket AND the provider's quota, and
    then fails anyway.
    """
    r = ResilientCaller(max_concurrency=1, acquire_timeout=0.05)

    async def slow():
        await asyncio.sleep(0.4)
        return "ok"

    t0 = time.monotonic()
    results = await asyncio.gather(
        *(r.call(slow) for _ in range(4)), return_exceptions=True
    )
    elapsed = time.monotonic() - t0

    shed = [x for x in results if isinstance(x, RateLimited)]
    assert len(shed) == 3, "1 served, 3 shed"
    assert all(e.retry_after for e in shed), "a shed request MUST say when to come back"
    assert elapsed < 0.4 + 0.2, "shed requests failed FAST — they did not queue for 60s"
    assert r.stats.shed == 3


async def test_our_shed_is_distinguishable_from_the_providers_429() -> None:
    """Both are a 429 to the client. They are NOT the same event.

    The loadtest reported provider-429s as "shed" until CapacityShed existed —
    a metric that says "raise your concurrency cap" when the real answer is
    "the provider is out of quota". A metric that can't tell you WHICH SYSTEM
    said no is not a metric.

    CapacityShed subclasses RateLimited on purpose: the client should handle it
    identically (back off, retry). Only YOUR dashboards need to tell them apart.
    """
    assert issubclass(CapacityShed, RateLimited), "the client sees a 429 either way"

    r = ResilientCaller(max_concurrency=1, max_retries=0, acquire_timeout=0.02)

    async def slow():
        await asyncio.sleep(0.2)
        return "ok"

    results = await asyncio.gather(*(r.call(slow) for _ in range(3)), return_exceptions=True)
    shed = [x for x in results if isinstance(x, CapacityShed)]
    assert len(shed) == 2
    assert all(e.code == "capacity_shed" for e in shed), "distinct code for OUR rejection"


async def test_a_retry_holds_its_semaphore_slot_while_backing_off() -> None:
    """A DESIGN DECISION, pinned — the loadtest surfaced it.

    A request sleeping in retry backoff KEEPS its concurrency slot. Measured
    below: B waits the full duration of A's backoff.

    That is CORRECT, not a leak. The semaphore caps calls that WILL hit the
    provider, and a backing-off request certainly will. Release the slot during
    backoff and the cap becomes a lie — 100 requests could all wake at once and
    blow straight past it.

    The consequence you must be able to state: concurrency and retry interact
    MULTIPLICATIVELY. cap x (1 + retries x backoff) bounds your throughput, which
    is why adding a semaphore to a retrying client can REDUCE completions.
    """
    r = ResilientCaller(max_concurrency=1, max_retries=2, acquire_timeout=5.0)
    state = {"n": 0}
    t0 = time.monotonic()

    async def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise RateLimited("429", retry_after=0.2)   # a 0.2s backoff
        return "ok"

    async def other():
        await asyncio.sleep(0.01)                        # ensure A holds the slot first
        await r.call(lambda: asyncio.sleep(0, result="ok"))
        return time.monotonic() - t0

    _, waited = await asyncio.gather(r.call(flaky), other())
    assert waited >= 0.2, (
        f"B got the slot after {waited:.2f}s — it must wait out A's 0.2s backoff, "
        "because A holds the slot while sleeping"
    )


async def test_semaphore_slot_is_released_on_failure() -> None:
    """The classic leak: release only on success -> the pool drains to zero and
    the app deadlocks. `finally` is not optional."""
    r = ResilientCaller(max_concurrency=2, max_retries=0, acquire_timeout=0.05)

    for _ in range(5):
        with pytest.raises(BadRequest):
            await r.call(_fail_with(BadRequest("nope"), times=99))

    # If slots leaked, this shed instead of running.
    assert await r.call(_fail_with(RateLimited("x"), times=0)) == "ok"


# =============================================================================
# CONTROL 4 — WHAT counts as ill-health. This test found a real bug in my code.
# =============================================================================
@pytest.mark.parametrize("exc,expected,why", [
    (ProviderUnavailable("503"), True, "5xx/timeout — the provider is genuinely sick"),
    (RateLimited("429"), False, "429 is NORMAL operation on a quota'd service"),
    (BadRequest("our bug"), False, "the provider is fine; WE sent rubbish"),
    (ContentFiltered("no"), False, "a correct, healthy refusal"),
])
def test_only_provider_illness_trips_the_breaker(exc, expected, why) -> None:
    assert is_health_signal(exc) is expected, why


async def test_our_own_bad_requests_do_not_disable_a_healthy_provider() -> None:
    """THE BUG THIS FILE CAUGHT.

    The first version of on_failure() counted ANY failure. Then the slot-release
    test above sent 5 BadRequests and the circuit OPENED — meaning our own
    malformed requests had disabled our access to a provider that was working
    perfectly.

    A 400 is the provider telling you, healthily and correctly, that you sent
    rubbish. Counting it as ill-health turns a client bug into a self-inflicted
    outage. The breaker measures PROVIDER HEALTH, not request failure.
    """
    r = ResilientCaller(max_retries=0, breaker_threshold=3)

    for _ in range(10):
        with pytest.raises(BadRequest):
            await r.call(_fail_with(BadRequest("malformed"), times=99))

    assert r.breaker.state == "closed", (
        "10 of OUR bugs must not open the circuit on a healthy provider"
    )
    # …and a good request still goes straight through.
    assert await r.call(_fail_with(RateLimited("x"), times=0)) == "ok"


async def test_sustained_429s_do_not_trip_the_breaker() -> None:
    """"Azure will return 429. Not might. Will." A breaker that trips on NORMAL
    OPERATION is a bug. 429 is handled by retry + jitter + Retry-After + the
    semaphore — the tools that fit "busy". The breaker is for "broken"."""
    r = ResilientCaller(max_retries=0, breaker_threshold=3)

    for _ in range(10):
        with pytest.raises(RateLimited):
            await r.call(_fail_with(RateLimited("429", retry_after=0.0), times=99))

    assert r.breaker.state == "closed", "429 is busy, not sick"


# =============================================================================
# CONTROL 4 — the circuit breaker
# =============================================================================
def test_breaker_opens_after_consecutive_failures() -> None:
    b = CircuitBreaker(threshold=3, cooldown=30.0)
    assert b.state == "closed"

    for _ in range(2):
        b.on_failure()
    assert b.state == "closed", "2 of 3 — not yet"

    b.on_failure()
    assert b.state == "open", "3rd consecutive failure trips it"

    with pytest.raises(ProviderUnavailable):
        b.before_call()                 # fails in ~1us, never touches the provider


def test_success_resets_the_failure_run() -> None:
    """CONSECUTIVE failures. One success and the count is zero — otherwise a
    healthy provider with occasional blips would eventually trip."""
    b = CircuitBreaker(threshold=3)
    b.on_failure()
    b.on_failure()
    b.on_success()
    b.on_failure()
    b.on_failure()
    assert b.state == "closed", "the run was broken by a success"


def test_breaker_half_opens_after_cooldown_and_probes_once() -> None:
    """HALF_OPEN is the clever bit: without it you either stay open forever, or
    slam 1000 requests at a still-broken provider the instant cooldown ends."""
    b = CircuitBreaker(threshold=1, cooldown=0.05)
    b.on_failure()
    assert b.state == "open"

    time.sleep(0.06)
    assert b.state == "half_open"

    b.before_call()                     # the ONE probe gets through
    with pytest.raises(ProviderUnavailable):
        b.before_call()                 # everyone else still fails fast


def test_half_open_probe_failure_reopens_and_resets_cooldown() -> None:
    b = CircuitBreaker(threshold=1, cooldown=0.05)
    b.on_failure()
    time.sleep(0.06)
    b.before_call()                     # probe
    b.on_failure()                      # …probe failed
    assert b.state == "open", "back to open — do NOT probe every request"


def test_half_open_probe_success_closes_the_circuit() -> None:
    b = CircuitBreaker(threshold=1, cooldown=0.05)
    b.on_failure()
    time.sleep(0.06)
    b.before_call()
    b.on_success()
    assert b.state == "closed", "provider recovered"


async def test_open_breaker_fails_fast_without_calling_the_provider() -> None:
    """What the breaker BUYS: during an outage, "hang 60s then fail" becomes
    "fail instantly", and your concurrency slots are FREE."""
    r = ResilientCaller(max_retries=0, breaker_threshold=2, breaker_cooldown=30.0)
    fn = _fail_with(ProviderUnavailable("503"), times=99)

    for _ in range(2):
        with pytest.raises(ProviderUnavailable):
            await r.call(fn)
    calls_before = fn.calls["n"]

    t0 = time.monotonic()
    for _ in range(50):
        with pytest.raises(ProviderUnavailable):
            await r.call(fn)
    elapsed = time.monotonic() - t0

    assert fn.calls["n"] == calls_before, "50 requests, ZERO reached the provider"
    assert elapsed < 0.05, "and they failed in microseconds, not 60s each"
    assert r.stats.breaker_rejections == 50
