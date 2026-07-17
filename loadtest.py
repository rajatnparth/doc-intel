"""The Module 1 lab artifact: prove the boundary works, with numbers.

    python loadtest.py

Fires 30 concurrent requests at a fault-injecting provider and prints the table
you would show an interviewer:

    p50 · p99 · 429s absorbed · requests shed · tokens billed

"We added resilience" is unfalsifiable. This file is the falsification.

It runs against the STUB, not Azure — deliberately. You cannot ask the real Azure
to 429 you on demand, or to die exactly halfway through a stream. A fault-
injecting fake is the only way to EXERCISE the controls you claim to have built.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import asyncio                          # stdlib — run the concurrent load
import statistics                       # stdlib — median / quantiles for p50, p99
import time                             # stdlib — perf_counter for per-request latency

from app.llm.base import LLMError, RateLimited  # local — app/llm/base.py
from app.llm.resilience import CapacityShed, ResilientCaller  # local — app/llm/resilience.py


class FlakyProvider:
    """A provider that 429s a fraction of calls, like a real quota'd one.

    `fail_rate=0.5` means the first ~50% of attempts get a 429 with a
    Retry-After. Deterministic (no RNG) so the numbers are reproducible.
    """

    def __init__(self, *, fail_first: int = 40, latency: float = 0.05) -> None:
        self._n = 0
        self._fail_first = fail_first
        self._latency = latency
        self.reached = 0                # calls that actually hit "the provider"
        self.tokens_billed = 0

    async def call(self) -> int:
        self._n += 1
        self.reached += 1
        await asyncio.sleep(self._latency)
        if self._n <= self._fail_first:
            # Azure sends Retry-After on a 429. Our policy must honour it.
            raise RateLimited("429 from stub", retry_after=0.05)
        tokens = 180                     # a realistic completion length
        self.tokens_billed += tokens
        return tokens


async def one_request(caller: ResilientCaller, provider: FlakyProvider) -> tuple[float, str]:
    t0 = time.perf_counter()
    try:
        await caller.call(provider.call)
        outcome = "ok"
    except CapacityShed:
        # WE said no, at our door. The provider never saw it.
        # NOTE: this except MUST come first — CapacityShed subclasses RateLimited.
        # The first draft caught RateLimited only, and reported provider-429s as
        # "shed" — a metric that told us to raise our concurrency cap when the
        # real answer was "the provider is out of quota". Ordering is load-bearing.
        outcome = "shed"
    except RateLimited:
        # THE PROVIDER said no, and our retries were exhausted.
        outcome = "429"
    except LLMError:
        outcome = "failed"
    return (time.perf_counter() - t0) * 1000, outcome


async def run(*, concurrency: int, n: int, max_conc: int, retries: int, acquire: float):
    caller = ResilientCaller(
        max_concurrency=max_conc,
        max_retries=retries,
        acquire_timeout=acquire,
        breaker_threshold=100,           # off: we're testing 429 handling, not outage
    )
    provider = FlakyProvider()

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_request(caller, provider) for _ in range(n)))
    wall = time.perf_counter() - t0

    latencies = [ms for ms, _ in results]
    outcomes = [o for _, o in results]
    return caller, provider, latencies, outcomes, wall


def pct(xs: list[float], p: float) -> float:
    return statistics.quantiles(xs, n=100)[int(p) - 1] if len(xs) > 1 else xs[0]


async def main() -> None:
    N = 30

    print("=" * 78)
    print(f"LOAD TEST — {N} concurrent requests against a 429-injecting provider")
    print("=" * 78)
    print("\nThe provider 429s its first 40 attempts (with Retry-After), then succeeds.")
    print("Same load, three boundary configurations.\n")

    configs = [
        ("no controls",      dict(max_conc=1000, retries=0, acquire=10.0)),
        ("retry only",       dict(max_conc=1000, retries=3, acquire=10.0)),
        ("retry + semaphore", dict(max_conc=8,   retries=3, acquire=0.5)),
    ]

    print(f"{'config':<20}{'ok':>4}{'429':>5}{'shed':>6}{'p50 ms':>9}{'p99 ms':>9}"
          f"{'absorbed':>10}{'retries':>9}{'reached':>9}{'tokens':>8}")
    print("-" * 78)

    rows = {}
    for label, cfg in configs:
        caller, provider, lat, out, wall = await run(concurrency=N, n=N, **cfg)
        rows[label] = dict(
            ok=out.count("ok"),
            rl=out.count("429"),
            shed=out.count("shed"),
            p50=pct(lat, 50),
            p99=pct(lat, 99),
            absorbed=caller.stats.rate_limited,
            retries=caller.stats.retries,
            reached=provider.reached,
            tokens=provider.tokens_billed,
        )
        d = rows[label]
        print(
            f"{label:<20}{d['ok']:>4}{d['rl']:>5}{d['shed']:>6}"
            f"{d['p50']:>9.1f}{d['p99']:>9.1f}"
            f"{d['absorbed']:>10}{d['retries']:>9}{d['reached']:>9}{d['tokens']:>8}"
        )

    # ---- the conclusion, COMPUTED. Never hardcoded. -------------------------
    # An earlier draft printed "retry+semaphore: same successes, far fewer calls".
    # The data said 8 ok vs 30 — the opposite. If the prose and the table ever
    # disagree, the table wins, so the prose is now derived FROM the table.
    none_, retry, both = (rows[k] for k, _ in configs)

    print("\n" + "=" * 78)
    print("WHAT THE NUMBERS ACTUALLY SAY")
    print("=" * 78)
    print(f"  no controls : {none_['ok']}/{N} ok — one 429 = one failed user request.")
    print(f"                No absorption at all: {none_['rl']} users just saw an error.")
    print(f"  retry only  : {retry['ok']}/{N} ok — every 429 absorbed ({retry['absorbed']} of them).")
    print(f"                But {retry['reached']} calls hit a provider already saying")
    print(f"                'over quota', and p99 rose {retry['p99'] - none_['p99']:.0f}ms.")
    print(f"  + semaphore : {both['ok']}/{N} ok, {both['shed']} SHED in ~{both['p50']:.0f}ms with Retry-After.")

    if both["ok"] < retry["ok"]:
        print("\n  ⚠️  THE SEMAPHORE SERVED FEWER REQUESTS. That is not a bug, and it is")
        print("      the most interesting number here. Two reasons:")
        print("\n      1. A RETRY HOLDS ITS SLOT WHILE IT SLEEPS. Measured: a request")
        print("         backing off for 0.4s occupies its slot for the full 0.4s.")
        print("         That is CORRECT — the cap counts calls that WILL hit the")
        print("         provider, and a backing-off request certainly will. Release")
        print("         the slot and the cap becomes a lie: 100 requests could all")
        print("         wake at once. So concurrency and retry interact")
        print("         MULTIPLICATIVELY: cap x (1 + retries x backoff) = throughput.")
        print("\n      2. Under sustained overload you CANNOT serve everyone. The real")
        print("         choice is: everyone waits and some time out, OR some are")
        print("         served and the rest are told 'come back in 2s' immediately.")
        print(f"         {both['shed']} clients got an honest answer in {both['p50']:.0f}ms")
        print("         instead of a 60s timeout. That is the trade, stated.")
        print(f"\n      Tune it: raise the cap, or raise acquire_timeout ({0.5}s here)")
        print("      to trade shedding for queueing. Both are defensible; pick with data.")

    print(f"\n  tokens billed: {none_['tokens']} / {retry['tokens']} / {both['tokens']}")
    print("  This is why Usage is a first-class type — it answers 'what does one")
    print("  request cost you?', the follow-up question roughly every time.")


if __name__ == "__main__":
    asyncio.run(main())
