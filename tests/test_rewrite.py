"""Phase 11 — query transformation, as executable claims.

The headline is NOT "rewriting improves retrieval" — that is a quality claim
and belongs to the eval harness, where it is measured on labelled data
(`python -m evals.ablation`). The claims here are the ENGINEERING ones, and
they are the reason this feature is safe to switch on by default:

  1. every failure degrades to the original query — never worse than before
  2. the model's output passes Gate 2 or is discarded whole
  3. the cache never serves a degraded result, and never crosses a boundary
     it shouldn't (the argument lives in app/retrieval/rewrite_cache.py)
"""

import json                             # stdlib — building verdicts
import asyncio                          # stdlib — driving the async transform

import pytest                           # 3rd-party: pytest

from app.llm.base import LLMError, RateLimited  # local — app/llm/base.py
from app.retrieval.rewrite import MAX_VARIANTS, TransformedQuery, transform  # local
from app.retrieval.rewrite_cache import RewriteCache  # local


class FakeLLM:
    """Returns a scripted extract() verdict and counts calls."""

    def __init__(self, verdict: str | Exception) -> None:
        self.verdict = verdict
        self.calls = 0

    async def stream_chat(self, prompt, *, temperature=0.0, max_tokens=512):
        raise AssertionError("the rewriter must never generate")
        yield  # pragma: no cover

    async def extract(self, text, schema, *, max_tokens=512):
        self.calls += 1
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict

    async def aclose(self) -> None:
        return None


def _run(coro):
    return asyncio.run(coro)


GOOD = json.dumps({
    "rewrites": ["what is the annual usage limit?", "how many claims per year are allowed?"],
    "hypothetical_answer": "Assistance is available up to 4 times in a year.",
})


# =============================================================================
# The happy path
# =============================================================================
def test_transform_returns_original_first_then_variants() -> None:
    t = _run(transform("how many times can I avail it?", FakeLLM(GOOD)))
    assert not t.degraded
    # The user's words are ALWAYS searched: a rewrite can drift, and the
    # original is the only phrasing we know the user meant.
    assert t.all_queries[0] == "how many times can I avail it?"
    assert "what is the annual usage limit?" in t.all_queries
    # The HyDE probe rides along as a query — never as prompt content.
    assert "Assistance is available up to 4 times in a year." in t.all_queries


def test_variants_are_capped_and_deduplicated() -> None:
    noisy = json.dumps({
        "rewrites": ["a", "A", " a ", "b", "c", "d", "e"],   # dupes + over cap
        "hypothetical_answer": "",
    })
    t = _run(transform("q", FakeLLM(noisy), max_variants=2))
    assert len(t.variants) <= 2
    assert len({v.strip().lower() for v in t.variants}) == len(t.variants)


# =============================================================================
# FAIL-SAFE — the claims that make this default-on
# =============================================================================
@pytest.mark.parametrize(
    "verdict",
    [
        RateLimited("429"),                     # provider shedding load
        LLMError("boom"),                       # anything else in the taxonomy
        "not json at all",                      # unparseable
        '{"rewrites": "not a list"}',           # wrong shape -> Gate 2 rejects
        '{"rewrites": [], "hypothetical_answer": ""}',   # empty verdict
        '{"rewrites": ["x"], "evil": "extra"}',  # extra field -> forbidden
    ],
)
def test_every_failure_degrades_to_the_original_query(verdict) -> None:
    """The whole safety argument in one test: whatever the provider does,
    the pipeline still runs the user's question — pre-phase-11 behaviour."""
    t = _run(transform("what is my excess?", FakeLLM(verdict)))
    assert t.degraded is True
    assert t.all_queries == ["what is my excess?"]


def test_disabled_costs_nothing() -> None:
    llm = FakeLLM(GOOD)
    t = _run(transform("q", llm, enabled=False))
    assert llm.calls == 0, "a disabled rewriter must not call the provider"
    assert t.all_queries == ["q"]


def test_the_rewriter_never_generates() -> None:
    """It uses extract(), never stream_chat(): a rewrite is structured data
    behind Gate 2, not prose. FakeLLM asserts on any generation call."""
    _run(transform("q", FakeLLM(GOOD)))


def test_hard_ceiling_survives_a_bad_config() -> None:
    t = _run(transform("q", FakeLLM(GOOD), max_variants=999))
    assert len(t.variants) <= MAX_VARIANTS


# =============================================================================
# The cache
# =============================================================================
def test_cache_round_trips_and_counts() -> None:
    c = RewriteCache(maxsize=4)
    value = TransformedQuery(original="q", variants=["v"])
    assert c.get("q", 3) is None
    c.put("q", 3, value)
    assert c.get("q", 3) is value
    assert (c.hits, c.misses) == (1, 1)


def test_cache_keys_on_the_variant_count_too() -> None:
    """The count changes the RESULT, so it belongs in the key — the same
    lesson the as_of gate taught the view cache in phase 5."""
    c = RewriteCache()
    c.put("q", 3, TransformedQuery(original="q", variants=["a", "b", "c"]))
    assert c.get("q", 1) is None, "a 3-variant entry must not serve a 1-variant request"


def test_degraded_results_are_never_cached() -> None:
    """Caching a degradation turns a transient provider outage into a
    permanent quality regression that outlives the incident."""
    c = RewriteCache()
    c.put("q", 3, TransformedQuery(original="q", degraded=True))
    assert c.get("q", 3) is None


def test_cache_is_bounded_lru() -> None:
    c = RewriteCache(maxsize=2)
    for q in ("a", "b", "c"):
        c.put(q, 3, TransformedQuery(original=q, variants=[q]))
    assert len(c) == 2
    assert c.get("a", 3) is None, "the oldest entry must be evicted"


def test_cache_can_be_switched_off() -> None:
    """QUERY_REWRITE_CACHE_SIZE=0 — for an environment where holding
    question text in memory is unacceptable at all."""
    c = RewriteCache(maxsize=0)
    c.put("q", 3, TransformedQuery(original="q", variants=["v"]))
    assert c.get("q", 3) is None
    assert len(c) == 0
