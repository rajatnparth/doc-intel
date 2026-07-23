"""Phase 4 — numbers never come from RAG, as executable claims.

The headline is `test_fact_question_streams_facts_and_never_calls_the_llm`:
the whole point of the router is a NON-EVENT (no generation), plus an exact
value with the system of record's authority behind it.
"""

import json                             # stdlib — parse SSE frame payloads
from pathlib import Path                # stdlib — read the kits for the consistency test

import pytest                           # 3rd-party: pytest — fixtures

from fastapi.testclient import TestClient  # 3rd-party: fastapi (submodule) — drives the app

from app.auth import mint               # local — app/auth.py
from app.config import get_settings     # local — app/config.py
from app.llm.stub import StubLLMClient  # local — app/llm/stub.py (tier-2 inertness proof)
from app.main import app, get_llm       # local — app/main.py
from app.policy_admin import StubPolicyAdmin  # local — app/policy_admin.py
from app.router import FactField, _parse_decision, classify, route  # local — app/router.py

DOCS = Path(__file__).parent.parent / "sample_docs"


def auth(tenant: str = "asha") -> dict:
    token = mint(tenant, ["customer"], secret=get_settings().auth_jwt_secret)
    return {"Authorization": f"Bearer {token}"}


def events_of(body: str) -> list:
    out = []
    for block in body.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            payload = block[len("data: "):]
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


class ExplodingLLM:
    """Generation, or a tier-2 ROUTING call, is a test failure.

    Used where neither may happen: tier-1 hits (short-circuit) and dated
    questions (router skipped).

    Since phase 11 the ask path has a SECOND llm.extract() consumer — the
    query rewriter — which legitimately runs on wording questions. So this
    fake discriminates by instruction rather than forbidding extract()
    outright: blanket-forbidding it would have quietly turned this test into
    "no LLM call happens", a claim this file never made and does not want.
    Rewrite calls are counted and answered with junk, so transformation
    degrades to the original query and the routing assertions stay exact.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.rewrite_calls = 0

    async def stream_chat(self, prompt, *, temperature=0.0, max_tokens=512):
        self.calls += 1
        raise AssertionError("the facts path called the generator")
        yield  # pragma: no cover — makes this an async generator

    async def extract(self, text, schema, *, max_tokens=512):
        if text.lstrip().startswith("Rewrite one customer question"):
            self.rewrite_calls += 1
            return "not a rewrite verdict"      # fails Gate 2 -> degrades
        raise AssertionError("tier 2 ran where tier 1 or the as_of guard should have decided")

    async def aclose(self) -> None:
        return None


class RoutingLLM:
    """Tier 2 with a scripted verdict. Pins OUR wiring — validation, fallback,
    the facts handoff — not the model's judgment (that needs an eval set)."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.extract_calls = 0

    async def stream_chat(self, prompt, *, temperature=0.0, max_tokens=512):
        raise AssertionError("a routed facts question must never generate")
        yield  # pragma: no cover

    async def extract(self, text, schema, *, max_tokens=512):
        self.extract_calls += 1
        return self.verdict

    async def aclose(self) -> None:
        return None


@pytest.fixture
def exploding_llm():
    fake = ExplodingLLM()
    app.dependency_overrides[get_llm] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


# =============================================================================
# The classifier — the hard case IS the design.
# =============================================================================
def test_value_questions_route_to_facts() -> None:
    assert classify("what is my excess?") is FactField.OWN_DAMAGE_EXCESS
    assert classify("how much is my premium?") is FactField.ANNUAL_PREMIUM
    assert classify("what will next year's renewal premium be?") is FactField.ANNUAL_PREMIUM
    assert classify("what is the IDV of my car?") is FactField.IDV
    assert classify("what is my no-claim bonus?") is FactField.NCB
    assert classify("what's my policy number?") is FactField.POLICY_NUMBER
    assert classify("when is my renewal date?") is FactField.RENEWAL_DATE


def test_wording_questions_fall_through_to_rag() -> None:
    """Same nouns, different questions — the value-shape requirement is what
    keeps a TERMS question away from the record. And the fall-through
    direction fails safe: RAG refuses what it cannot ground; a false route to
    facts would answer a question nobody asked, with authority."""
    assert classify("does the excess apply to windscreen claims?") is None
    assert classify("is windscreen glass replacement covered?") is None
    assert classify("how quickly must I report an accident?") is None
    assert classify("how do I renew my policy online?") is None
    # Known limitation, documented rather than hidden: "what is my renewal
    # process?" WOULD false-route ("what is" + "renew"). The deterministic
    # router buys explainability at the price of edge cases like that — the
    # function-calling upgrade (agents module) is where that price is repaid.
    assert classify("what is my renewal process?") is FactField.RENEWAL_DATE  # the documented wart
    assert classify("what is the status of claim CLM-2026-0891?") is None
    assert classify("what is the limit of liability?") is None


# =============================================================================
# Tier 2 — Gate-2 discipline on the model's verdict.
# =============================================================================
def test_parse_decision_is_a_closed_gate() -> None:
    """Everything that is not a clean, complete facts verdict routes to
    wording — including a syntactically perfect verdict naming a field that
    does not exist. The decision space is an enum, not a text field."""
    assert _parse_decision('{"route": "facts", "field": "own_damage_excess"}') is FactField.OWN_DAMAGE_EXCESS
    assert _parse_decision('{"route": "wording", "field": null}') is None
    assert _parse_decision('{"route": "facts", "field": null}') is None, "facts without a field is not actionable"
    assert _parse_decision('{"route": "facts", "field": "bank_balance"}') is None, "unknown field: closed set"
    assert _parse_decision('{"invoice_total": 1240.5}') is None, "the stub's canned output falls through"
    assert _parse_decision("not json at all") is None


@pytest.mark.asyncio
async def test_tier1_hit_never_spends_a_tier2_call() -> None:
    """The short-circuit is the cost model: explicit phrasing must be free."""
    llm = RoutingLLM('{"route": "wording", "field": null}')
    assert await route("what is my excess?", llm) is FactField.OWN_DAMAGE_EXCESS
    assert llm.extract_calls == 0


@pytest.mark.asyncio
async def test_tier2_routes_the_paraphrase_tier1_cannot_see() -> None:
    """No fact noun anywhere — this is the question that used to leak to RAG
    and get its number FROM PROSE. The scripted verdict stands in for the
    model; what this pins is that a valid verdict actually routes."""
    q = "how much do I pay from my own pocket when I claim?"
    assert classify(q) is None, "tier 1 must genuinely miss, or this test tests nothing"

    llm = RoutingLLM('{"route": "facts", "field": "own_damage_excess"}')
    assert await route(q, llm) is FactField.OWN_DAMAGE_EXCESS
    assert llm.extract_calls == 1


@pytest.mark.asyncio
async def test_tier2_is_inert_with_the_stub_provider() -> None:
    """The keyless quickstart behaves tier-1-only WITHOUT special-casing: the
    stub's canned extract() output fails RouteDecision validation and falls to
    wording. A real provider lights tier 2 up with zero code change."""
    stub = StubLLMClient(token_delay=0.0)
    assert await route("how much do I pay from my own pocket when I claim?", stub) is None


# =============================================================================
# THE HEADLINE: facts come from the record, and no model is involved.
# =============================================================================
def test_fact_question_streams_facts_and_never_calls_the_llm(exploding_llm) -> None:
    with TestClient(app) as client:
        r = client.post("/v1/ask", json={"question": "what is my excess?"}, headers=auth())

    ev = events_of(r.text)
    kinds = [e["type"] if isinstance(e, dict) else e for e in ev]

    assert kinds == ["facts", "done", "[DONE]"], "facts, then a clean close — nothing else"
    facts = ev[0]
    assert facts["source"] == "policy_admin", "the client must SEE this wasn't generated"
    assert facts["policy_number"] == "MTR-2026-1147"
    assert facts["facts"] == [{"name": "Own damage excess", "value": "₹2,000"}]
    assert exploding_llm.calls == 0


def test_facts_are_tenant_scoped_from_the_token(exploding_llm) -> None:
    """Phase 2's chain reaches structured data: the record lookup is keyed by
    the VERIFIED tenant, so Vikram's token gets Vikram's numbers."""
    with TestClient(app) as client:
        r = client.post("/v1/ask", json={"question": "what is my excess?"}, headers=auth("vikram"))

    facts = events_of(r.text)[0]
    assert facts["policy_number"] == "MTR-2026-2210"
    assert facts["facts"][0]["value"] == "₹5,000"


def test_the_measured_false_answer_is_now_an_honest_one(exploding_llm) -> None:
    """calibrate.py measured this exact question scoring 0.7785 through RAG —
    a false answer in waiting (section 2 DISCUSSES premiums; the number isn't
    there). Routed to the record it gets the honest answer: the current
    premium, plus WHEN it changes. Next year's number exists in no subsystem."""
    with TestClient(app) as client:
        r = client.post(
            "/v1/ask",
            json={"question": "what will next year's renewal premium be?"},
            headers=auth(),
        )

    facts = events_of(r.text)[0]
    names = [f["name"] for f in facts["facts"]]
    assert names == ["Annual premium", "Renewal date"]
    assert facts["facts"][0]["value"] == "₹18,900"


def test_the_question_rag_refused_is_now_answered(exploding_llm) -> None:
    """'what is my no-claim bonus?' was in calibrate's UNANSWERABLE set — the
    gate refused it, correctly: the corpus doesn't contain it. The refusal was
    never a dead end; it was the wrong subsystem. The record answers in 0 tokens."""
    with TestClient(app) as client:
        r = client.post("/v1/ask", json={"question": "what is my no-claim bonus?"}, headers=auth())

    facts = events_of(r.text)[0]
    assert facts["facts"] == [{"name": "No-claim bonus", "value": "20%"}]


def test_dated_value_questions_bypass_the_facts_path(exploding_llm) -> None:
    """The stub record knows the CURRENT term only. A question anchored to a
    past date needs the record as of that date — which the effective-dated
    wording archive can serve and this connector cannot. So as_of routes to
    retrieval. (The LLM here explodes on contact; the refusal path doesn't
    reach it, and sources/refusal frames prove which pipeline ran.)"""
    with TestClient(app) as client:
        r = client.post(
            "/v1/ask",
            json={"question": "what is my excess?", "as_of": "2025-12-20"},
            headers=auth(),
        )

    kinds = [e["type"] if isinstance(e, dict) else e for e in events_of(r.text)]
    assert "facts" not in kinds, "a dated question must not be served the current record"


def test_tier2_verdict_reaches_the_record_end_to_end() -> None:
    """HTTP-level: a paraphrased excess question + a scripted facts verdict →
    the facts frame, the record's number, and the generator never runs (the
    RoutingLLM's stream_chat explodes on contact)."""
    fake = RoutingLLM('{"route": "facts", "field": "own_damage_excess"}')
    app.dependency_overrides[get_llm] = lambda: fake
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/ask",
                json={"question": "how much do I pay from my own pocket when I claim?"},
                headers=auth(),
            )
    finally:
        app.dependency_overrides.clear()

    ev = events_of(r.text)
    kinds = [e["type"] if isinstance(e, dict) else e for e in ev]
    assert kinds == ["facts", "done", "[DONE]"]
    assert ev[0]["facts"] == [{"name": "Own damage excess", "value": "₹2,000"}]
    assert fake.extract_calls == 1


# =============================================================================
# The fixture cannot silently drift: where a fact also appears in the wording,
# the record is the source and the kit quotes it.
# =============================================================================
def test_record_and_wording_agree() -> None:
    admin = StubPolicyAdmin()
    asha = admin.get_record("asha")
    vikram = admin.get_record("vikram")
    asha_kit = (DOCS / "asha_policy_kit.md").read_text()
    vikram_kit = (DOCS / "vikram_policy_kit.md").read_text()

    assert asha.own_damage_excess in asha_kit
    assert asha.idv in asha_kit
    assert asha.policy_number in asha_kit
    assert vikram.own_damage_excess in vikram_kit
    assert vikram.idv in vikram_kit
    assert vikram.policy_number in vikram_kit
