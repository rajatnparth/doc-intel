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
from app.main import app, get_llm       # local — app/main.py
from app.policy_admin import StubPolicyAdmin  # local — app/policy_admin.py
from app.router import FactField, classify  # local — app/router.py

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
    """Any call is a test failure. The facts path must never generate."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, prompt, *, temperature=0.0, max_tokens=512):
        self.calls += 1
        raise AssertionError("the facts path called the generator")
        yield  # pragma: no cover — makes this an async generator

    async def extract(self, text, schema, *, max_tokens=512):
        raise AssertionError("extract must not be called")

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
