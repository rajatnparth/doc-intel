"""Phase 6 — audit + handoff, as executable claims.

The headline is `test_handoff_is_tenant_gated`: request ids leak (headers,
logs, screenshots), so the audit lookup behind /v1/handoff must verify the
record belongs to the CALLER'S tenant — and must answer 404 either way,
because a 403 on a foreign id confirms the id exists.
"""

import json                             # stdlib — parse SSE frames + audit lines

import pytest                           # 3rd-party: pytest — fixtures

from fastapi.testclient import TestClient  # 3rd-party: fastapi (submodule) — drives the app

from app.audit import AuditRecord, JsonlAuditSink  # local — app/audit.py
from app.auth import mint               # local — app/auth.py (dev token minter)
from app.config import get_settings      # local — app/config.py
from app.llm.base import TokenChunk, Usage  # local — app/llm/base.py (wire types)
from app.main import app, get_llm       # local — app/main.py


def auth(tenant: str = "asha", groups: tuple[str, ...] = ("customer",)) -> dict:
    token = mint(tenant, list(groups), secret=get_settings().auth_jwt_secret)
    return {"Authorization": f"Bearer {token}"}


class ScriptedLLM:
    """Streams a fixed answer; conforms to LLMClient by shape."""

    async def stream_chat(self, prompt, *, temperature=0.0, max_tokens=512):
        for word in ["Report", "within", "24", "hours", "[1]."]:
            yield TokenChunk(text=word + " ")
        yield TokenChunk(text="", usage=Usage(prompt_tokens=100, completion_tokens=5))

    async def extract(self, text, schema, *, max_tokens=512):
        return "not json"                # tier-2 router verdicts fail closed -> RAG

    async def aclose(self) -> None:
        return None


@pytest.fixture
def scripted_llm():
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM()
    yield
    app.dependency_overrides.clear()


def _record_for(client, request_id: str) -> AuditRecord | None:
    return client.app.state.audit.get(request_id)


# =============================================================================
# AUDIT — one record per exchange, saying what was DELIVERED
# =============================================================================
def test_answer_outcome_records_delivered_text_and_cost(scripted_llm) -> None:
    with TestClient(app) as client:
        r = client.post(
            "/v1/ask",
            json={"question": "how quickly must I report an accident?"},
            headers=auth(),
        )
        rec = _record_for(client, r.headers["x-request-id"])

    assert rec is not None, "every exchange must leave a record"
    assert rec.outcome == "answer"
    assert rec.tenant_id == "asha"
    # The record holds the words the customer saw — assembled at stream END.
    assert rec.answer_text == "Report within 24 hours [1]. "
    assert rec.prompt_tokens == 100 and rec.completion_tokens == 5
    assert rec.sources, "an answered exchange records what was cited"
    assert rec.rerank_score is not None and rec.threshold is not None


def test_refusal_outcome_records_the_gate_decision(scripted_llm) -> None:
    with TestClient(app) as client:
        r = client.post(
            "/v1/ask",
            json={"question": "does my policy cover veterinary bills for my dog?"},
            headers=auth(),
        )
        rec = _record_for(client, r.headers["x-request-id"])

    assert rec is not None
    assert rec.outcome == "refusal"
    # The replayable WHY: the score the gate read, the threshold it missed,
    # and the near-misses it considered. "We refused" alone defends nothing.
    assert rec.rerank_score is not None and rec.rerank_score < rec.threshold
    assert rec.refusal_reason
    assert rec.retrieved, "near-misses are part of the decision"
    assert rec.answer_text == "", "nothing was generated, and the record says so"


def test_facts_outcome_records_values_and_no_model_cost(scripted_llm) -> None:
    with TestClient(app) as client:
        r = client.post(
            "/v1/ask",
            json={"question": "what is my excess for an own damage claim?"},
            headers=auth(),
        )
        rec = _record_for(client, r.headers["x-request-id"])

    assert rec is not None
    assert rec.outcome == "facts"
    assert rec.facts and any("₹" in f.value for f in rec.facts)
    # No model ran: no tokens, no cost — the ABSENCE is audit information.
    assert rec.prompt_tokens is None and rec.completion_tokens is None


def test_sink_is_append_only_one_line_per_exchange(tmp_path) -> None:
    sink = JsonlAuditSink(str(tmp_path / "audit.jsonl"))
    a = AuditRecord(
        request_id="req-aaaa", at=__import__("app.audit", fromlist=["now_utc"]).now_utc(),
        tenant_id="asha", groups=["customer"], question="q1", outcome="refusal",
    )
    b = a.model_copy(update={"request_id": "req-bbbb", "question": "q2"})
    sink.write(a)
    sink.write(b)

    lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(l)["request_id"] for l in lines] == ["req-aaaa", "req-bbbb"]
    # And each is retrievable by id — the handoff endpoint's read path.
    assert sink.get("req-aaaa").question == "q1"
    assert sink.get("req-bbbb").question == "q2"
    assert sink.get("req-cccc") is None


# =============================================================================
# HANDOFF — a refusal becomes a ticket that REFERENCES the audit record
# =============================================================================
def test_refusal_to_handoff_full_loop(scripted_llm) -> None:
    with TestClient(app) as client:
        r = client.post(
            "/v1/ask",
            json={"question": "does my policy cover veterinary bills for my dog?"},
            headers=auth(),
        )
        rid = r.headers["x-request-id"]

        h = client.post(
            "/v1/handoff",
            json={"request_id": rid, "note": "please call me about this"},
            headers=auth(),
        )

        assert h.status_code == 201
        body = h.json()
        assert body["request_id"] == rid
        assert body["status"] == "open"

        # The ticket carries the REFERENCE plus queue-view context — the
        # conversation itself stays in the audit trail, single source of truth.
        ticket = client.app.state.tickets.get(body["ticket_id"])
        assert ticket is not None
        assert ticket.tenant_id == "asha"
        assert ticket.question == "does my policy cover veterinary bills for my dog?"
        assert "score" in ticket.reason, "the refusal's WHY travels to the agent"
        assert ticket.note == "please call me about this"


def test_handoff_is_tenant_gated(scripted_llm) -> None:
    """Vikram, fully authenticated, replays ASHA'S request id. 404 — and
    indistinguishable from an id that never existed."""
    with TestClient(app) as client:
        r = client.post(
            "/v1/ask",
            json={"question": "does my policy cover veterinary bills for my dog?"},
            headers=auth("asha"),
        )
        asha_rid = r.headers["x-request-id"]

        foreign = client.post(
            "/v1/handoff", json={"request_id": asha_rid}, headers=auth("vikram")
        )
        unknown = client.post(
            "/v1/handoff", json={"request_id": "never-existed-0000"}, headers=auth("vikram")
        )

    assert foreign.status_code == 404
    assert unknown.status_code == 404
    # Same code AND same body shape: the foreign case must not be
    # distinguishable from the nonexistent case, or existence leaks.
    assert foreign.json()["error"]["code"] == unknown.json()["error"]["code"] == "not_found"


def test_handoff_requires_auth() -> None:
    with TestClient(app) as client:
        r = client.post("/v1/handoff", json={"request_id": "whatever-1234"})
    assert r.status_code == 401
