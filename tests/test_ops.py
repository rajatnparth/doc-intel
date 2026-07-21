"""Phase 9 — the operability surfaces, as executable claims.

The headline is the strict-audit lifecycle test: a failing sink flips the
health flag on a REAL failed write, admission then refuses new exchanges
with 503 + Retry-After, /ready rotates the pod out — and one successful
write brings everything back without a restart. The flag's whole story,
observed end to end.
"""

import pytest                           # 3rd-party: pytest — fixtures

from fastapi.testclient import TestClient  # 3rd-party: fastapi (submodule) — drives the app

from app.audit import AuditRecord       # local — app/audit.py
from app.auth import mint               # local — app/auth.py
from app.config import get_settings     # local — app/config.py
from app.llm.base import TokenChunk, Usage  # local — app/llm/base.py
from app.main import app, get_audit, get_llm  # local — app/main.py
from app.ops import REGISTRY            # local — app/ops.py (read counters directly)


def auth(tenant: str = "asha") -> dict:
    token = mint(tenant, ["customer"], secret=get_settings().auth_jwt_secret)
    return {"Authorization": f"Bearer {token}"}


class ScriptedLLM:
    async def stream_chat(self, prompt, *, temperature=0.0, max_tokens=512):
        yield TokenChunk(text="Within 24 hours [1]. ")
        yield TokenChunk(text="", usage=Usage(prompt_tokens=40, completion_tokens=5))

    async def extract(self, text, schema, *, max_tokens=512):
        return "not json"

    async def aclose(self) -> None:
        return None


class FailingSink:
    """A sink whose disk is 'full'. get() still works — reads and writes can
    fail independently in real storage, and the flag must key on WRITES."""

    def write(self, record: AuditRecord) -> None:
        raise OSError("disk full")

    def get(self, request_id: str):
        return None

    def probe(self) -> None:
        raise OSError("disk full")


@pytest.fixture
def scripted_llm():
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM()
    yield
    app.dependency_overrides.clear()


def _counter(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels or None) or 0.0


# =============================================================================
# METRICS — the scrape contract
# =============================================================================
def test_ask_outcomes_and_tokens_are_counted(scripted_llm) -> None:
    before_ans = _counter("ask_requests_total", outcome="answer")
    before_tok = _counter("llm_tokens_total", kind="completion")

    with TestClient(app) as client:
        client.post(
            "/v1/ask",
            json={"question": "how quickly must I report an accident?"},
            headers=auth(),
        )
        r = client.get("/metrics")

    assert r.status_code == 200
    assert _counter("ask_requests_total", outcome="answer") == before_ans + 1
    assert _counter("llm_tokens_total", kind="completion") == before_tok + 5
    # The scrape page carries our metrics…
    assert "ask_requests_total" in r.text
    # …and NO tenant identifiers: cardinality + the phase-8 minimization
    # argument, now enforced at the metrics boundary too.
    assert "asha" not in r.text and "tenant" not in r.text


def test_refusals_are_a_first_class_metric(scripted_llm) -> None:
    before = _counter("ask_requests_total", outcome="refusal")
    with TestClient(app) as client:
        client.post(
            "/v1/ask",
            json={"question": "does my policy cover veterinary bills for my dog?"},
            headers=auth(),
        )
    assert _counter("ask_requests_total", outcome="refusal") == before + 1


# =============================================================================
# READINESS — traffic, not restarts
# =============================================================================
def test_ready_reports_ok_on_a_healthy_stack() -> None:
    with TestClient(app) as client:
        r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"ready": True, "store": "ok", "audit": "ok"}


# =============================================================================
# THE STRICT-AUDIT LIFECYCLE — fail, refuse, recover
# =============================================================================
def test_no_record_no_answer_lifecycle(scripted_llm) -> None:
    q = {"question": "how quickly must I report an accident?"}
    with TestClient(app) as client:
        # 1. The write that DISCOVERS the failure: this exchange is served
        #    (the flag was still optimistic) and is the one un-audited
        #    exchange strict mode cannot prevent without pre-writing fakes.
        app.dependency_overrides[get_audit] = lambda: FailingSink()
        first = client.post("/v1/ask", json=q, headers=auth())
        assert first.status_code == 200

        # 2. Admission control: the NEXT exchange is refused at the door —
        #    a clean 503 with Retry-After, before any 200 is spent.
        second = client.post("/v1/ask", json=q, headers=auth())
        assert second.status_code == 503
        assert second.headers["Retry-After"] == "30"
        assert second.json()["error"]["code"] == "audit_unavailable"
        assert second.json()["error"]["retryable"] is True

        # 3. /ready rotates the pod out — same flag, second consumer.
        assert client.get("/ready").status_code == 503
        assert client.get("/ready").json()["audit"] == "failing writes"
        # …while LIVENESS stays green: this pod needs no restart.
        assert client.get("/health").status_code == 200

        # 4. The disk comes back. Note what CANNOT recover us: an exchange —
        #    admission blocks them all while the flag is down (the deadlock
        #    this test originally caught). The READINESS PROBE is the retry
        #    loop: the orchestrator polls /ready anyway, the probe succeeds
        #    against the working sink, the flag flips, traffic resumes.
        app.dependency_overrides.pop(get_audit)
        assert client.get("/ready").status_code == 200      # probe = recovery
        recovered = client.post("/v1/ask", json=q, headers=auth())
        assert recovered.status_code == 200


def test_lenient_mode_serves_but_counts_the_gap(scripted_llm) -> None:
    """AUDIT_STRICT=false: availability outranks the record — an explicit
    operator decision. Exchanges keep flowing, and every unrecorded one is
    on the failure counter, where an alert can see it."""
    lenient = get_settings().model_copy(update={"audit_strict": False})
    app.dependency_overrides[get_settings] = lambda: lenient
    app.dependency_overrides[get_audit] = lambda: FailingSink()
    before = REGISTRY.get_sample_value("audit_write_failures_total") or 0.0

    try:
        with TestClient(app) as client:
            q = {"question": "how quickly must I report an accident?"}
            assert client.post("/v1/ask", json=q, headers=auth()).status_code == 200
            assert client.post("/v1/ask", json=q, headers=auth()).status_code == 200
    finally:
        app.dependency_overrides.pop(get_settings, None)
        app.dependency_overrides.pop(get_audit, None)

    assert (REGISTRY.get_sample_value("audit_write_failures_total") or 0.0) == before + 2
