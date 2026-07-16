"""Smoke tests. These prove the scaffold actually runs, and they double as
executable documentation of section 1.2's claims.

Run: pytest -q
"""

import json                             # stdlib — build/parse JSON payloads in tests

import pytest                           # 3rd-party: pytest — test runner (pytest.raises, fixtures)
from fastapi.testclient import TestClient  # 3rd-party: fastapi — drives the app in-process,
                                        #   runs the lifespan (must use `with TestClient(app)`)
from pydantic import ValidationError    # 3rd-party: pydantic — the exception we assert gets raised

from app.llm.stub import FaultMode, StubLLMClient  # local — app/llm/stub.py
from app.main import app                            # local — app/main.py (the FastAPI instance)
from app.schemas import AskRequest, InvoiceExtract  # local — app/schemas.py


# -----------------------------------------------------------------------------
# The app boots and /health answers without touching a provider.
# -----------------------------------------------------------------------------
def test_health() -> None:
    # `with TestClient(app)` runs the lifespan. Without the `with`, startup never
    # fires and app.state.llm does not exist — a very common test bug.
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "provider": "stub"}


# -----------------------------------------------------------------------------
# GATE 1 — extra="forbid" turns a client's typo into a loud failure.
# -----------------------------------------------------------------------------
def test_typo_in_field_name_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        AskRequest(
            question="what is the total?",
            document_ids=["doc-00000001"],
            temprature=2.0,  # type: ignore[call-arg]  # the typo, on purpose
        )
    # Without extra="forbid" this constructs fine, temperature stays 0.0, and
    # nobody ever finds out.
    assert "temprature" in str(exc.value)


def test_min_length_constrains_the_list_not_the_string() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="q", document_ids=[])          # empty list -> rejected

    with pytest.raises(ValidationError):
        AskRequest(question="q", document_ids=["short"])   # id < 8 chars -> rejected

    ok = AskRequest(question="q", document_ids=["doc-00000001"])
    assert ok.temperature == 0.0


# -----------------------------------------------------------------------------
# GATE 2 — strict mode refuses to guess, and Literal enforces value space.
# -----------------------------------------------------------------------------
def test_strict_mode_refuses_to_coerce_money() -> None:
    good = InvoiceExtract.model_validate_json(
        json.dumps(
            {
                "invoice_total": 1240.50,
                "currency": "INR",
                "invoice_number": "INV-001",
                "source_page": 1,
            }
        )
    )
    assert good.invoice_total == 1240.50

    # Default (non-strict) Pydantic would coerce "1240.50" -> 1240.5 and move on.
    # A value you had to coerce is a value you don't understand.
    with pytest.raises(ValidationError):
        InvoiceExtract.model_validate_json(
            json.dumps(
                {
                    "invoice_total": "1240.50",
                    "currency": "INR",
                    "invoice_number": "INV-001",
                    "source_page": 1,
                }
            )
        )


def test_literal_rejects_a_currency_symbol() -> None:
    # Type validity is not semantic validity. `currency: str` would accept "₹".
    with pytest.raises(ValidationError):
        InvoiceExtract(
            invoice_total=1240.50,
            currency="₹",  # type: ignore[arg-type]
            invoice_number="INV-001",
            source_page=1,
        )


# -----------------------------------------------------------------------------
# The stub streams, and its faults actually fire.
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stub_streams_and_reports_usage() -> None:
    client = StubLLMClient(token_delay=0.0)

    chunks = [c async for c in client.stream_chat("hello", max_tokens=5)]

    # Last chunk carries usage and no text — exactly like Azure with
    # stream_options={"include_usage": True}.
    assert chunks[-1].text == ""
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 5
    assert "".join(c.text for c in chunks).strip() != ""


@pytest.mark.asyncio
async def test_mid_stream_fault_fires_after_tokens_have_flowed() -> None:
    """This is the fault that makes SSE hard: we have already sent HTTP 200."""
    from app.llm.base import ProviderUnavailable

    client = StubLLMClient(token_delay=0.0)
    seen = []
    with pytest.raises(ProviderUnavailable):
        async for chunk in client.stream_chat(
            "hello", max_tokens=8, fault=FaultMode.MID_STREAM_ERROR
        ):
            seen.append(chunk.text)

    assert len(seen) > 0, "the point is that tokens flowed BEFORE it died"


@pytest.mark.asyncio
async def test_bad_json_fault_returns_schema_invalid_payload_once() -> None:
    """Proves the repair-retry path can be exercised: bad once, good after."""
    client = StubLLMClient(token_delay=0.0)
    schema = InvoiceExtract.model_json_schema()

    raw1 = await client.extract("...", schema, fault=FaultMode.BAD_JSON)
    with pytest.raises(ValidationError):
        InvoiceExtract.model_validate_json(raw1)

    raw2 = await client.extract("...", schema, fault=FaultMode.BAD_JSON)
    assert InvoiceExtract.model_validate_json(raw2).currency == "INR"
