"""The contracts. Everything from section 1.2 lives here.

Two boundaries, two sets of models:

  GATE 1  client  -> us    : AskRequest / AskResponse / ErrorEnvelope
  GATE 2  the LLM -> us    : InvoiceExtract

They look similar and they are NOT the same thing. Gate 1 protects us from
clients. Gate 2 protects us from the model, whose output is untrusted input.
"""

from datetime import date, datetime    # stdlib — AskRequest.as_of + handoff created_at
from typing import Annotated, Literal   # stdlib — Annotated attaches metadata to a type;
                                        #   Literal restricts a value to a fixed set

from pydantic import BaseModel, ConfigDict, Field  # 3rd-party: pydantic — the whole
                                        #   validation layer: base class, per-model config,
                                        #   field constraints

# -----------------------------------------------------------------------------
# Reusable constrained types.
#
# `Annotated[X, Field(...)]` = "the type is X, and here is metadata about it".
# Python itself ignores the metadata entirely; Pydantic reads it.
#
# Defining these once means every model in the codebase validates IDs the same
# way. Types are a place to put decisions, not just labels.
# -----------------------------------------------------------------------------
DocumentId = Annotated[str, Field(min_length=8, max_length=64)]
TenantId = Annotated[str, Field(min_length=1, max_length=64)]


# =============================================================================
# GATE 1 — the REST contract
# =============================================================================
class AskRequest(BaseModel):
    # extra="forbid": a client sending {"temprature": 2.0} gets a 422 naming the
    # field, instead of a silent no-op where we run at the default and they swear
    # our API ignores them. Fail loudly at the boundary; fail nowhere else.
    model_config = ConfigDict(extra="forbid")

    # max_length on free text is not just DoS hygiene here. It is a COST control
    # and a context-window control. An unbounded question is an unbounded bill.
    question: str = Field(..., min_length=1, max_length=8_000)

    # Optional since phase 1: retrieval is corpus-wide behind the tenant gates.
    # When present it will become a metadata filter once documents have real ids
    # (phase 5, persistence). A REQUIRED field the server ignores is a lie.
    # Outer Field bounds the list (1..50 items).
    # Inner DocumentId bounds each element (8..64 chars).
    document_ids: Annotated[list[DocumentId], Field(min_length=1, max_length=50)] | None = None

    # NOTE what is ABSENT: tenant_id and groups. They lived here for exactly one
    # phase, marked unshippable, and were deleted when JWT identity landed —
    # the principal now arrives ONLY via the Authorization header (app/auth.py).
    # With extra="forbid", a client still sending tenant_id gets a 422, not a
    # silent ignore: the API actively rejects the old, unsafe shape.

    # And note what IS in the body: as_of. The contrast with tenant_id is the
    # lesson. tenant_id EXPANDS what you may see, so it must arrive signed.
    # as_of only SELECTS among versions you already own — a time cursor inside
    # your authorization scope, not an escalation. "Which knobs need a
    # signature" is a per-knob decision, not a blanket rule.
    # None = today ("what does my policy say?"). A claims handler sends the
    # DATE OF LOSS: a December accident is governed by December's wording.
    as_of: date | None = None

    # temperature=2.0 does not error at Azure. It just produces garbage, expensively.
    temperature: float = Field(0.0, ge=0.0, le=1.0)

    # Same quota lesson as ChatStreamRequest: Azure reserves prompt + max_tokens
    # against TPM at admission time. Sized for a cited answer over policy
    # extracts, not for an essay.
    max_tokens: int = Field(400, ge=1, le=4096)


class Citation(BaseModel):
    document_id: DocumentId
    page: int = Field(..., ge=1)
    snippet: str = Field(..., max_length=1_000)


class AskResponse(BaseModel):
    """What we PROMISE to return.

    Used as `response_model=AskResponse` on the route. FastAPI then serialises
    ONLY these fields, dropping anything else the service layer handed back.

    That is a confidentiality control, not a docs feature: in a multi-tenant
    system the service object may carry another tenant's retrieved chunks, the
    raw prompt, or internal IDs. Without response_model, one careless `return obj`
    ships them to the client.
    """

    answer: str
    citations: list[Citation]
    # NOTE: this is the MODEL'S self-reported confidence. It is not calibrated
    # and it is not evidence. Module 3 replaces trust in this number with
    # grounding + citation checking. Named here so we remember to distrust it.
    confidence: float = Field(..., ge=0.0, le=1.0)


class ErrorBody(BaseModel):
    code: str            # machine-readable: "rate_limited", "content_filtered"
    message: str         # human-readable, safe to show a user
    request_id: str      # paste this into a support ticket; we grep for it

    # The field that earns its place.
    #   Azure 429            -> retryable=True   (back off, try again)
    #   Azure content filter -> retryable=False  (retrying is guaranteed to fail
    #                                             forever, and burns quota doing it)
    # Both are "the LLM call failed" and both are plausibly HTTP 502. The status
    # code cannot distinguish them. We know the answer — so we put it in the body
    # rather than making every client guess, differently.
    retryable: bool


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# =============================================================================
# GATE 2 — the model contract
#
# This is what we ask Azure to produce, AND what we re-validate its answer against.
# `model_json_schema()` on this class is handed to Azure as the json_schema, so
# the schema can never drift from the class.
# =============================================================================
class InvoiceExtract(BaseModel):
    # strict=True: do NOT coerce. "1240.50" stays a string and raises.
    #
    # By default Pydantic is a parser — it tries to make data fit ("1240.50" -> 1240.5).
    # That is right at an HTTP boundary where everything is a string.
    # It is WRONG in a money path: a value you had to coerce is a value you don't
    # understand. A rejection you can escalate beats a number you can't trace.
    model_config = ConfigDict(strict=True, extra="forbid")

    invoice_total: float = Field(..., ge=0)

    # Type validity is not semantic validity. `currency: str` would happily accept
    # "₹" or "rupees" or "". Literal turns the value space itself into the contract.
    currency: Literal["INR", "USD", "EUR", "GBP"]

    invoice_number: str = Field(..., min_length=1, max_length=64)

    # Where in the source document each figure came from. This is the seed of the
    # grounding story: an extraction you cannot point at is an extraction you
    # cannot defend. Module 3 makes this load-bearing.
    source_page: int = Field(..., ge=1)


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=32_000)


class HandoffRequest(BaseModel):
    """Gate 1 for /v1/handoff (phase 6).

    request_id references the caller's OWN audited exchange — the route
    verifies that ownership against the verified principal, and answers 404
    (never 403) on a miss: a 403 on someone else's id confirms it exists.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=8, max_length=64)
    # The customer's own words, size-capped like every free-text field: this
    # travels into a ticketing system whose limits we don't control.
    note: str = Field("", max_length=2_000)


class HandoffResponse(BaseModel):
    """What the client renders on the refusal card: the ticket reference."""

    ticket_id: str
    request_id: str
    status: Literal["open"]
    created_at: datetime


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, max_length=8_000)
    temperature: float = Field(0.0, ge=0.0, le=1.0)

    # max_tokens is a QUOTA decision, not merely a length cap (section 1.4).
    # Azure reserves prompt_tokens + max_tokens against your TPM budget at
    # admission time — before the model writes a word. An inflated ceiling
    # throws away quota you never use, and you eat 429s at 40% real utilisation.
    max_tokens: int = Field(256, ge=1, le=4096)

