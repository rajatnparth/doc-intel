"""Phase 6 — the audit trail. One record per exchange: what was asked, what
was retrieved and scored, which pipeline decided, and what was DELIVERED.

AUDIT IS NOT LOGGING
--------------------
A log line serves an operator debugging now; it is unstructured, rotated
away, and allowed to be incomplete. An audit record serves a dispute handler
months later — "your assistant told me X in March" — and a regulator after
that. It is structured, append-only, retained, and complete or loudly not.
Different consumer, different artifact, different module.

THE STREAMING COMPLICATION (the reason this file's shape is what it is)
-----------------------------------------------------------------------
The record's most important field is `answer_text` — the words the customer
actually saw. With streaming, that is unknowable until the stream ENDS, so
the record is finalized in the stream's `finally` path, not at request time.
A disconnect after 40 tokens produces a record that says exactly that, which
is precisely what the dispute needs. (`main.py` owns the wiring; this module
owns the shape and the sink.)

Nothing here imports FastAPI. The sink is a seam (same argument as llm/ and
store/): JSONL file here, the insurer's WORM store or append-only table on
the private side of the split.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import json                             # stdlib — one JSON object per line
import threading                        # stdlib — one lock; see JsonlAuditSink
from datetime import date, datetime, timezone  # stdlib — timestamps are DATA here
from pathlib import Path                # stdlib — sink path handling
from typing import Literal, Protocol, runtime_checkable  # stdlib — the seam

from pydantic import BaseModel, ConfigDict, Field  # 3rd-party: pydantic — the record IS
                                        #   a contract: it will be read years later by
                                        #   code that must trust its shape


class RetrievedRef(BaseModel):
    """A chunk as the pipeline saw it — enough to re-open the exact wording.

    chunk_index is None for CITED sources: they are deduped PARENT sections
    (rag.py), which span chunks — a single index would be an invention."""

    doc_title: str
    heading: str
    chunk_index: int | None = None


class FactRef(BaseModel):
    name: str
    value: str


class AuditRecord(BaseModel):
    # extra="forbid" both ways: a record that gained a mystery field is as
    # suspect as one that lost a required one.
    model_config = ConfigDict(extra="forbid")

    request_id: str
    at: datetime                        # UTC, tz-aware; naive timestamps are a lie
    tenant_id: str
    groups: list[str]

    question: str
    as_of: date | None = None

    # Which pipeline decided, and how it ended. "disconnected" is an OUTCOME,
    # not an error: the customer walked away mid-answer, and the record says
    # what they had seen by then.
    outcome: Literal["facts", "answer", "refusal", "error", "disconnected"]

    # -- the decision's inputs, for replaying WHY -----------------------------
    rerank_score: float | None = None   # the number the refusal gate read
    threshold: float | None = None
    retrieved: list[RetrievedRef] = Field(default_factory=list)  # candidates (or near-misses)
    sources: list[RetrievedRef] = Field(default_factory=list)    # what was actually cited

    # -- what was delivered ---------------------------------------------------
    facts: list[FactRef] | None = None  # the system-of-record path
    answer_text: str = ""               # the words the customer saw, verbatim
    refusal_reason: str = ""
    error_code: str = ""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_ms: int = 0


@runtime_checkable
class AuditSink(Protocol):
    def write(self, record: AuditRecord) -> None:
        """Append. Never update, never delete — an audit trail you can edit
        is a liability with extra steps."""
        ...

    def get(self, request_id: str) -> AuditRecord | None:
        """Fetch one exchange. The handoff endpoint is the consumer: a ticket
        REFERENCES the audited exchange rather than copying it."""
        ...


class JsonlAuditSink:
    """One JSON object per line, appended. Honest about its scale: get() is a
    linear scan, which is fine for a demo corpus and wrong past ~10^5 records
    — the enterprise version is an append-only table with request_id indexed
    (or the insurer's WORM store). The Protocol is the part that survives
    that upgrade.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # One process-wide lock so two concurrent stream-endings can't
        # interleave partial lines. (Multi-process is the server's problem —
        # O_APPEND keeps whole lines atomic on POSIX for sane record sizes,
        # but the honest fix at that scale is the table above.)
        self._lock = threading.Lock()

    def write(self, record: AuditRecord) -> None:
        line = record.model_dump_json() + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line)

    def get(self, request_id: str) -> AuditRecord | None:
        if not self._path.exists():
            return None
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                if request_id in line:  # cheap pre-filter before parsing
                    rec = AuditRecord.model_validate_json(line)
                    if rec.request_id == request_id:
                        return rec
        return None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
