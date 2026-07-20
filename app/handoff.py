"""Phase 6 — handoff: a refusal must not be a dead end.

The gate refuses honestly (gated.py), and until now the customer was left
holding the refusal. In the product, "I can't answer that" becomes a TICKET
into the insurer's workflow, and the design decision that makes the ticket
USEFUL is that it does not copy the conversation — it references the audit
record by request_id. The human who picks it up sees what the customer saw
plus what the system retrieved and scored. Audit is not just compliance;
audit is what makes handoff work.

The tenancy check on that reference lives in the ROUTE (main.py), not here —
enforcement in deterministic code at the boundary, same as every other gate.
This module is the ticket shape and the store seam: the stub below in the
open engine, the insurer's ticketing connector (ServiceNow, Zendesk, the
in-house thing) on the private side implementing the same Protocol.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import itertools                        # stdlib — monotonically increasing ticket numbers
import threading                        # stdlib — the counter is shared mutable state
from datetime import datetime           # stdlib — created_at
from typing import Literal, Protocol, runtime_checkable  # stdlib — the seam

from pydantic import BaseModel, ConfigDict  # 3rd-party: pydantic — tickets cross a boundary

from app.audit import now_utc           # local — app/audit.py (one clock for phase 6)


class HandoffTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    request_id: str                     # the reference INTO the audit trail
    tenant_id: str                      # denormalised so a ticket queue can be
                                        #   filtered without N audit lookups
    question: str                       # the one convenience copy: queue views
                                        #   need a subject line
    reason: str                         # why the exchange ended up human-bound
    note: str = ""                      # the customer's own words, if offered
    status: Literal["open"] = "open"    # the stub only opens; lifecycle is the
                                        #   ticketing system's job, not ours
    created_at: datetime


@runtime_checkable
class TicketStore(Protocol):
    def create(
        self, *, request_id: str, tenant_id: str, question: str, reason: str, note: str
    ) -> HandoffTicket:
        ...

    def get(self, ticket_id: str) -> HandoffTicket | None:
        ...


class StubTicketStore:
    """In-memory, sequential ids (HD-0001…). Enough to make the loop real in
    the demo and the tests; dies with the process, and says so."""

    def __init__(self) -> None:
        self._tickets: dict[str, HandoffTicket] = {}
        self._seq = itertools.count(1)
        self._lock = threading.Lock()

    def create(
        self, *, request_id: str, tenant_id: str, question: str, reason: str, note: str
    ) -> HandoffTicket:
        with self._lock:
            ticket_id = f"HD-{next(self._seq):04d}"
            t = HandoffTicket(
                ticket_id=ticket_id,
                request_id=request_id,
                tenant_id=tenant_id,
                question=question,
                reason=reason,
                note=note,
                created_at=now_utc(),
            )
            self._tickets[ticket_id] = t
            return t

    def get(self, ticket_id: str) -> HandoffTicket | None:
        return self._tickets.get(ticket_id)
