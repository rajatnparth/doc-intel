"""Phase 4 — the system of record. Numbers live HERE, never in prose.

WHY THIS MODULE EXISTS (the measured version)
---------------------------------------------
calibrate.py caught "what will next year's renewal premium be?" scoring 0.7785
through the RAG pipeline — a false answer in waiting, because the reranker
scores topical proximity and section 2 is ABOUT premiums. The number itself is
not in the corpus, and no retrieval tuning can put it there. For account
questions the failure mode isn't "bad retrieval", it's "wrong subsystem":
an excess, a premium, an IDV is a FACT about this policy, and facts come from
the system of record — exact, typed, current — not from a language model
paraphrasing last year's PDF.

In production this Protocol is implemented against the insurer's policy
administration system (Guidewire, Duck Creek, a homegrown core). The stub
below is the fixture-corpus version — and the CONSISTENCY RULE that makes the
demo honest: where a fact also appears in the policy wording (the excess, the
IDV), the record is the source and the kit quotes it. tests/test_router.py
asserts they agree, so the fixture cannot silently drift.

Nothing here imports FastAPI, openai, or anything retrieval. A system of
record is a domain backend, not an LLM concern.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from dataclasses import dataclass       # stdlib — PolicyRecord is plain data
from datetime import date               # stdlib — renewal dates
from typing import Protocol             # stdlib — the connector seam


@dataclass(frozen=True)
class PolicyRecord:
    """One policy, as the core system knows it.

    Values are display strings for the demo (₹ formatting included) — a real
    connector returns typed money (minor units + currency) and the rendering
    happens at the edge. What matters architecturally is that every field here
    is AUTHORITATIVE: nothing downstream may contradict it, including the LLM.
    """

    policy_number: str
    annual_premium: str
    own_damage_excess: str
    idv: str                            # Insured's Declared Value
    ncb_percent: str                    # no-claim bonus at last renewal
    renewal_date: date


class PolicyAdmin(Protocol):
    """The connector seam, same idea as LLMClient: the engine depends on this
    shape; which core system sits behind it is a deployment detail. (In the
    open-core split, implementations against real cores live in the private
    product — this Protocol is the boundary they plug into.)"""

    def get_record(self, tenant_id: str) -> PolicyRecord | None: ...


class StubPolicyAdmin:
    """The fixture 'core system'. Keyed by tenant_id — the SAME verified
    identity that gates retrieval. Tenant isolation doesn't stop at documents:
    a lookup keyed by anything the client controls would leak numbers exactly
    the way post-filtering leaked chunks."""

    _RECORDS = {
        "asha": PolicyRecord(
            policy_number="MTR-2026-1147",
            annual_premium="₹18,900",
            own_damage_excess="₹2,000",
            idv="₹6,45,000",
            ncb_percent="20%",
            renewal_date=date(2027, 1, 10),
        ),
        "vikram": PolicyRecord(
            policy_number="MTR-2026-2210",
            annual_premium="₹34,200",
            own_damage_excess="₹5,000",
            idv="₹11,20,000",
            ncb_percent="35%",
            renewal_date=date(2027, 1, 10),
        ),
    }

    def get_record(self, tenant_id: str) -> PolicyRecord | None:
        return self._RECORDS.get(tenant_id)
