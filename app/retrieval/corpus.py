"""The demo corpus: two policyholders, one superseded policy kit.

Northwind Motor Insurance is invented; so are both customers. Shared by
gated_demo.py, calibrate.py and the tests so they all reason about the same
data. Real ingestion would read tenancy from the upload request; this is the
fixture.

The ACL split is the insurance-shaped one: `customer` is the policyholder's
self-service view, `agent` is the call-centre view. The claims file carries
internal handler notes, so it is agent-only — a customer asking about their
own claim must get the answer from a human, never from the raw file.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from datetime import date              # stdlib — the effective windows below
from pathlib import Path                # stdlib — locate sample_docs/

from app.ingest import Chunk, ChunkMeta, chunk_document  # local — app/ingest/

_DOCS = Path(__file__).resolve().parents[2] / "sample_docs"

# Both policies renew on 10 January. The 2025 kit's window CLOSED at the 2026
# renewal — it is not "superseded" by decree, it is out of force by date. It
# remains the governing wording for any loss dated inside its window, which is
# exactly why it must stay in the store AND stay reachable via as_of.
_RENEWAL_2026 = date(2026, 1, 10)

# (file, title, tenant, acl, effective_from, effective_to)
_MANIFEST = [
    ("asha_policy_kit.md",   "Asha Rao — Motor Policy Kit (2026)",     "asha",   {"customer", "agent"}, _RENEWAL_2026, None),
    # The claims file is a LIVING RECORD, not a versioned wording — it has no
    # supersession story, so its window is effectively unbounded.
    ("asha_claims_file.md",  "Claims File — Asha Rao",                 "asha",   {"agent"},             date(2025, 1, 1), None),
    # Same product, different policyholder. This is what makes the leak REAL:
    # "what is my excess?" genuinely matches both customers' policy kits.
    ("vikram_policy_kit.md", "Vikram Mehta — Motor Policy Kit (2026)", "vikram", {"customer", "agent"}, _RENEWAL_2026, None),
    # Last year's kit: out of force TODAY, in force for December's losses.
    ("asha_policy_kit_v1_superseded.md", "Asha Rao — Motor Policy Kit (2025)", "asha", {"customer", "agent"}, date(2025, 1, 10), _RENEWAL_2026),
]


def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for fname, title, tenant, acl, eff_from, eff_to in _MANIFEST:
        chunks += chunk_document(
            (_DOCS / fname).read_text(),
            doc_title=title,
            max_chars=700,
            meta=ChunkMeta(
                tenant_id=tenant,
                acl=frozenset(acl),
                effective_from=eff_from,
                effective_to=eff_to,
            ),
        )
    # chunk_index must be globally unique — RRF keys its fusion dict on it.
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks


# Principals used across demos and tests.
ASHA_CUSTOMER = ("asha", frozenset({"customer"}))
ASHA_AGENT = ("asha", frozenset({"agent"}))
VIKRAM_CUSTOMER = ("vikram", frozenset({"customer"}))
