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

from pathlib import Path                # stdlib — locate sample_docs/

from app.ingest import Chunk, ChunkMeta, chunk_document  # local — app/ingest/

_DOCS = Path(__file__).resolve().parents[2] / "sample_docs"

# (file, title, tenant, acl, status)
_MANIFEST = [
    ("asha_policy_kit.md",   "Asha Rao — Motor Policy Kit (2026)",     "asha",   {"customer", "agent"}, "active"),
    ("asha_claims_file.md",  "Claims File — Asha Rao",                 "asha",   {"agent"},             "active"),
    # Same product, different policyholder. This is what makes the leak REAL:
    # "what is my excess?" genuinely matches both customers' policy kits.
    ("vikram_policy_kit.md", "Vikram Mehta — Motor Policy Kit (2026)", "vikram", {"customer", "agent"}, "active"),
    # Superseded: last year's kit, still in the store, must never be retrievable.
    ("asha_policy_kit_v1_superseded.md", "Asha Rao — Motor Policy Kit (2025)", "asha", {"customer", "agent"}, "superseded"),
]


def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for fname, title, tenant, acl, status in _MANIFEST:
        chunks += chunk_document(
            (_DOCS / fname).read_text(),
            doc_title=title,
            max_chars=700,
            meta=ChunkMeta(tenant_id=tenant, acl=frozenset(acl), status=status),
        )
    # chunk_index must be globally unique — RRF keys its fusion dict on it.
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks


# Principals used across demos and tests.
ASHA_CUSTOMER = ("asha", frozenset({"customer"}))
ASHA_AGENT = ("asha", frozenset({"agent"}))
VIKRAM_CUSTOMER = ("vikram", frozenset({"customer"}))
