"""The demo corpus: two tenants, one superseded document.

Shared by gated_demo.py, calibrate.py and the tests so they all reason about the
same data. Real ingestion would read tenancy from the upload request; this is
the fixture.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from pathlib import Path                # stdlib — locate sample_docs/

from app.ingest import Chunk, ChunkMeta, chunk_document  # local — app/ingest/

_DOCS = Path(__file__).resolve().parents[2] / "sample_docs"

# (file, title, tenant, acl, status)
_MANIFEST = [
    ("acme_msa.md",                "Acme MSA (2024)",      "acme",    {"legal", "finance"}, "active"),
    ("invoices.md",                "Invoice Register 2024", "acme",    {"finance"},          "active"),
    # Same shape of contract, different tenant. This is what makes the leak REAL:
    # "what are our payment terms?" genuinely matches both tenants' documents.
    ("contoso_msa.md",             "Contoso MSA (2024)",   "contoso", {"legal", "finance"}, "active"),
    # Superseded: still in the store, must never be retrievable.
    ("acme_msa_v1_superseded.md",  "Acme MSA (2022)",      "acme",    {"legal", "finance"}, "superseded"),
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
ACME_FINANCE = ("acme", frozenset({"finance"}))
ACME_LEGAL = ("acme", frozenset({"legal"}))
CONTOSO_LEGAL = ("contoso", frozenset({"legal"}))
