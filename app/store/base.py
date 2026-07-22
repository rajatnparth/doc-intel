"""The storage seam: what a vector store must be able to do.

This file defines an INTERFACE and nothing else — no qdrant, no numpy, no
FastAPI. The third seam in this codebase, and the same argument each time:
chat, embeddings, and now STORAGE are all "a thing we rent", and renting a
different one must be a config change, not a refactor.

THE IDEA THAT MAKES THIS PHASE MATTER
-------------------------------------
`search()` takes the Gate as an argument. The visibility predicate — tenant,
ACL, as-of date — travels WITH the query and is enforced INSIDE the store's
search, not around it. Before this seam existed we simulated that by building
a separate index per principal and caching it, and the cache key silently
became the security boundary (the as_of time-leak had to be pinned by a
test). With the gate as a parameter there is no key to forget: a store that
ignores it fails the same test suite that the honest ones pass.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from dataclasses import dataclass       # stdlib — Gate is plain data
from datetime import date               # stdlib — the as_of anchor
from typing import Protocol, runtime_checkable  # stdlib — structural typing, like LLMClient

from app.ingest import Chunk            # local — app/ingest/chunker.py (what stores hold)


@dataclass(frozen=True)
class Gate:
    """The visibility predicate, as data.

    Frozen, because a gate that can be mutated between construction and use
    is a gate that can be widened between construction and use. One instance
    describes one question's entire visibility scope.
    """

    tenant_id: str
    groups: frozenset[str]
    as_of: date


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or REPLACE, keyed on (doc_title, chunk_index).

        Idempotent by contract: running the same ingest twice must leave the
        store exactly as one run left it. "Append-only + dedupe later" is how
        stores silently double-count a re-ingested document.

        Vectors arrive L2-normalised (the ingest pipeline's job — a retrieval
        decision, not a storage one).
        """
        ...

    def search(self, vector: list[float], gate: Gate, k: int) -> list[tuple[Chunk, float]]:
        """Nearest neighbours AMONG WHAT THE GATE PERMITS, best first.

        The filter is applied during the search, not to its results: a chunk
        outside the gate must never be a candidate, so asking for k results
        returns k gate-passing results (or fewer only if the visible corpus
        is smaller than k) — never "50 retrieved, 3 survived filtering".
        """
        ...

    def visible_chunks(self, gate: Gate) -> list[Chunk]:
        """Everything the gate permits, unranked. Feeds the in-process BM25."""
        ...

    def delete_doc(self, doc_title: str, tenant_id: str) -> int:
        """Remove every chunk of one document. Returns how many went.

        Exists because REPLACE must be delete-then-upsert: deterministic ids
        make v2's chunks overwrite v1's — but only the ids v2 also produces.
        A shorter revision would orphan the old tail (v1 chunks 12..19:
        stale wording, silently retrievable, forever) — so the old document
        goes first. tenant_id is part of the predicate for the same reason
        it is part of every other one: a title is not a permission.
        """
        ...

    def count(self) -> int:
        """Total stored chunks (all tenants). For boot checks and idempotency
        tests — never for serving decisions."""
        ...

    def close(self) -> None:
        """Release the store (file locks, connections). No-op where there is
        nothing to release."""
        ...
