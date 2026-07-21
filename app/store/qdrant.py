"""The persistent store — Qdrant, behind the seam.

THE ONLY FILE THAT MAY IMPORT qdrant_client (tests/test_seam.py enforces it,
the same fence as openai→azure.py and fastembed→local.py).

Why Qdrant, argued once: the same client speaks to a LOCAL FOLDER (this repo:
no server, no Docker, `git clone && pytest` stays true), a Docker server, or
their managed cloud — so scaling the deployment is config, not code. And its
filters are evaluated DURING the HNSW traversal, which is the pre-filter
property gated.py could only simulate with per-principal index copies.

Local-mode caveat, stated honestly: it is pure Python, single-process (the
folder is locked), and slower than the server. At this corpus size none of
that matters; the moment it does, the fix is `QDRANT_URL=` in .env.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import uuid                             # stdlib — uuid5 for DETERMINISTIC point ids
import warnings                         # stdlib — silence one KNOWN local-mode warning
from datetime import date               # stdlib — payload date encoding

from qdrant_client import QdrantClient, models  # 3rd-party: qdrant-client — the store SDK

from app.ingest import Chunk, ChunkMeta  # local — app/ingest/chunker.py
from app.store.base import Gate          # local — app/store/base.py (the seam)

# uuid5 = name-based, deterministic: the same chunk always maps to the same
# point id, which is what makes upsert IDEMPOTENT — re-ingesting a document
# overwrites its points instead of duplicating them. A random uuid4 here would
# quietly turn every re-ingest into a second copy of the corpus.
_NS = uuid.UUID("a3a37f3e-6c2b-4c8e-9d1f-5a9b1c2d3e4f")

# effective_to=None ("still in force") must survive a round-trip through a
# RANGE FILTER, and ranges don't speak null. We store dates as ordinals and
# encode the open end as date.max — the predicate stays ONE range check
# instead of a range-OR-is-null composite that every backend implements
# slightly differently.
_OPEN_END = date.max.toordinal()


def _point_id(c: Chunk) -> str:
    return str(uuid.uuid5(_NS, f"{c.doc_title}|{c.chunk_index}"))


def _payload(c: Chunk) -> dict:
    assert c.meta is not None, "a chunk without gate metadata must never be stored"
    return {
        # -- the gate fields (what the filter reads) --------------------------
        "tenant_id": c.meta.tenant_id,
        "acl": sorted(c.meta.acl),
        "effective_from_ord": c.meta.effective_from.toordinal(),
        "effective_to_ord": (
            _OPEN_END if c.meta.effective_to is None else c.meta.effective_to.toordinal()
        ),
        # -- the chunk itself -------------------------------------------------
        "doc_title": c.doc_title,
        "heading": c.heading,
        "text": c.text,
        "parent_text": c.parent_text,
        "is_table": c.is_table,
        "chunk_index": c.chunk_index,
    }


def _to_chunk(p: dict) -> Chunk:
    return Chunk(
        doc_title=p["doc_title"],
        heading=p["heading"],
        text=p["text"],
        parent_text=p["parent_text"],
        is_table=p["is_table"],
        chunk_index=p["chunk_index"],
        meta=ChunkMeta(
            tenant_id=p["tenant_id"],
            acl=frozenset(p["acl"]),
            effective_from=date.fromordinal(p["effective_from_ord"]),
            effective_to=(
                None
                if p["effective_to_ord"] == _OPEN_END
                else date.fromordinal(p["effective_to_ord"])
            ),
        ),
    )


def _gate_filter(gate: Gate) -> models.Filter:
    """ChunkMeta.visible_to(), translated clause for clause.

    This translation IS the risk of this phase: two implementations of one
    predicate. The parametrised store tests exist to hold them identical —
    delete a clause here and the same tests that pass for MemoryStore go red.
    """
    ao = gate.as_of.toordinal()
    return models.Filter(
        must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=gate.tenant_id)),
            # MatchAny on a keyword array = "any overlap" — bool(acl & groups).
            models.FieldCondition(key="acl", match=models.MatchAny(any=sorted(gate.groups))),
            # effective_from <= as_of < effective_to (exclusive end).
            models.FieldCondition(key="effective_from_ord", range=models.Range(lte=ao)),
            models.FieldCondition(key="effective_to_ord", range=models.Range(gt=ao)),
        ]
    )


class QdrantStore:
    def __init__(self, *, path: str = "", url: str = "", collection: str = "chunks") -> None:
        # url wins: pointing at a server is the deliberate act.
        self._client = QdrantClient(url=url) if url else QdrantClient(path=path)
        self._collection = collection

    # -- writes ---------------------------------------------------------------
    def _ensure_collection(self, dim: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        # Payload indexes on every field the gate filter reads. Local mode
        # scans fine without them; a SERVER does not — an unindexed filter
        # falls back to full scans and the pre-filter advantage quietly
        # becomes a performance incident. Declaring them here means the same
        # code is honest at both scales.
        # Local mode warns that these are no-ops there — true, expected, and
        # exactly why they're declared anyway (see comment above). Silenced so
        # a KNOWN condition doesn't train anyone to ignore the warning stream.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field, schema in (
                ("tenant_id", models.PayloadSchemaType.KEYWORD),
                ("acl", models.PayloadSchemaType.KEYWORD),
                ("effective_from_ord", models.PayloadSchemaType.INTEGER),
                ("effective_to_ord", models.PayloadSchemaType.INTEGER),
                ("doc_title", models.PayloadSchemaType.KEYWORD),  # delete_doc's predicate
            ):
                self._client.create_payload_index(self._collection, field_name=field, field_schema=schema)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._ensure_collection(dim=len(vectors[0]))
        self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(id=_point_id(c), vector=v, payload=_payload(c))
                for c, v in zip(chunks, vectors)
            ],
            wait=True,   # the CLI's "ingested N chunks" must not be a promise
        )

    # -- reads ----------------------------------------------------------------
    def search(self, vector: list[float], gate: Gate, k: int) -> list[tuple[Chunk, float]]:
        if not self._client.collection_exists(self._collection):
            return []
        res = self._client.query_points(
            self._collection,
            query=vector,
            limit=k,
            query_filter=_gate_filter(gate),   # ← the phase, in one argument
            with_payload=True,
        )
        return [(_to_chunk(p.payload), float(p.score)) for p in res.points]

    def visible_chunks(self, gate: Gate) -> list[Chunk]:
        if not self._client.collection_exists(self._collection):
            return []
        out: list[Chunk] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                scroll_filter=_gate_filter(gate),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend(_to_chunk(p.payload) for p in points)
            if offset is None:
                return out

    def delete_doc(self, doc_title: str, tenant_id: str) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(key="doc_title", match=models.MatchValue(value=doc_title)),
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
            ]
        )
        before = self._client.count(self._collection, count_filter=doc_filter, exact=True).count
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=doc_filter),
            wait=True,   # replace = delete THEN upsert; "eventually gone" would
                         # let the upsert race the delete it depends on
        )
        return before

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(self._collection, exact=True).count

    def close(self) -> None:
        """Local mode holds a lock on the folder; release it so another
        process (the ingest CLI, a second uvicorn) can open the store."""
        self._client.close()
