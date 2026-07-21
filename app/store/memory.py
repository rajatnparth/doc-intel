"""The in-memory store — today's behaviour, behind the seam.

The stub analog for storage: zero infrastructure, gone on restart, and the
default (`VECTOR_STORE=memory`) so `git clone && pytest` stays true. It is
NOT a toy in one respect that matters: the gate is applied BEFORE the
similarity math, by masking rows out of the candidate matrix — so it upholds
the same never-a-candidate property the Qdrant filter does, and passes the
same parametrised tests.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import numpy as np                      # 3rd-party: numpy — the vector matrix

from app.ingest import Chunk            # local — app/ingest/chunker.py
from app.store.base import Gate         # local — app/store/base.py (the seam)


class MemoryStore:
    def __init__(self) -> None:
        # Keyed storage, not parallel lists: upsert must REPLACE on the same
        # key, and a dict makes that the only possible behaviour.
        self._rows: dict[tuple[str, int], tuple[Chunk, np.ndarray]] = {}

    # -- writes ---------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for c, v in zip(chunks, vectors):
            self._rows[(c.doc_title, c.chunk_index)] = (
                c,
                np.asarray(v, dtype=np.float32),
            )

    # -- reads ----------------------------------------------------------------
    def _visible(self, gate: Gate) -> list[tuple[Chunk, np.ndarray]]:
        return [
            (c, v)
            for c, v in self._rows.values()
            if c.meta and c.meta.visible_to(gate.tenant_id, gate.groups, gate.as_of)
        ]

    def search(self, vector: list[float], gate: Gate, k: int) -> list[tuple[Chunk, float]]:
        # Mask FIRST, then score. Scoring everything and dropping the foreign
        # rows afterwards would be post-filtering with extra steps — the exact
        # shape gated.py's villain exists to discredit.
        visible = self._visible(gate)
        if not visible:
            return []
        mat = np.stack([v for _, v in visible])
        sims = mat @ np.asarray(vector, dtype=np.float32)   # unit vectors: dot == cosine
        order = np.argsort(-sims)[:k]
        return [(visible[i][0], float(sims[i])) for i in order]

    def visible_chunks(self, gate: Gate) -> list[Chunk]:
        return [c for c, _ in self._visible(gate)]

    def delete_doc(self, doc_title: str, tenant_id: str) -> int:
        doomed = [
            key
            for key, (c, _) in self._rows.items()
            if c.doc_title == doc_title and c.meta and c.meta.tenant_id == tenant_id
        ]
        for key in doomed:
            del self._rows[key]
        return len(doomed)

    def count(self) -> int:
        return len(self._rows)

    def close(self) -> None:
        """Nothing to release — memory dies with the process."""
