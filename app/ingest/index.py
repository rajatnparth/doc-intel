"""Ingestion: chunks → vectors → the store. Run once, not at every boot.

    python -m app.ingest.index

Phase 5 splits ingestion off from serving. Before this, the API re-embedded
the whole corpus at every boot — invisible at 4 documents, absurd at 40,000,
and it welded "the app is up" to "the corpus is small". Now embedding happens
HERE, once, and boot becomes a fast read-only open of the store.

Idempotent by construction: point ids are deterministic (see store/qdrant.py),
so running this twice leaves the store exactly as one run left it. The CLI
prints the count before and after — same number twice is the proof.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import numpy as np                      # 3rd-party: numpy — L2-normalise before storing

from app.config import get_settings     # local — app/config.py
from app.ingest import Chunk            # local — app/ingest/ (the unit of storage)
from app.llm.base import EmbeddingClient        # local — app/llm/base.py (the seam)
from app.llm.factory import build_embedding_client  # local — app/llm/factory.py
from app.store.base import VectorStore  # local — app/store/base.py (the seam)


def ingest_into(
    store: VectorStore,
    chunks: list[Chunk],
    embedder: EmbeddingClient | None = None,
) -> int:
    """Embed and upsert. Returns how many chunks were written.

    Vectors are L2-normalised HERE, before storage — the same rule as
    retrieval-time queries (hybrid.py), because a store holding unit vectors
    and a query that isn't (or vice versa) fails silently: every similarity
    is wrong by a factor nobody ever sees.
    """
    if not chunks:
        return 0
    embedder = embedder or build_embedding_client(get_settings())
    vecs = np.asarray(embedder.embed([c.text_to_embed for c in chunks]), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.upsert(chunks, vecs.tolist())
    return len(chunks)


def main() -> None:
    # Imported here, not at module top: ingest_into() is also used by the
    # app's lifespan for memory mode, and THAT import path must not drag the
    # fixture corpus or the store factory along with it.
    from app.retrieval.corpus import load_corpus    # local — app/retrieval/corpus.py
    from app.store.factory import build_vector_store  # local — app/store/factory.py

    settings = get_settings()
    store = build_vector_store(settings)
    try:
        before = store.count()
        n = ingest_into(store, load_corpus())
        after = store.count()
        target = settings.qdrant_url or settings.qdrant_path
        print(f"store   : {settings.vector_store} ({target if settings.vector_store == 'qdrant' else 'process memory'})")
        print(f"ingested: {n} chunks")
        print(f"count   : {before} -> {after}")
        if settings.vector_store == "memory":
            print("note    : VECTOR_STORE=memory — this store dies with this process.")
            print("          Set VECTOR_STORE=qdrant in .env to persist.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
