"""Measure the recall you sell for latency (and memory).

    python -m app.retrieval.ann_bench

Three ways to answer the same k-NN query over the SAME vectors:

    exact   — checks every vector. Correct by definition. This is GROUND TRUTH.
    HNSW    — a navigable graph. Sweep efSearch: the recall<->latency edge.
    IVF-PQ  — cluster + compress. Sweep nprobe AND show the MEMORY win.

Nothing here is asserted. Every recall@10, every latency, every byte is measured
against exact search. "We use HNSW" is a vibe. This file is the number.

NOTE ON DATA: we use synthetic CLUSTERED vectors, not real Azure embeddings
(no key needed, and it's reproducible). This is honest for an ANN benchmark —
real embeddings also live on clusters, not spread uniformly, and clustering is
exactly what makes ANN's shortcuts work. The SHAPE of the result is the lesson.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import time                             # stdlib — wall-clock timing for latency percentiles

import faiss                            # 3rd-party: faiss-cpu — Facebook AI Similarity Search;
                                        #   gives us IVF-PQ (clustering + product quantisation)
import hnswlib                          # 3rd-party: hnswlib — a small, focused HNSW implementation
import numpy as np                      # 3rd-party: numpy — the array math under everything here


# =============================================================================
# 1. Synthetic data that behaves like embeddings
# =============================================================================
def make_clustered_vectors(
    n: int = 20_000, dim: int = 256, n_clusters: int = 80, seed: int = 7
) -> np.ndarray:
    """n vectors in `dim` dimensions, grouped into clusters, L2-normalised.

    Real embeddings cluster by topic — invoices near invoices, SLAs near SLAs.
    Uniform random vectors would make ANN look artificially perfect, so we
    reproduce the clustering that makes the problem realistic.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n)
    # each point = its cluster centre + a little noise
    pts = centers[assign] + rng.normal(scale=0.35, size=(n, dim)).astype(np.float32)
    # normalise to unit length. After this, cosine == dot product == (2 - L2^2)/2,
    # so every index below is ranking the SAME thing — a fair comparison. This is
    # also the "normalise and cosine==dot" fact from the section, made load-bearing.
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts.astype(np.float32)


# =============================================================================
# 2. Ground truth: exact k-NN by brute force
# =============================================================================
def exact_topk(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """The correct answer, for every query. Slow on purpose.

    On unit vectors, largest dot product == nearest neighbour. `queries @ data.T`
    is every query against every vector — the full scan ANN spends its cleverness
    avoiding. Returns the indices of the true top-k for each query.
    """
    sims = queries @ data.T                       # (n_queries, n_data) — the whole point is this is BIG
    return np.argpartition(-sims, kth=k, axis=1)[:, :k]


def recall_at_k(truth: np.ndarray, got: np.ndarray) -> float:
    """Fraction of the TRUE top-k that a method actually returned, averaged.

    recall@10 = 0.94 means 9.4 of your 10 results were genuinely the best 10;
    0.6 were impostors that slipped in because the real ones were skipped.
    """
    hits = 0
    for t_row, g_row in zip(truth, got):
        hits += len(set(t_row.tolist()) & set(g_row.tolist()))
    return hits / (truth.shape[0] * truth.shape[1])


def p_latency(fn, queries: np.ndarray, pct: float = 95.0) -> tuple[float, np.ndarray]:
    """Time fn() per query, return the pct-th percentile latency (ms) + results.

    p95, not the mean: tail latency is what a user feels and what an SLA measures.
    """
    per_query_ms: list[float] = []
    out = []
    for q in queries:
        t0 = time.perf_counter()
        res = fn(q)
        per_query_ms.append((time.perf_counter() - t0) * 1000.0)
        out.append(res)
    return float(np.percentile(per_query_ms, pct)), np.array(out)


# =============================================================================
# 3. HNSW — sweep efSearch (the recall<->latency edge)
# =============================================================================
def build_hnsw(data: np.ndarray, *, m: int = 16, ef_construction: int = 200):
    dim = data.shape[1]
    index = hnswlib.Index(space="ip", dim=dim)     # "ip" = inner product == cosine on unit vecs
    # m               : edges per node. Higher = better recall, MORE MEMORY. Build-time.
    # ef_construction : how hard it works building the graph. Higher = better graph, slower build.
    index.init_index(max_elements=data.shape[0], ef_construction=ef_construction, M=m)
    index.add_items(data, np.arange(data.shape[0]))
    return index


# =============================================================================
# 4. IVF-PQ — sweep nprobe, and show the MEMORY win
# =============================================================================
def build_ivfflat(data: np.ndarray, *, nlist: int = 256):
    """IVF WITHOUT compression: partition into nlist clusters, store full vectors.

    This isolates the CLUSTERING loss (the nprobe knob) from the COMPRESSION loss
    (PQ). Compare its recall curve against IVF-PQ below and the gap between them
    is exactly what product quantisation cost you.
    """
    dim = data.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(data)
    index.add(data)
    return index


def build_ivfpq(data: np.ndarray, *, nlist: int = 256, m_pq: int = 64, nbits: int = 8):
    """IVF + product quantisation: cluster, THEN compress each vector to m_pq bytes.

    m_pq splits the `dim`-dim vector into m_pq sub-vectors, each replaced by a
    1-byte codebook id. So storage = m_pq bytes/vector, regardless of dim. That is
    the memory win — and the recall gap vs IVFFlat is what it cost.

    m_pq must divide dim. dim=256, m_pq=64 -> each sub-vector is 4 dims -> 64 B/vec,
    a 16x cut from 1024 B. (Earlier m_pq=32 was 8-dim sub-vectors: more compression,
    more recall loss — a knob, like everything here.)
    """
    dim = data.shape[1]
    quantizer = faiss.IndexFlatIP(dim)             # the coarse "which cluster" index
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(data)                              # PQ must LEARN the codebook from data first
    index.add(data)
    return index


def bytes_per_vector_float32(dim: int) -> int:
    return dim * 4                                 # HNSW/exact store full float32 vectors


def bytes_per_vector_pq(m_pq: int = 64) -> int:
    return m_pq                                    # PQ stores m_pq bytes, full stop. That's the win.


# =============================================================================
# 5. Run it
# =============================================================================
def main() -> None:
    N, DIM, K = 50_000, 256, 10
    print(f"Building {N:,} clustered {DIM}-dim vectors (synthetic, reproducible)...")
    data = make_clustered_vectors(n=N, dim=DIM)
    rng = np.random.default_rng(99)
    q_idx = rng.integers(0, N, size=500)
    # queries = real points nudged slightly, so each HAS a true neighbour to find
    queries = data[q_idx] + rng.normal(scale=0.05, size=(500, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    print("Computing EXACT ground truth (the slow scan ANN avoids)...")
    t0 = time.perf_counter()
    truth = exact_topk(data, queries, K)
    exact_ms = (time.perf_counter() - t0) / len(queries) * 1000
    print(f"  exact: {exact_ms:.2f} ms/query average (full scan), recall = 1.000 by definition\n")

    # ---- HNSW: the recall<->latency edge -------------------------------------
    print("=" * 74)
    print("HNSW — one knob, efSearch, tuned PER QUERY with no reindex")
    print("=" * 74)
    hnsw = build_hnsw(data)
    print(f"{'efSearch':>10} | {'recall@10':>10} | {'p95 latency':>12} | note")
    print("-" * 74)
    for ef in (10, 25, 50, 100, 200, 400):
        hnsw.set_ef(ef)                            # THE dial. Query-time. No rebuild.
        lat, got = p_latency(lambda q: hnsw.knn_query(q, k=K)[0][0], queries)
        rec = recall_at_k(truth, got)
        note = "recall lost here is INVISIBLE on a latency dashboard" if ef <= 25 else ""
        print(f"{ef:>10} | {rec:>10.3f} | {lat:>9.2f} ms | {note}")
    mem_hnsw = bytes_per_vector_float32(DIM)
    print(f"\n  memory: ~{mem_hnsw} bytes/vector (full float32) + graph edges")

    # ---- IVFFlat: isolate the CLUSTERING loss (nprobe only, no compression) --
    print("\n" + "=" * 74)
    print("IVF-Flat — cluster only, full vectors. nprobe = the CLUSTERING recall knob")
    print("=" * 74)
    ivfflat = build_ivfflat(data)
    print(f"{'nprobe':>10} | {'recall@10':>10} | {'p95 latency':>12} | note")
    print("-" * 74)

    def ivfflat_search(q, nprobe):
        ivfflat.nprobe = nprobe
        return ivfflat.search(q.reshape(1, -1), K)[1][0]

    for nprobe in (1, 4, 8, 16, 32, 64):
        lat, got = p_latency(lambda q, np_=nprobe: ivfflat_search(q, np_), queries)
        rec = recall_at_k(truth, got)
        note = "true neighbour in an unprobed cluster" if nprobe <= 4 else "recall recovers as you probe more"
        print(f"{nprobe:>10} | {rec:>10.3f} | {lat:>9.2f} ms | {note}")
    print(f"\n  memory: ~{mem_hnsw} bytes/vector (full float32) — no compression yet")

    # ---- IVF-PQ: add COMPRESSION on top; the gap vs IVFFlat is the PQ cost ----
    print("\n" + "=" * 74)
    print("IVF-PQ — SAME clustering + compress to 64 B/vec. Gap vs IVF-Flat = the PQ tax")
    print("=" * 74)
    ivfpq = build_ivfpq(data, m_pq=64)
    print(f"{'nprobe':>10} | {'recall@10':>10} | {'p95 latency':>12} | note")
    print("-" * 74)

    def ivfpq_search(q, nprobe):
        ivfpq.nprobe = nprobe
        return ivfpq.search(q.reshape(1, -1), K)[1][0]

    for nprobe in (1, 4, 8, 16, 32, 64):
        lat, got = p_latency(lambda q, np_=nprobe: ivfpq_search(q, np_), queries)
        rec = recall_at_k(truth, got)
        note = "compression caps recall BELOW IVF-Flat" if nprobe >= 16 else ""
        print(f"{nprobe:>10} | {rec:>10.3f} | {lat:>9.2f} ms | {note}")
    mem_pq = bytes_per_vector_pq(64)
    print(f"\n  memory: ~{mem_pq} bytes/vector (PQ code) — {mem_hnsw // mem_pq}x smaller than full vectors")

    # ---- the triangle, in one paragraph --------------------------------------
    print("\n" + "=" * 74)
    print("THE TRIANGLE (pick your losses) — TWO stacked recall costs in IVF-PQ")
    print("=" * 74)
    print(f"  exact   : recall 1.000, {exact_ms:.2f} ms/q, {mem_hnsw} B/vec  — correct, slow, heavy")
    print(f"  HNSW    : recall→1.0 via efSearch, {mem_hnsw} B/vec  — fast+accurate, RAM-hungry")
    print(f"  IVFFlat : recall→high via nprobe (clustering loss only), {mem_hnsw} B/vec")
    print(f"  IVF-PQ  : + compression loss on top, {mem_pq} B/vec — {mem_hnsw // mem_pq}x less RAM")
    print("\n  IVF-PQ has TWO recall costs: which cluster you probe, AND that the vector")
    print("  itself is approximated. nprobe fixes the first, never the second.")
    print("  At ~1M vectors: HNSW. At ~100M on a budget: IVF-PQ. That's where it flips.")


if __name__ == "__main__":
    main()
