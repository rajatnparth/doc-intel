"""Section 3.2 — the recall/latency/memory claims, as assertions.

Small N so it runs in the normal test suite. The full picture is in
`python -m app.retrieval.ann_bench`; this just guards the shape of the result.
"""

import numpy as np                      # 3rd-party: numpy — arrays for the vectors
import pytest                           # 3rd-party: pytest — test runner

from app.retrieval.ann_bench import (   # local — app/retrieval/ann_bench.py
    build_hnsw,
    build_ivfflat,
    build_ivfpq,
    exact_topk,
    make_clustered_vectors,
    recall_at_k,
)

K = 10


@pytest.fixture(scope="module")
def bench():
    data = make_clustered_vectors(n=8_000, dim=128, n_clusters=40, seed=1)
    rng = np.random.default_rng(2)
    idx = rng.integers(0, len(data), size=200)
    q = data[idx] + rng.normal(scale=0.05, size=(200, 128)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    truth = exact_topk(data, q, K)
    return data, q, truth


def _hnsw_recall(index, q, truth, ef):
    index.set_ef(ef)
    got = np.array([index.knn_query(x, k=K)[0][0] for x in q])
    return recall_at_k(truth, got)


def test_hnsw_recall_rises_with_efsearch(bench):
    """efSearch IS the recall/latency dial: wider search, higher recall."""
    data, q, truth = bench
    idx = build_hnsw(data)
    low = _hnsw_recall(idx, q, truth, ef=10)
    high = _hnsw_recall(idx, q, truth, ef=200)
    assert high > low, "widening efSearch must raise recall"
    assert high > 0.95, "at ef=200 HNSW should nearly match exact"


def _ivf_recall(index, q, truth, nprobe):
    index.nprobe = nprobe
    got = np.array([index.search(x.reshape(1, -1), K)[1][0] for x in q])
    return recall_at_k(truth, got)


def test_ivfflat_clustering_loss_is_recoverable(bench):
    """nprobe fixes the CLUSTERING loss: probe enough clusters, recall → ~1."""
    data, q, truth = bench
    idx = build_ivfflat(data, nlist=128)
    assert _ivf_recall(idx, q, truth, nprobe=1) < _ivf_recall(idx, q, truth, nprobe=32)
    assert _ivf_recall(idx, q, truth, nprobe=32) > 0.95, "clustering loss recovers"


def test_pq_compression_loss_is_NOT_recoverable_by_nprobe(bench):
    """The section's core point: IVF-PQ has a SECOND loss nprobe can't fix.

    Even at high nprobe, PQ-compressed recall stays measurably below IVF-Flat
    (same clustering, full vectors). That gap is the compression tax.
    """
    data, q, truth = bench
    flat = build_ivfflat(data, nlist=128)
    pq = build_ivfpq(data, nlist=128, m_pq=32)

    flat_hi = _ivf_recall(flat, q, truth, nprobe=32)
    pq_hi = _ivf_recall(pq, q, truth, nprobe=32)

    assert pq_hi < flat_hi, "compression must cost recall the clustering knob can't buy back"
