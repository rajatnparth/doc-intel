"""Retrieval: turning a query vector into the nearest chunks.

Section 3.2 lives here as a MEASUREMENT, not a lecture. ann_bench.py builds the
same vectors three ways — exact (ground truth), HNSW, IVF-PQ — and prints the
recall@k / latency / memory you actually gave up. The whole point of the section
is that this number exists and almost nobody looks at it.
"""
