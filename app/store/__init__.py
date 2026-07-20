"""The storage boundary.

Everything that knows about a vector database lives in this package, behind
the VectorStore Protocol — the third seam, after the LLM and the embedder.
The gate (tenant · acl · as_of) enters `search()` as an ARGUMENT, so
enforcement travels with the query instead of living in a cache key.
"""

from app.store.base import Gate, VectorStore  # local — app/store/base.py
                                        #   re-exported so callers write
                                        #   `from app.store import VectorStore`

__all__ = ["Gate", "VectorStore"]
