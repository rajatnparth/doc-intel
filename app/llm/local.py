"""Local ONNX models, wrapped to the seam's protocols.

THE ONLY FILE THAT MAY IMPORT fastembed — enforced by tests/test_seam.py, the
same way `openai` is fenced into azure.py. hybrid.py importing fastembed
directly worked fine; it also made "swap the embedding provider by config" a
false claim. Same rule, same reason, second SDK.

fastembed is chosen over sentence-transformers because it doesn't drag in
427MB of PyTorch: ONNX on CPU, ~210MB of models total, downloaded on first use.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from functools import lru_cache          # stdlib — one ONNX session per process

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"       # ~130MB download on first use
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # ~80MB download on first use


# The caches live at MODULE level, not on the client instances: tests and
# demos construct retrievers freely, and constructing a client must never
# cost a model load. lru_cache(maxsize=1) on a zero-arg function is the same
# singleton trick as config.get_settings().
@lru_cache(maxsize=1)
def _embedding_model():
    from fastembed import TextEmbedding  # 3rd-party: fastembed — lazy: stub-mode
                                         #   API tests never touch onnxruntime

    return TextEmbedding(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _rerank_model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder  # 3rd-party: fastembed
                                         #   lazy: only needed when you rerank

    return TextCrossEncoder(RERANK_MODEL)


class LocalEmbeddingClient:
    """bge-small-en-v1.5. Conforms to EmbeddingClient by shape (Protocol)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in _embedding_model().embed(texts)]


class LocalRerankClient:
    """ms-marco-MiniLM-L-6-v2 — a real cross-encoder.

    A bi-encoder (retrieval) embeds query and doc separately, in advance, and
    compares two frozen points. A cross-encoder reads the PAIR jointly,
    attending across both — too slow for a corpus, right for the top ~50.

    ⚠️ Emits LOGITS (+5.60 for a matching pair, -11.33 for an unrelated one,
    measured on our corpus) and they are returned AS-IS: raw is the contract
    (see RerankClient in base.py). The sigmoid lives in app/retrieval/gated.py
    because presentation of the score is the refusal path's decision.
    """

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in _rerank_model().rerank(query, texts)]
