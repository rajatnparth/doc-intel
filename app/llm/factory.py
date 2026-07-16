"""Three factory functions, one `if` each. This is the entire provider swap.

If adding a second model provider next quarter means touching more than this
file, the boundary leaked. That is the thing Marta is testing for in round 5 —
and embeddings and reranking are model providers too, which is exactly where
the first draft failed: chat crossed the seam, embeddings snuck around it.
"""

from app.config import Settings         # local — app/config.py
from app.llm.base import (               # local — app/llm/base.py (the return TYPES; the
                                        #   concrete classes are imported lazily inside the
                                        #   functions so stub mode never needs `openai` and
                                        #   API-only tests never need onnxruntime)
    EmbeddingClient,
    LLMClient,
    RerankClient,
)


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "azure":
        # Imported lazily so that `stub` mode never needs the openai package to
        # be importable or credentials to exist. A small thing that makes the
        # test suite fast and hermetic.
        from app.llm.azure import AzureLLMClient

        return AzureLLMClient(settings)

    from app.llm.stub import StubLLMClient

    return StubLLMClient()


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "azure":
        from app.llm.azure import AzureEmbeddingClient

        return AzureEmbeddingClient(settings)

    from app.llm.local import LocalEmbeddingClient

    return LocalEmbeddingClient()


def build_reranker(settings: Settings) -> RerankClient:
    # One branch today, and honestly so: Azure OpenAI exposes no cross-encoder
    # endpoint. When a hosted reranker enters the stack (Cohere Rerank via AI
    # Foundry, Azure AI Search's semantic ranker), it becomes a branch HERE —
    # and nothing in app/retrieval moves. That claim is the point of the file.
    from app.llm.local import LocalRerankClient

    return LocalRerankClient()
