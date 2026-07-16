"""The LLM boundary.

Everything that knows about a model provider lives in this package.
NOTHING in here imports FastAPI. That is the rule, and it is the whole point:

    route handler  ->  LLMClient (a Protocol)  ->  StubLLMClient | AzureLLMClient

When Marta asks "where does your retry logic live, and why does that matter
when we add a second provider next quarter?" — the answer is this package.
Retry, timeout, semaphore and breaker are properties of the PROVIDER
RELATIONSHIP, not of the endpoint. Smear them across route handlers and the
seam disappears.
"""

from app.llm.base import (               # local — app/llm/base.py
                                        #   re-exported here so callers can write
                                        #   `from app.llm import LLMClient` instead of
                                        #   reaching into the base submodule
    EmbeddingClient,
    LLMClient,
    LLMError,
    RerankClient,
    TokenChunk,
    Usage,
)

__all__ = [
    "EmbeddingClient",
    "LLMClient",
    "LLMError",
    "RerankClient",
    "TokenChunk",
    "Usage",
]
