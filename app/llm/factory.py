"""One function. One `if`. This is the entire provider swap.

If adding a second model provider next quarter means touching more than this
file, the boundary leaked. That is the thing Marta is testing for in round 5.
"""

from app.config import Settings         # local — app/config.py
from app.llm.base import LLMClient       # local — app/llm/base.py (the return TYPE; the
                                        #   concrete classes are imported lazily inside the
                                        #   function so stub mode never needs `openai`)


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "azure":
        # Imported lazily so that `stub` mode never needs the openai package to
        # be importable or credentials to exist. A small thing that makes the
        # test suite fast and hermetic.
        from app.llm.azure import AzureLLMClient

        return AzureLLMClient(settings)

    from app.llm.stub import StubLLMClient

    return StubLLMClient()
