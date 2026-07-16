"""The real Azure OpenAI client.

Structurally identical to StubLLMClient from the caller's point of view. That is
the seam working: `main.py` never learns which one it got.

SECTION STATUS
--------------
The happy path is complete and runnable today (given credentials).
The resilience wrapper — timeout / semaphore / jittered retry / breaker — is
scaffolded with TODOs and gets filled in during section 1.4. Read the TODOs;
they are the spec.
"""

from typing import AsyncIterator        # stdlib — stream_chat's return type

from openai import (                     # 3rd-party: openai — the official SDK. Note every
                                        #   name here is an EXCEPTION type except the client,
                                        #   because _translate() maps them onto our own taxonomy.
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAzureOpenAI,      # ← Async. Not AzureOpenAI. Section 1.1, the whole point.
    BadRequestError,
    RateLimitError,
)

from app.config import Settings          # local — app/config.py
from app.llm.base import (               # local — app/llm/base.py: OUR error taxonomy + wire
                                        #   types. Every openai.* exception above dies in
                                        #   _translate() and is reborn as one of these.
    BadRequest,
    ContentFiltered,
    LLMClient,
    ProviderUnavailable,
    RateLimited,
    TokenChunk,
    Usage,
)


class AzureLLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            # api_version is a DATE that pins the REST contract shape.
            # json_schema structured outputs need 2024-08-01-preview or later.
            api_version=settings.azure_openai_api_version,
            # Two timeouts, not one:
            #   connect  — a second or two. A TCP handshake that slow is a dead host.
            #   total    — sized to your longest LEGITIMATE generation.
            # No timeout means a hung socket holds a concurrency slot forever.
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # we own retry policy; the SDK's is invisible to our metrics
        )

        # TODO(1.4): asyncio.Semaphore(settings.llm_max_concurrency)
        #   Cap in-flight calls from this process. Shed excess with our OWN 429 +
        #   Retry-After. Rejecting a request in 5ms is kinder than timing it out
        #   in 60 seconds — and it stops a thousand coroutines queueing to die.
        #
        # TODO(1.4): retry with FULL JITTER, honouring Retry-After.
        #   Retry: RateLimited, ProviderUnavailable.
        #   Never: BadRequest, ContentFiltered. They are deterministic refusals;
        #          retrying burns quota to be told no again.
        #   Jitter is not optional: fifty pods retrying after exactly 2s is a
        #          self-inflicted DDoS that re-synchronises on every cycle.
        #
        # TODO(1.4): circuit breaker. After N consecutive failures, stop calling
        #   for a window. Stops one degraded region eating the whole thread budget.

    # -------------------------------------------------------------------------
    def _translate(self, exc: Exception) -> Exception:
        """Normalise provider exceptions into our taxonomy.

        This method IS the seam. Every `openai.*` exception dies here. If an
        `openai.RateLimitError` ever reaches a route handler, swapping to Bedrock
        next quarter becomes a refactor instead of a config change.
        """
        if isinstance(exc, RateLimitError):
            retry_after = None
            if exc.response is not None:
                raw = exc.response.headers.get("retry-after")
                # Azure knows when capacity frees up. Our backoff formula guesses.
                retry_after = float(raw) if raw else None
            return RateLimited(str(exc), retry_after=retry_after)

        if isinstance(exc, BadRequestError):
            # Azure signals the content filter via a 400 with a specific code.
            body = getattr(exc, "body", None) or {}
            if isinstance(body, dict) and body.get("code") == "content_filter":
                return ContentFiltered(str(exc))
            return BadRequest(str(exc))

        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return ProviderUnavailable(str(exc))

        if isinstance(exc, APIStatusError) and exc.status_code >= 500:
            return ProviderUnavailable(str(exc))

        return exc

    # -------------------------------------------------------------------------
    async def stream_chat(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[TokenChunk]:
        try:
            stream = await self._client.chat.completions.create(
                # This is the DEPLOYMENT NAME you chose in Azure, not the model
                # name. A deployment called "prod-chat" may point at gpt-4o.
                # The `model=` kwarg is a lie inherited from the OpenAI SDK.
                model=self._settings.azure_openai_chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                # max_tokens is a QUOTA decision, not just a length cap.
                # Azure reserves prompt_tokens + max_tokens against your TPM at
                # ADMISSION time. max_tokens=4096 on a route whose answers average
                # 180 tokens throws away ~95% of your quota on every call — you
                # eat 429s at 40% real utilisation. Set it honestly per route.
                max_tokens=max_tokens,
                stream=True,
                # Without this, streaming responses carry NO usage block, and you
                # cannot answer "what does one request cost?" — asked every time.
                stream_options={"include_usage": True},
            )

            async for event in stream:
                # The final usage event has an empty `choices` list.
                if event.usage is not None:
                    yield TokenChunk(
                        text="",
                        usage=Usage(
                            prompt_tokens=event.usage.prompt_tokens,
                            completion_tokens=event.usage.completion_tokens,
                        ),
                    )
                    continue

                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta and delta.content:
                    yield TokenChunk(text=delta.content)

        except Exception as exc:  # noqa: BLE001 — deliberate: translate, then re-raise
            raise self._translate(exc) from exc

    # -------------------------------------------------------------------------
    async def extract(self, text: str, schema: dict, *, max_tokens: int = 512) -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=self._settings.azure_openai_chat_deployment,
                messages=[
                    {"role": "system", "content": "Extract invoice fields. Cite the page."},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "invoice_extract",
                        # `schema` is InvoiceExtract.model_json_schema() — the
                        # Pydantic class emits its own JSON Schema, so the contract
                        # we send Azure can never drift from the class we validate
                        # against. Define the shape once, in Python.
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc

        # Return the RAW STRING. Not a dict. Not a parsed model.
        #
        # Azure's `strict: true` guarantees this string is schema-valid JSON.
        # It guarantees NOTHING about whether invoice_total appears anywhere in
        # the document. Validation is a separate, deterministic gate that runs
        # in code the model cannot influence — see main.py, not here.
        return resp.choices[0].message.content or ""

    async def aclose(self) -> None:
        await self._client.close()


# Unlike stub.py, there is no runtime `assert isinstance(..., LLMClient)` here:
# constructing this class requires credentials. The shape is checked statically
# by mypy, which is the correct tool for a structural-typing claim anyway.
# (Section 1.2: static checking protects you from yourself; Pydantic protects
# you from the world. This is a "from yourself" problem.)
_CONFORMS_TO: type = LLMClient  # noqa: F401 — imported for the type-check contract
