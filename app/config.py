"""Configuration, loaded from environment once, validated at startup.

WHY A FILE FOR THIS
-------------------
The alternative is `os.environ["AZURE_OPENAI_KEY"]` scattered through the code.
That fails at 3am, on the line that happens to run first, in production.

A Settings model fails at *import time*, on your laptop, naming the missing var.
This is the same principle as section 1.2: validate at the boundary. Config is
just another untrusted input — it comes from a deploy pipeline you don't control.
"""

from functools import lru_cache        # stdlib — @lru_cache, the singleton trick below
from typing import Literal              # stdlib — Literal["stub","azure"] value constraint

from pydantic import Field              # 3rd-party: pydantic — field constraints (gt, ge)
from pydantic_settings import BaseSettings, SettingsConfigDict  # 3rd-party: pydantic-settings
                                        #   (separate pip package from pydantic since v2;
                                        #    reads env vars INTO a validated model)


class Settings(BaseSettings):
    # SettingsConfigDict is ConfigDict's cousin, with extra knobs for env loading.
    model_config = SettingsConfigDict(
        env_file=".env",          # read this file if present
        env_file_encoding="utf-8",
        extra="ignore",           # the OS env has thousands of vars we don't own.
                                  # This is the ONE place "ignore" is correct.
    )

    # `Literal` restricts the value to exactly these two strings.
    # A typo in .env ("stubb") is now a startup crash, not a runtime mystery.
    llm_provider: Literal["stub", "azure"] = "stub"

    # Embeddings cross the same seam as chat (app/llm/base.py) but get their
    # OWN knob: "local + azure-chat" is a legitimate production mix (local
    # embeddings are free and private; chat quality you pay for). "local" is
    # bge-small-en-v1.5 on CPU — no key, no account. Flipping this word is the
    # entire provider swap; if it takes more than that, see factory.py.
    embedding_provider: Literal["local", "azure"] = "local"

    # ---- Vector store (phase 5) ----------------------------------------------
    # "memory" embeds the fixture corpus at boot and dies with the process —
    # the default, so `git clone && uvicorn` needs zero infrastructure.
    # "qdrant" is persistent and requires one prior step:
    #     python -m app.ingest.index
    # The app then boots read-only and FAILS CLOSED if the store is empty —
    # serving an empty index would look exactly like "every question refused".
    vector_store: Literal["memory", "qdrant"] = "memory"

    # ---- Documents (phase 10) ------------------------------------------------
    # Upload size cap, enforced BEFORE parsing: a parser fed unbounded
    # attacker bytes is a denial-of-service invitation. 413 at the door.
    max_upload_bytes: int = Field(5_000_000, ge=1)

    # ---- Ops (phase 9) -------------------------------------------------------
    # "No record, no answer": when the audit sink is failing, refuse NEW
    # exchanges at admission (503 + Retry-After) instead of serving un-audited
    # answers. Default ON — in a compliance product, "we briefly refused" is
    # a better Monday than "we can't prove what we told customers". The
    # opt-out is for deployments where availability outranks the record.
    audit_strict: bool = True

    # ---- Safety (phase 8) ----------------------------------------------------
    # Redact PII (emails, phones, vehicle registrations) from audit records
    # and handoff notes BEFORE they are written. Default ON: storing
    # identifiers is the thing you opt INTO, with a reason.
    audit_redact_pii: bool = True

    # ---- Audit trail (phase 6) -----------------------------------------------
    # Where exchange records are appended (one JSON object per line). The
    # JSONL file is the open-core sink; a WORM store / append-only table
    # implements the same AuditSink Protocol on the private side.
    audit_path: str = "var/audit.jsonl"
    # A local FOLDER by default (embedded mode: no server, single process).
    # Setting qdrant_url flips the SAME client to a server — that one line is
    # the local-laptop -> docker -> managed-cloud path, config not code.
    qdrant_path: str = "var/qdrant"
    qdrant_url: str = ""
    qdrant_collection: str = "chunks"

    # ---- Azure ---------------------------------------------------------------
    # These are Optional-ish (empty string default) because they're only needed
    # when llm_provider == "azure". We enforce that in `validate_for_provider()`
    # rather than in the type, so `stub` mode needs no Azure config at all.
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    # ---- Resilience (section 1.4) --------------------------------------------
    # ge/gt constraints mean a nonsensical env var is a startup error.
    llm_timeout_seconds: float = Field(60.0, gt=0)
    llm_connect_timeout_seconds: float = Field(2.0, gt=0)
    llm_max_concurrency: int = Field(8, ge=1)
    llm_max_retries: int = Field(2, ge=0)

    # How long a request will queue for a semaphore slot before we SHED it with
    # our own 429. Rejecting in 5ms is kinder than timing out in 60s: the client
    # can back off sensibly, and the slot goes to someone we can actually serve.
    llm_acquire_timeout_seconds: float = Field(0.5, gt=0)

    # Circuit breaker. Consecutive failures before we stop calling entirely, and
    # how long we stay open before allowing ONE probe through.
    llm_breaker_threshold: int = Field(5, ge=1)
    llm_breaker_cooldown_seconds: float = Field(30.0, gt=0)

    # /v1/ask context budget, in chars (~4 chars/token). A cost control and an
    # attention control, not just a window limit: past a point, more context
    # makes answers worse AND more expensive at the same time.
    ask_context_chars: int = Field(6_000, ge=500)

    # HS256 signing secret for bearer tokens. DELIBERATELY NO DEFAULT: the only
    # acceptable default secret is no secret. A service that can start in an
    # unsafe state will be run in an unsafe state — so the API refuses to boot
    # without one (validate_for_serving), while the retrieval demos, which
    # never touch auth, stay keyless. Generate one:
    #   python -c "import secrets; print(secrets.token_hex(32))"
    # Production replaces this scheme with the IdP's RS256/JWKS keys (app/auth.py).
    auth_jwt_secret: str | None = None

    def validate_for_provider(self) -> None:
        """Fail fast if we're told to use Azure but weren't given credentials.

        Called from the app's lifespan on startup. Deliberately NOT a Pydantic
        validator: config that is *conditionally* required reads better as an
        explicit check than as a model_validator you have to go hunting for.
        """
        if self.llm_provider == "azure":
            missing = [
                name
                for name in (
                    "azure_openai_endpoint",
                    "azure_openai_api_key",
                    "azure_openai_chat_deployment",
                )
                if not getattr(self, name)
            ]
            if missing:
                raise RuntimeError(
                    f"LLM_PROVIDER=azure but these are unset: {', '.join(missing)}. "
                    f"See AZURE_SETUP.md."
                )

        if self.embedding_provider == "azure":
            missing = [
                name
                for name in (
                    "azure_openai_endpoint",
                    "azure_openai_api_key",
                    "azure_openai_embedding_deployment",
                )
                if not getattr(self, name)
            ]
            if missing:
                raise RuntimeError(
                    f"EMBEDDING_PROVIDER=azure but these are unset: {', '.join(missing)}. "
                    f"See AZURE_SETUP.md."
                )

    def validate_for_serving(self) -> None:
        """Everything validate_for_provider checks, PLUS what only the API
        needs. Split from it on purpose: the retrieval demos call the provider
        factories but never serve HTTP, and demanding an auth secret from
        `python hybrid_demo.py` would be a requirement with no requirer.

        Fail closed, at boot, with the fix in the message — not at 3am with a
        stack trace from jwt.decode(None).
        """
        self.validate_for_provider()

        if not self.auth_jwt_secret:
            raise RuntimeError(
                "AUTH_JWT_SECRET is not set. The API will not serve without one — "
                "there is no default secret, by design. Generate one:\n"
                '  python -c "import secrets; print(f\'AUTH_JWT_SECRET={secrets.token_hex(32)}\')" >> .env'
            )
        if len(self.auth_jwt_secret) < 32:
            raise RuntimeError(
                "AUTH_JWT_SECRET is shorter than 32 chars. HS256 is only as strong "
                "as this string is unguessable — use secrets.token_hex(32)."
            )


@lru_cache
def get_settings() -> Settings:
    """Build Settings once, then hand back the same object forever.

    `@lru_cache` on a zero-arg function is the standard Python singleton.
    Two reasons it matters here:
      1. We read the .env file and validate exactly once, not per request.
      2. Tests can call `get_settings.cache_clear()` to force a reload with
         different env vars. A module-level `settings = Settings()` cannot.
    """
    return Settings()
