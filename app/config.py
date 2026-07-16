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
