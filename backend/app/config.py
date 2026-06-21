"""Runtime configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Absolute path to backend/.env so it's found whether we're launched from backend/
# (uvicorn) or from the repo root (the eval runner) — not the current directory.
_ENV_FILE = str(Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_prefix="", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority is left-to-right. Put the .env file AHEAD of the shell
        # environment so a stale global export (e.g. an old OPENAI_API_KEY in
        # ~/.zshrc) can never silently shadow the project's backend/.env.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    # --- LLM ---
    # LiteLLM-style provider string. Swap providers with a one-line change here.
    # Default: quality-first v1. Fallback / A-B target: "openai/gpt-5.4-mini".
    transclassify_model: str = "openai/gpt-5.5"
    openai_api_key: str | None = None
    llm_max_retries: int = 2
    # Bound concurrent LLM calls so a large CSV doesn't open hundreds of sockets.
    llm_concurrency: int = 8

    # --- SERP (enrichment web lookup) ---
    serper_api_key: str | None = None
    serp_timeout_seconds: float = 8.0
    # Billed per SERP query, added to enrich row cost (Serper ~ $1 / 1k queries).
    serp_cost_per_query: float = 0.001

    # --- Database (minimal logging only; off the hot path) ---
    database_url: str | None = None

    # --- CORS (frontend dev server) ---
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()


def _export_resolved_keys() -> None:
    """Push our resolved (.env-authoritative) secrets into the process env.

    Libraries like LiteLLM read provider keys straight from os.environ and never
    see our Settings object. Without this, a stale global export (e.g. an old
    OPENAI_API_KEY in ~/.zshrc) would still win inside LiteLLM even when
    backend/.env holds the correct key. Exporting here makes .env authoritative
    everywhere in the process, not just for Settings lookups.
    """
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.serper_api_key:
        os.environ["SERPER_API_KEY"] = settings.serper_api_key


_export_resolved_keys()
