"""
app/config.py
─────────────
Centralised application settings loaded from environment variables.
Uses pydantic-settings so every variable is type-validated on startup.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration.

    All values can be overridden by setting the corresponding environment
    variable or by placing them in a `.env` file in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM Provider Settings ──────────────────────────────────────────────
    openai_api_key: Optional[str] = None
    """Primary API key for OpenAI (recommended)."""

    openai_model: str = "gpt-4o-mini"
    """OpenAI model used for Chat Completions."""

    anthropic_api_key: Optional[str] = None
    """Temporary backward-compatible fallback key during migration."""

    anthropic_model: str = "claude-sonnet-4-20250514"
    """Legacy fallback model name used only for compatibility paths."""

    max_tokens: int = 1500
    """Maximum tokens per LLM call."""

    # ── Application ────────────────────────────────────────────────────────
    app_name: str = "AI Workflow Automation"
    app_version: str = "1.0.0"
    app_env: str = "development"
    """Runtime environment: 'development' or 'production'."""

    log_level: str = "INFO"
    """Logging level: DEBUG | INFO | WARNING | ERROR."""

    cors_origins: str = "*"
    """
    Comma-separated list of allowed CORS origins.
    Use '*' to allow all (development only).
    Example: 'https://myapp.com,https://admin.myapp.com'
    """

    @property
    def cors_origins_list(self) -> list[str]:
        """Returns CORS origins as a Python list."""
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"

    @property
    def llm_api_key(self) -> str:
        """
        Resolve the API key for the active provider.
        Priority: OPENAI_API_KEY -> ANTHROPIC_API_KEY (temporary fallback).
        """
        if self.openai_api_key:
            return self.openai_api_key
        if self.anthropic_api_key:
            return self.anthropic_api_key
        raise ValueError(
            "Missing API key. Set OPENAI_API_KEY (preferred) or ANTHROPIC_API_KEY (temporary fallback)."
        )

    @property
    def active_model(self) -> str:
        """Resolve the configured model for status and metadata reporting."""
        if self.openai_api_key:
            return self.openai_model
        if self.anthropic_api_key:
            return self.anthropic_model
        return self.openai_model


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    Using @lru_cache means the .env file is read exactly once per process,
    which is efficient and consistent. Call ``get_settings.cache_clear()``
    in tests to reload fresh settings between test cases.
    """
    return Settings()
