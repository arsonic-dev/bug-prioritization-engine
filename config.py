"""
config.py — Centralised environment-based configuration.

All secrets are loaded exclusively from environment variables (or a .env file).
Nothing is hardcoded. Import `settings` from this module wherever config is needed.
"""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ Jira
    jira_url: str = Field(..., description="Base URL of your Jira instance, e.g. https://myorg.atlassian.net")
    jira_email: str = Field(..., description="Email address used to authenticate with Jira")
    jira_api_token: str = Field(..., description="Jira API token (never a password)")
    jira_project_key: str = Field(..., description="Jira project key to pull issues from, e.g. MYPROJ")

    # ------------------------------------------------------------------ Models
    model_path: Path = Field(
        default=Path("models/artifacts"),
        description="Directory where trained models and artefacts are persisted",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name for description encoding",
    )

    # ------------------------------------------------------------------ API
    retrain_api_key: str = Field(..., description="Secret key required to call POST /api/v1/retrain")
    cache_ttl_seconds: int = Field(default=300, description="TTL for the ranked-queue response cache (seconds)")

    # ------------------------------------------------------------------ Scheduling
    retrain_cron_hour: int = Field(default=2, description="Hour (0-23) at which the daily retrain job fires")
    retrain_lookback_days: int = Field(default=90, description="Days of resolved issues to pull for retraining")

    # ------------------------------------------------------------------ Logging
    log_level: str = Field(default="INFO", description="Python logging level: DEBUG | INFO | WARNING | ERROR")

    @field_validator("jira_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("model_path", mode="before")
    @classmethod
    def ensure_path(cls, v) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @field_validator("log_level")
    @classmethod
    def uppercase_log_level(cls, v: str) -> str:
        return v.upper()


# Singleton — import this everywhere
settings = Settings()  # type: ignore[call-arg]
