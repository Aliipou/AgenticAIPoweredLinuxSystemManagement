"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "AGENTIC_"}

    openai_api_key: str = Field(description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    db_path: Path = Field(
        default=Path.home() / ".agentic" / "history.db",
        description="SQLite database path",
    )
    dry_run: bool = Field(default=False, description="Global dry-run mode")
    log_level: str = Field(default="INFO", description="Logging level")
    max_risk_level: str = Field(
        default="HIGH",
        description="Maximum allowed risk level (SAFE/LOW/MEDIUM/HIGH/CRITICAL)",
    )
    require_confirmation: bool = Field(
        default=True, description="Require user confirmation for MEDIUM+ risk"
    )
