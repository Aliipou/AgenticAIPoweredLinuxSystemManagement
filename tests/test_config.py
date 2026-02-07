"""Brutal tests for settings configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic.config.settings import Settings


class TestSettings:
    def test_default_values(self, monkeypatch):
        monkeypatch.setenv("AGENTIC_OPENAI_API_KEY", "sk-test")
        s = Settings()  # type: ignore[call-arg]
        assert s.openai_model == "gpt-4o"
        assert s.dry_run is False
        assert s.log_level == "INFO"
        assert s.max_risk_level == "HIGH"
        assert s.require_confirmation is True

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AGENTIC_OPENAI_API_KEY", "sk-override")
        monkeypatch.setenv("AGENTIC_OPENAI_MODEL", "gpt-3.5-turbo")
        monkeypatch.setenv("AGENTIC_DRY_RUN", "true")
        monkeypatch.setenv("AGENTIC_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("AGENTIC_MAX_RISK_LEVEL", "LOW")
        monkeypatch.setenv("AGENTIC_REQUIRE_CONFIRMATION", "false")
        s = Settings()  # type: ignore[call-arg]
        assert s.openai_api_key == "sk-override"
        assert s.openai_model == "gpt-3.5-turbo"
        assert s.dry_run is True
        assert s.log_level == "DEBUG"
        assert s.max_risk_level == "LOW"
        assert s.require_confirmation is False

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("AGENTIC_OPENAI_API_KEY", raising=False)
        # Clear any other potential sources
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_db_path_is_path(self, monkeypatch):
        monkeypatch.setenv("AGENTIC_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AGENTIC_DB_PATH", "/tmp/test.db")
        s = Settings()  # type: ignore[call-arg]
        assert isinstance(s.db_path, Path)
        assert s.db_path == Path("/tmp/test.db")

    def test_env_prefix(self, monkeypatch):
        # Ensure non-prefixed vars don't affect settings
        monkeypatch.setenv("OPENAI_MODEL", "wrong-model")
        monkeypatch.setenv("AGENTIC_OPENAI_API_KEY", "sk-test")
        s = Settings()  # type: ignore[call-arg]
        assert s.openai_model == "gpt-4o"  # Default, not "wrong-model"
