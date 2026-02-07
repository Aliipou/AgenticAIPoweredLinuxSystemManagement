"""Shared fixtures and mocks for all tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from agentic.config.settings import Settings
from agentic.memory.store import MemoryStore
from agentic.models.action import ActionCandidate, ActionPlan, ActionResult, ActionType
from agentic.models.intent import Entity, IntentType, ParsedIntent
from agentic.models.policy import PolicyDecision, RiskLevel


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("AGENTIC_OPENAI_API_KEY", "sk-test-key-fake")
    monkeypatch.setenv("AGENTIC_OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("AGENTIC_DRY_RUN", "false")
    monkeypatch.setenv("AGENTIC_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AGENTIC_MAX_RISK_LEVEL", "HIGH")
    monkeypatch.setenv("AGENTIC_REQUIRE_CONFIRMATION", "false")
    return Settings()  # type: ignore[call-arg]


@pytest_asyncio.fixture
async def temp_db():
    store = MemoryStore(db_path=":memory:")
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def sample_focus_intent():
    return ParsedIntent(
        id="intent-focus-001",
        raw_query="help me focus",
        intent_type=IntentType.FOCUS,
        confidence=0.95,
        entities=[Entity(name="process", value="firefox", source="firefox")],
        reasoning="User wants to focus by closing distractions",
    )


@pytest.fixture
def sample_update_intent():
    return ParsedIntent(
        id="intent-update-001",
        raw_query="update my system",
        intent_type=IntentType.UPDATE,
        confidence=0.9,
        entities=[],
        reasoning="User wants system packages updated",
    )


@pytest.fixture
def sample_clean_memory_intent():
    return ParsedIntent(
        id="intent-clean-001",
        raw_query="free up some RAM",
        intent_type=IntentType.CLEAN_MEMORY,
        confidence=0.88,
        entities=[],
        reasoning="User wants to free memory",
    )


@pytest.fixture
def sample_unknown_intent():
    return ParsedIntent(
        id="intent-unknown-001",
        raw_query="tell me a joke",
        intent_type=IntentType.UNKNOWN,
        confidence=0.2,
        entities=[],
        reasoning="Not a system management request",
    )


@pytest.fixture
def sample_action_candidate():
    return ActionCandidate(
        id="action-001",
        action_type=ActionType.SUSPEND_PROCESS,
        description="Suspend process: firefox",
        command="kill -STOP $(pgrep -f firefox)",
        target="firefox",
        rollback_command="kill -CONT $(pgrep -f firefox)",
    )


@pytest.fixture
def sample_action_plan(sample_action_candidate):
    return ActionPlan(
        id="plan-001",
        intent_id="intent-focus-001",
        actions=[sample_action_candidate],
        reasoning="Generated 1 action(s) for FOCUS.",
    )


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI chat completion response."""
    def _make(data: dict):
        message = MagicMock()
        message.content = json.dumps(data)
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response
    return _make
