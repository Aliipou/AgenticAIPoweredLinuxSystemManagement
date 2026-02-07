"""Brutal tests for Typer CLI app commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest
from typer.testing import CliRunner

from agentic.cli.app import app
from agentic.exceptions import ParseError, PolicyDeniedError
from agentic.models.action import ActionCandidate, ActionPlan, ActionResult, ActionType
from agentic.models.intent import IntentType, ParsedIntent
from agentic.models.policy import PolicyDecision, RiskLevel

runner = CliRunner()


class TestGetPipeline:
    def test_get_pipeline_creates_pipeline(self, mock_settings):
        from agentic.cli.app import _get_pipeline
        pipeline = _get_pipeline(dry_run=True, force=True)
        assert pipeline is not None
        assert pipeline._dry_run is True


class TestAskCommand:
    def test_ask_basic(self):
        mock_pipeline = MagicMock()
        mock_intent = ParsedIntent(
            raw_query="focus", intent_type=IntentType.FOCUS, confidence=0.9,
        )
        mock_plan = ActionPlan(intent_id=mock_intent.id, actions=[])
        mock_pipeline.run = AsyncMock(return_value=(mock_intent, mock_plan, []))
        mock_pipeline._store = AsyncMock()
        mock_pipeline._gate = MagicMock()
        mock_pipeline._gate.evaluate_plan.return_value = []
        mock_pipeline._gate.filter_approved.return_value = ([], [])

        with patch("agentic.cli.app._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["ask", "focus"])
        assert result.exit_code == 0

    def test_ask_with_actions_in_plan(self):
        mock_pipeline = MagicMock()
        action = ActionCandidate(
            id="a1",
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend firefox",
            target="firefox",
        )
        mock_intent = ParsedIntent(
            raw_query="focus", intent_type=IntentType.FOCUS, confidence=0.9,
        )
        mock_plan = ActionPlan(intent_id=mock_intent.id, actions=[action])
        mock_result = ActionResult(action_id="a1", success=True, output="done")
        mock_pipeline.run = AsyncMock(return_value=(mock_intent, mock_plan, [mock_result]))
        mock_pipeline._store = AsyncMock()
        mock_pipeline._gate = MagicMock()
        decision = PolicyDecision(
            action_id="a1", risk_level=RiskLevel.LOW, approved=True,
        )
        mock_pipeline._gate.evaluate_plan.return_value = [decision]
        mock_pipeline._gate.filter_approved.return_value = ([action], [decision])

        with patch("agentic.cli.app._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["ask", "focus"])
        assert result.exit_code == 0

    def test_ask_dry_run_with_actions(self):
        mock_pipeline = MagicMock()
        action = ActionCandidate(
            id="a1",
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend firefox",
            target="firefox",
        )
        mock_intent = ParsedIntent(
            raw_query="focus", intent_type=IntentType.FOCUS, confidence=0.9,
        )
        mock_plan = ActionPlan(intent_id=mock_intent.id, actions=[action])
        mock_pipeline.run = AsyncMock(return_value=(mock_intent, mock_plan, []))
        mock_pipeline._store = AsyncMock()
        mock_pipeline._gate = MagicMock()
        decision = PolicyDecision(
            action_id="a1", risk_level=RiskLevel.LOW, approved=True,
        )
        mock_pipeline._gate.evaluate_plan.return_value = [decision]
        mock_pipeline._gate.filter_approved.return_value = ([action], [decision])

        with patch("agentic.cli.app._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["ask", "focus", "--dry-run"])
        assert result.exit_code == 0

    def test_ask_error(self):
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(side_effect=ParseError("API timeout"))
        mock_pipeline._store = AsyncMock()

        with patch("agentic.cli.app._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["ask", "focus"])
        assert result.exit_code == 1

    def test_ask_with_force(self):
        mock_pipeline = MagicMock()
        mock_intent = ParsedIntent(
            raw_query="focus", intent_type=IntentType.FOCUS, confidence=0.9,
        )
        mock_plan = ActionPlan(intent_id=mock_intent.id, actions=[])
        mock_pipeline.run = AsyncMock(return_value=(mock_intent, mock_plan, []))
        mock_pipeline._store = AsyncMock()
        mock_pipeline._gate = MagicMock()

        with patch("agentic.cli.app._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["ask", "focus", "--force"])
        assert result.exit_code == 0

    def test_ask_with_verbose(self):
        mock_pipeline = MagicMock()
        mock_intent = ParsedIntent(
            raw_query="focus", intent_type=IntentType.FOCUS, confidence=0.9,
        )
        mock_plan = ActionPlan(intent_id=mock_intent.id, actions=[])
        mock_pipeline.run = AsyncMock(return_value=(mock_intent, mock_plan, []))
        mock_pipeline._store = AsyncMock()
        mock_pipeline._gate = MagicMock()

        with patch("agentic.cli.app._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["ask", "focus", "--verbose"])
        assert result.exit_code == 0


class TestHistoryCommand:
    def test_history_empty(self):
        mock_pipeline = MagicMock()
        mock_pipeline._store = AsyncMock()
        mock_pipeline._store.get_history = AsyncMock(return_value=[])
        mock_pipeline._store.initialize = AsyncMock()
        mock_pipeline._store.close = AsyncMock()

        with patch("agentic.cli.app._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["history"])
        assert result.exit_code == 0

    def test_history_with_data(self):
        mock_pipeline = MagicMock()
        mock_pipeline._store = AsyncMock()
        mock_pipeline._store.get_history = AsyncMock(return_value=[
            {
                "created_at": "2025-01-01",
                "raw_query": "focus",
                "intent_type": "FOCUS",
                "action_type": "SUSPEND_PROCESS",
                "approved": True,
            }
        ])
        mock_pipeline._store.initialize = AsyncMock()
        mock_pipeline._store.close = AsyncMock()

        with patch("agentic.cli.app._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["history"])
        assert result.exit_code == 0

    def test_history_with_limit(self):
        mock_pipeline = MagicMock()
        mock_pipeline._store = AsyncMock()
        mock_pipeline._store.get_history = AsyncMock(return_value=[])
        mock_pipeline._store.initialize = AsyncMock()
        mock_pipeline._store.close = AsyncMock()

        with patch("agentic.cli.app._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["history", "--limit", "5"])
        assert result.exit_code == 0


class TestRollbackCommand:
    def test_rollback_stub(self):
        result = runner.invoke(app, ["rollback", "act-123"])
        assert result.exit_code == 0


class TestStatusCommand:
    def test_status(self):
        with (
            patch("agentic.cli.app.psutil.cpu_percent", return_value=25.0),
            patch("agentic.cli.app.psutil.virtual_memory", return_value=MagicMock(percent=60.0)),
            patch("agentic.cli.app.psutil.process_iter", return_value=[]),
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0

    def test_status_with_processes(self):
        mock_proc = MagicMock()
        mock_proc.info = {
            "pid": 1,
            "name": "chrome",
            "memory_percent": 15.0,
            "cpu_percent": 5.0,
        }
        with (
            patch("agentic.cli.app.psutil.cpu_percent", return_value=25.0),
            patch("agentic.cli.app.psutil.virtual_memory", return_value=MagicMock(percent=60.0)),
            patch("agentic.cli.app.psutil.process_iter", return_value=[mock_proc]),
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0

    def test_status_with_process_exception(self):
        mock_proc = MagicMock()
        type(mock_proc).info = property(
            lambda self: (_ for _ in ()).throw(psutil.NoSuchProcess(1))
        )
        with (
            patch("agentic.cli.app.psutil.cpu_percent", return_value=25.0),
            patch("agentic.cli.app.psutil.virtual_memory", return_value=MagicMock(percent=60.0)),
            patch("agentic.cli.app.psutil.process_iter", return_value=[mock_proc]),
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0

    def test_status_with_process_null_values(self):
        mock_proc = MagicMock()
        mock_proc.info = {
            "pid": 1,
            "name": None,
            "memory_percent": None,
            "cpu_percent": None,
        }
        with (
            patch("agentic.cli.app.psutil.cpu_percent", return_value=25.0),
            patch("agentic.cli.app.psutil.virtual_memory", return_value=MagicMock(percent=60.0)),
            patch("agentic.cli.app.psutil.process_iter", return_value=[mock_proc]),
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0


class TestConfigCommand:
    def test_config_display(self, monkeypatch):
        monkeypatch.setenv("AGENTIC_OPENAI_API_KEY", "sk-test")
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0

    def test_config_missing_key(self, monkeypatch):
        monkeypatch.delenv("AGENTIC_OPENAI_API_KEY", raising=False)
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 1
