"""Brutal tests for Rich display helpers."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from agentic.cli.output import (
    print_action_plan,
    print_error,
    print_history,
    print_info,
    print_intent,
    print_results,
    print_status,
)
from agentic.models.action import ActionCandidate, ActionPlan, ActionResult, ActionType
from agentic.models.intent import Entity, IntentType, ParsedIntent
from agentic.models.policy import PolicyDecision, RiskLevel


def _capture(func, *args, **kwargs) -> str:
    """Capture Rich output by temporarily replacing the module console."""
    import agentic.cli.output as mod
    buf = StringIO()
    original = mod.console
    mod.console = Console(file=buf, force_terminal=True, width=120)
    try:
        func(*args, **kwargs)
    finally:
        mod.console = original
    return buf.getvalue()


class TestPrintIntent:
    def test_displays_intent_type(self):
        intent = ParsedIntent(
            raw_query="focus",
            intent_type=IntentType.FOCUS,
            confidence=0.95,
            reasoning="User wants focus",
        )
        output = _capture(print_intent, intent)
        assert "FOCUS" in output
        assert "95%" in output

    def test_displays_entities(self):
        intent = ParsedIntent(
            raw_query="close firefox",
            intent_type=IntentType.FOCUS,
            confidence=0.9,
            entities=[Entity(name="process", value="firefox", source="firefox")],
            reasoning="Focus",
        )
        output = _capture(print_intent, intent)
        assert "firefox" in output

    def test_no_entities(self):
        intent = ParsedIntent(
            raw_query="update",
            intent_type=IntentType.UPDATE,
            confidence=0.8,
        )
        output = _capture(print_intent, intent)
        assert "UPDATE" in output


class TestPrintActionPlan:
    def test_displays_actions_with_decisions(self):
        action = ActionCandidate(
            id="a1",
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend firefox",
            target="firefox",
        )
        plan = ActionPlan(intent_id="i1", actions=[action])
        decision = PolicyDecision(
            action_id="a1",
            risk_level=RiskLevel.LOW,
            approved=True,
        )
        output = _capture(print_action_plan, plan, [decision])
        assert "Suspend firefox" in output
        assert "APPROVED" in output

    def test_denied_action(self):
        action = ActionCandidate(
            id="a2",
            action_type=ActionType.KILL_BY_MEMORY,
            description="Kill hog",
            target="chrome",
        )
        plan = ActionPlan(intent_id="i1", actions=[action])
        decision = PolicyDecision(
            action_id="a2",
            risk_level=RiskLevel.HIGH,
            approved=False,
        )
        output = _capture(print_action_plan, plan, [decision])
        assert "DENIED" in output

    def test_missing_decision_shows_question_mark(self):
        action = ActionCandidate(
            id="a3",
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend",
            target="proc",
        )
        plan = ActionPlan(intent_id="i1", actions=[action])
        output = _capture(print_action_plan, plan, [])
        assert "?" in output


class TestPrintResults:
    def test_success_result(self):
        results = [ActionResult(action_id="a1", success=True, output="Suspended PID 123")]
        output = _capture(print_results, results)
        assert "OK" in output
        assert "Suspended" in output

    def test_failure_result(self):
        results = [ActionResult(action_id="a1", success=False, error="Permission denied")]
        output = _capture(print_results, results)
        assert "FAIL" in output
        assert "Permission denied" in output


class TestPrintError:
    def test_error_message(self):
        output = _capture(print_error, "Something went wrong")
        assert "Something went wrong" in output


class TestPrintInfo:
    def test_info_message(self):
        output = _capture(print_info, "Some info")
        assert "Some info" in output


class TestPrintHistory:
    def test_displays_rows(self):
        rows = [
            {
                "created_at": "2025-01-01",
                "raw_query": "focus mode",
                "intent_type": "FOCUS",
                "action_type": "SUSPEND_PROCESS",
                "approved": True,
            }
        ]
        output = _capture(print_history, rows)
        assert "focus mode" in output
        assert "FOCUS" in output

    def test_none_approved(self):
        rows = [
            {
                "created_at": "2025-01-01",
                "raw_query": "test",
                "intent_type": "UNKNOWN",
                "action_type": None,
                "approved": None,
            }
        ]
        output = _capture(print_history, rows)
        assert "test" in output

    def test_empty_history(self):
        output = _capture(print_history, [])
        assert "History" in output


class TestPrintStatus:
    def test_displays_metrics(self):
        output = _capture(print_status, 45.2, 67.3, [])
        assert "45.2" in output
        assert "67.3" in output

    def test_displays_processes(self):
        procs = [
            {"pid": 1, "name": "chrome", "memory_percent": 12.5, "cpu_percent": 5.3},
        ]
        output = _capture(print_status, 10.0, 50.0, procs)
        assert "chrome" in output
        assert "12.5" in output
