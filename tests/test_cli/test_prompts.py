"""Brutal tests for CLI prompts."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from rich.console import Console

from agentic.cli.prompts import confirm_execution, display_dry_run
from agentic.models.action import ActionCandidate, ActionType
from agentic.models.policy import PolicyDecision, RiskLevel


def _make_action_and_decision(
    action_id: str = "a1",
    action_type: ActionType = ActionType.SUSPEND_PROCESS,
    risk: RiskLevel = RiskLevel.LOW,
    sudo: bool = False,
):
    action = ActionCandidate(
        id=action_id,
        action_type=action_type,
        description=f"Test action {action_id}",
        command=f"test-cmd-{action_id}",
        target="target",
    )
    decision = PolicyDecision(
        action_id=action_id,
        risk_level=risk,
        approved=True,
        requires_sudo=sudo,
    )
    return action, decision


class TestConfirmExecution:
    def test_returns_true_on_yes(self):
        action, decision = _make_action_and_decision()
        with patch("agentic.cli.prompts.Confirm.ask", return_value=True):
            result = confirm_execution([action], [decision])
        assert result is True

    def test_returns_false_on_no(self):
        action, decision = _make_action_and_decision()
        with patch("agentic.cli.prompts.Confirm.ask", return_value=False):
            result = confirm_execution([action], [decision])
        assert result is False

    def test_shows_sudo_indicator(self):
        action, decision = _make_action_and_decision(sudo=True, risk=RiskLevel.HIGH)
        import agentic.cli.prompts as mod
        buf = StringIO()
        original = mod.console
        mod.console = Console(file=buf, force_terminal=True, width=120)
        try:
            with patch("agentic.cli.prompts.Confirm.ask", return_value=True):
                confirm_execution([action], [decision])
        finally:
            mod.console = original
        output = buf.getvalue()
        assert "sudo" in output

    def test_multiple_actions(self):
        a1, d1 = _make_action_and_decision("a1")
        a2, d2 = _make_action_and_decision("a2", action_type=ActionType.KILL_PROCESS, risk=RiskLevel.MEDIUM)
        with patch("agentic.cli.prompts.Confirm.ask", return_value=True):
            result = confirm_execution([a1, a2], [d1, d2])
        assert result is True

    def test_action_without_command(self):
        action = ActionCandidate(
            id="no-cmd",
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend",
            command="",
            target="proc",
        )
        decision = PolicyDecision(
            action_id="no-cmd",
            risk_level=RiskLevel.LOW,
            approved=True,
        )
        with patch("agentic.cli.prompts.Confirm.ask", return_value=True):
            result = confirm_execution([action], [decision])
        assert result is True


class TestDisplayDryRun:
    def test_displays_actions(self):
        action, decision = _make_action_and_decision()
        import agentic.cli.prompts as mod
        buf = StringIO()
        original = mod.console
        mod.console = Console(file=buf, force_terminal=True, width=120)
        try:
            display_dry_run([action], [decision])
        finally:
            mod.console = original
        output = buf.getvalue()
        assert "DRY RUN" in output
        assert "test-cmd-a1" in output

    def test_empty_actions(self):
        import agentic.cli.prompts as mod
        buf = StringIO()
        original = mod.console
        mod.console = Console(file=buf, force_terminal=True, width=120)
        try:
            display_dry_run([], [])
        finally:
            mod.console = original
        output = buf.getvalue()
        assert "DRY RUN" in output
