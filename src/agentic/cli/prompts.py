"""User confirmation dialogs."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Confirm

from agentic.models.action import ActionCandidate
from agentic.models.policy import PolicyDecision

console = Console()


def confirm_execution(
    actions: list[ActionCandidate],
    decisions: list[PolicyDecision],
) -> bool:
    console.print("\n[bold yellow]The following actions require confirmation:[/]\n")

    decision_map = {d.action_id: d for d in decisions}
    for i, action in enumerate(actions, 1):
        d = decision_map.get(action.id)
        risk = d.risk_level.name if d else "UNKNOWN"
        sudo = " [red](sudo)[/]" if d and d.requires_sudo else ""
        console.print(f"  {i}. [{risk}]{sudo} {action.description}")
        if action.command:
            console.print(f"     Command: [dim]{action.command}[/]")

    console.print()
    return Confirm.ask("Proceed with execution?", default=False)


def display_dry_run(
    actions: list[ActionCandidate],
    decisions: list[PolicyDecision],
) -> None:
    console.print("\n[bold cyan]DRY RUN — no actions will be executed:[/]\n")

    decision_map = {d.action_id: d for d in decisions}
    for i, action in enumerate(actions, 1):
        d = decision_map.get(action.id)
        risk = d.risk_level.name if d else "UNKNOWN"
        console.print(f"  {i}. [{risk}] {action.description}")
        if action.command:
            console.print(f"     Command: [dim]{action.command}[/]")
