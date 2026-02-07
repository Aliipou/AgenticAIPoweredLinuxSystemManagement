"""Rich display helpers for CLI output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic.models.action import ActionCandidate, ActionPlan, ActionResult
from agentic.models.intent import ParsedIntent
from agentic.models.policy import PolicyDecision

console = Console()


def print_intent(intent: ParsedIntent) -> None:
    table = Table(title="Parsed Intent", show_header=False, expand=True)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Intent", intent.intent_type.value)
    table.add_row("Confidence", f"{intent.confidence:.0%}")
    table.add_row("Reasoning", intent.reasoning)
    if intent.entities:
        entities_str = ", ".join(f"{e.name}={e.value}" for e in intent.entities)
        table.add_row("Entities", entities_str)
    console.print(table)


def print_action_plan(plan: ActionPlan, decisions: list[PolicyDecision]) -> None:
    table = Table(title="Action Plan", expand=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Action", style="cyan")
    table.add_column("Target")
    table.add_column("Risk", justify="center")
    table.add_column("Status", justify="center")

    decision_map = {d.action_id: d for d in decisions}
    for i, action in enumerate(plan.actions, 1):
        d = decision_map.get(action.id)
        risk = d.risk_level.name if d else "?"
        status = "[green]APPROVED[/]" if d and d.approved else "[red]DENIED[/]"
        table.add_row(str(i), action.description, action.target, risk, status)

    console.print(table)


def print_results(results: list[ActionResult]) -> None:
    table = Table(title="Execution Results", expand=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Status", justify="center")
    table.add_column("Output")

    for i, r in enumerate(results, 1):
        status = "[green]OK[/]" if r.success else "[red]FAIL[/]"
        output = r.output if r.success else r.error
        table.add_row(str(i), status, output)

    console.print(table)


def print_error(message: str) -> None:
    console.print(Panel(f"[red]{message}[/]", title="Error", border_style="red"))


def print_info(message: str) -> None:
    console.print(f"[dim]{message}[/]")


def print_history(rows: list[dict]) -> None:
    table = Table(title="Action History", expand=True)
    table.add_column("Time")
    table.add_column("Query")
    table.add_column("Intent")
    table.add_column("Action")
    table.add_column("Approved", justify="center")

    for row in rows:
        approved = ""
        if row.get("approved") is not None:
            approved = "[green]Yes[/]" if row["approved"] else "[red]No[/]"
        table.add_row(
            str(row.get("created_at", "")),
            row.get("raw_query", ""),
            row.get("intent_type", ""),
            row.get("action_type", "") or "",
            approved,
        )

    console.print(table)


def print_status(cpu: float, memory_percent: float, top_procs: list[dict]) -> None:
    table = Table(title="System Status", show_header=False, expand=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("CPU Usage", f"{cpu:.1f}%")
    table.add_row("Memory Usage", f"{memory_percent:.1f}%")
    console.print(table)

    if top_procs:
        ptable = Table(title="Top Processes (by memory)", expand=True)
        ptable.add_column("PID", style="bold")
        ptable.add_column("Name")
        ptable.add_column("Memory %", justify="right")
        ptable.add_column("CPU %", justify="right")
        for p in top_procs:
            ptable.add_row(
                str(p["pid"]),
                p["name"],
                f"{p['memory_percent']:.1f}%",
                f"{p['cpu_percent']:.1f}%",
            )
        console.print(ptable)
