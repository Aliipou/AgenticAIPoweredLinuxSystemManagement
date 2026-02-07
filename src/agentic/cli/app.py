"""Typer CLI commands."""

from __future__ import annotations

import asyncio
from typing import Optional

import psutil
import typer
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
from agentic.cli.prompts import confirm_execution, display_dry_run
from agentic.exceptions import AgenticError

console = Console()
app = typer.Typer(name="agentic", help="AI-powered Linux system management.")


def _get_pipeline(dry_run: bool = False, force: bool = False):
    from agentic.main import build_pipeline
    return build_pipeline(dry_run=dry_run, force=force)


@app.command()
def ask(
    query: str = typer.Argument(..., help="Natural language request"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without executing"),
    force: bool = typer.Option(False, "--force", help="Skip confirmations"),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output"),
) -> None:
    """Parse a natural language request and execute system actions."""
    async def _run():
        pipeline = _get_pipeline(dry_run=dry_run, force=force)
        await pipeline._store.initialize()
        try:
            intent, plan, results = await pipeline.run(query)

            print_intent(intent)

            if plan.actions:
                decisions = pipeline._gate.evaluate_plan(plan)
                print_action_plan(plan, decisions)

                if dry_run:
                    approved, approved_d = pipeline._gate.filter_approved(plan, decisions)
                    display_dry_run(approved, approved_d)

            if results:
                print_results(results)
        except AgenticError as exc:
            print_error(str(exc))
            raise typer.Exit(1)
        finally:
            await pipeline._store.close()

    asyncio.run(_run())


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of records"),
) -> None:
    """Show recent action history."""
    async def _run():
        pipeline = _get_pipeline()
        await pipeline._store.initialize()
        try:
            rows = await pipeline._store.get_history(limit=limit)
            if not rows:
                print_info("No history found.")
            else:
                print_history(rows)
        finally:
            await pipeline._store.close()

    asyncio.run(_run())


@app.command()
def rollback(
    action_id: str = typer.Argument(..., help="Action ID to rollback"),
) -> None:
    """Rollback a previously executed action."""
    print_info(f"Rollback for action {action_id} — not yet implemented.")
    raise typer.Exit(0)


@app.command()
def status() -> None:
    """Show current system CPU/memory/top processes."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent

    procs: list[dict] = []
    for proc in psutil.process_iter(["pid", "name", "memory_percent", "cpu_percent"]):
        try:
            info = proc.info
            procs.append({
                "pid": info["pid"],
                "name": info.get("name", ""),
                "memory_percent": info.get("memory_percent", 0.0) or 0.0,
                "cpu_percent": info.get("cpu_percent", 0.0) or 0.0,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    procs.sort(key=lambda p: p["memory_percent"], reverse=True)
    print_status(cpu, mem, procs[:10])


@app.command(name="config")
def show_config() -> None:
    """Show current configuration."""
    from agentic.config.settings import Settings

    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        print_error(f"Failed to load settings: {exc}")
        raise typer.Exit(1)

    table_data = {
        "Model": settings.openai_model,
        "DB Path": str(settings.db_path),
        "Dry Run": str(settings.dry_run),
        "Log Level": settings.log_level,
        "Max Risk": settings.max_risk_level,
        "Require Confirmation": str(settings.require_confirmation),
    }

    from rich.table import Table
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value")
    for k, v in table_data.items():
        table.add_row(k, v)
    console.print(table)
