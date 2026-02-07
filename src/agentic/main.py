"""Entry point and dependency wiring."""

from __future__ import annotations

from agentic.cli.app import app
from agentic.cli.prompts import confirm_execution
from agentic.config.settings import Settings
from agentic.engine.action_registry import ActionRegistry
from agentic.engine.decision_engine import DecisionEngine
from agentic.executor.action_executor import ActionExecutor
from agentic.memory.context import ContextRetriever
from agentic.memory.store import MemoryStore
from agentic.parser.intent_parser import IntentParser
from agentic.pipeline import Pipeline
from agentic.policy.safety_gate import SafetyGate


def build_pipeline(
    dry_run: bool = False,
    force: bool = False,
    settings: Settings | None = None,
) -> Pipeline:
    settings = settings or Settings()  # type: ignore[call-arg]

    store = MemoryStore(db_path=settings.db_path)
    context_retriever = ContextRetriever(store)
    parser = IntentParser(settings)
    registry = ActionRegistry()
    engine = DecisionEngine(registry)
    gate = SafetyGate(
        max_risk_level=settings.max_risk_level,
        force=force,
    )
    executor = ActionExecutor()

    return Pipeline(
        parser=parser,
        engine=engine,
        gate=gate,
        executor=executor,
        store=store,
        context_retriever=context_retriever,
        dry_run=dry_run or settings.dry_run,
        confirm_callback=confirm_execution if settings.require_confirmation else None,
    )


if __name__ == "__main__":
    app()
