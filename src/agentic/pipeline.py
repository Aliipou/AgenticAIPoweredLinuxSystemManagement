"""Pipeline orchestrator — wires parse → context → decide → evaluate → execute → log."""

from __future__ import annotations

import uuid

from agentic.engine.decision_engine import DecisionEngine
from agentic.exceptions import LowConfidenceError, PolicyDeniedError, UnsafeCommandError, UserCancelledError
from agentic.executor.action_executor import ActionExecutor
from agentic.executor.command_validator import CommandValidator
from agentic.memory.context import ContextRetriever
from agentic.memory.models import ActionRecord, ExecutionRecord, RequestRecord
from agentic.memory.store import MemoryStore
from agentic.models.action import ActionPlan, ActionResult
from agentic.models.intent import IntentType, ParsedIntent
from agentic.parser.intent_parser import IntentParser
from agentic.policy.confidence_gate import ConfidenceGate
from agentic.policy.safety_gate import SafetyGate


class Pipeline:
    def __init__(
        self,
        parser: IntentParser,
        engine: DecisionEngine,
        gate: SafetyGate,
        executor: ActionExecutor,
        store: MemoryStore,
        context_retriever: ContextRetriever,
        dry_run: bool = False,
        confirm_callback=None,
        confidence_gate: ConfidenceGate | None = None,
        command_validator: CommandValidator | None = None,
    ) -> None:
        self._parser = parser
        self._engine = engine
        self._gate = gate
        self._executor = executor
        self._store = store
        self._context = context_retriever
        self._dry_run = dry_run
        self._confirm_callback = confirm_callback
        self._confidence_gate = confidence_gate
        self._command_validator = command_validator

    async def run(self, query: str) -> tuple[ParsedIntent, ActionPlan, list[ActionResult]]:
        # 1. Get context
        context_str = await self._context.format_context(query)

        # 2. Parse intent
        intent = await self._parser.parse(query, context=context_str)

        # 3. Log request
        await self._store.log_request(
            RequestRecord(
                id=intent.id,
                raw_query=query,
                intent_type=intent.intent_type.value,
                confidence=intent.confidence,
            )
        )

        # 4. Check for UNKNOWN
        if intent.intent_type == IntentType.UNKNOWN:
            return intent, ActionPlan(intent_id=intent.id, reasoning="Unknown intent"), []

        # 4.5: Confidence gate — deterministic guard against LLM hallucination
        effective_dry_run = self._dry_run
        if self._confidence_gate is not None:
            cd = self._confidence_gate.evaluate(intent)
            if not cd.passed:
                raise LowConfidenceError(cd.reason)
            if cd.force_dry_run:
                effective_dry_run = True

        # 5. Generate action plan
        plan = await self._engine.decide(intent)

        if not plan.actions:
            return intent, plan, []

        # 6. Evaluate policy
        decisions = self._gate.evaluate_plan(plan)
        approved_actions, approved_decisions = self._gate.filter_approved(plan, decisions)

        # Log policy decisions
        for d in decisions:
            await self._store.log_policy_decision(
                action_id=d.action_id,
                risk_level=d.risk_level.value,
                approved=d.approved,
                requires_sudo=d.requires_sudo,
                reason=d.reason,
            )

        if not approved_actions:
            raise PolicyDeniedError("All actions were denied by the safety gate.")

        # Log approved actions
        for action in approved_actions:
            decision_map = {d.action_id: d for d in approved_decisions}
            d = decision_map.get(action.id)
            await self._store.log_action(
                ActionRecord(
                    id=action.id,
                    request_id=intent.id,
                    action_type=action.action_type.value,
                    description=action.description,
                    command=action.command,
                    risk_level=d.risk_level.value if d else 1,
                    approved=True,
                )
            )

        # 6.5: Command validator — deterministic pre-flight safety check
        if self._command_validator is not None:
            for action, vr in self._command_validator.validate_many(approved_actions):
                if not vr.valid:
                    raise UnsafeCommandError(vr.reason, action_id=action.id)

        # 7. User confirmation (if needed and not in dry-run mode)
        needs_confirm = any(d.requires_confirmation for d in approved_decisions)
        if needs_confirm and not effective_dry_run and self._confirm_callback:
            confirmed = self._confirm_callback(approved_actions, approved_decisions)
            if not confirmed:
                raise UserCancelledError("User cancelled execution.")

        # 8. Execute
        results = await self._executor.execute_many(
            approved_actions, dry_run=effective_dry_run
        )

        # 9. Log results
        for result in results:
            await self._store.log_execution(
                ExecutionRecord(
                    id=uuid.uuid4().hex,
                    action_id=result.action_id,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    rolled_back=result.rolled_back,
                )
            )

        return intent, plan, results
