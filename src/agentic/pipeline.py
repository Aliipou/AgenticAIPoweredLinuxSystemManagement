"""Pipeline orchestrator — wires parse → context → decide → evaluate → execute → log."""

from __future__ import annotations

import uuid

from agentic.engine.decision_engine import DecisionEngine
from agentic.exceptions import PolicyDeniedError, UserCancelledError
from agentic.executor.action_executor import ActionExecutor
from agentic.memory.context import ContextRetriever
from agentic.memory.models import ActionRecord, ExecutionRecord, RequestRecord
from agentic.memory.store import MemoryStore
from agentic.models.action import ActionPlan, ActionResult
from agentic.models.intent import IntentType, ParsedIntent
from agentic.parser.intent_parser import IntentParser
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
    ) -> None:
        self._parser = parser
        self._engine = engine
        self._gate = gate
        self._executor = executor
        self._store = store
        self._context = context_retriever
        self._dry_run = dry_run
        self._confirm_callback = confirm_callback

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

        # 4. Check for low-confidence / unknown
        if intent.intent_type == IntentType.UNKNOWN:
            return intent, ActionPlan(intent_id=intent.id, reasoning="Unknown intent"), []

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

        # 7. User confirmation (if needed and not dry-run)
        needs_confirm = any(d.requires_confirmation for d in approved_decisions)
        if needs_confirm and not self._dry_run and self._confirm_callback:
            confirmed = self._confirm_callback(approved_actions, approved_decisions)
            if not confirmed:
                raise UserCancelledError("User cancelled execution.")

        # 8. Execute
        results = await self._executor.execute_many(
            approved_actions, dry_run=self._dry_run
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
