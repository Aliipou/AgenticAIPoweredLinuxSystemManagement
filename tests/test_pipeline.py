"""Brutal tests for the pipeline orchestrator."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic.exceptions import PolicyDeniedError, UserCancelledError
from agentic.models.action import ActionCandidate, ActionPlan, ActionResult, ActionType
from agentic.models.intent import IntentType, ParsedIntent
from agentic.models.policy import PolicyDecision, RiskLevel
from agentic.pipeline import Pipeline


@pytest.fixture
def mock_pipeline_deps():
    parser = AsyncMock()
    engine = AsyncMock()
    gate = MagicMock()
    executor = AsyncMock()
    store = AsyncMock()
    context_retriever = AsyncMock()
    context_retriever.format_context = AsyncMock(return_value="No previous context.")

    return {
        "parser": parser,
        "engine": engine,
        "gate": gate,
        "executor": executor,
        "store": store,
        "context_retriever": context_retriever,
    }


def _make_intent(intent_type=IntentType.FOCUS, confidence=0.9):
    return ParsedIntent(
        id="intent-test",
        raw_query="test query",
        intent_type=intent_type,
        confidence=confidence,
    )


def _make_action(action_id="act-test"):
    return ActionCandidate(
        id=action_id,
        action_type=ActionType.SUSPEND_PROCESS,
        description="Suspend test",
        command="test cmd",
        target="test",
    )


def _make_plan(intent_id="intent-test", actions=None):
    return ActionPlan(
        id="plan-test",
        intent_id=intent_id,
        actions=actions or [],
        reasoning="test",
    )


def _make_decision(action_id="act-test", approved=True, risk=RiskLevel.LOW):
    return PolicyDecision(
        action_id=action_id,
        risk_level=risk,
        approved=approved,
        requires_confirmation=False,
    )


class TestPipelineRun:
    @pytest.mark.asyncio
    async def test_full_flow_success(self, mock_pipeline_deps):
        intent = _make_intent()
        action = _make_action()
        plan = _make_plan(actions=[action])
        decision = _make_decision(action_id=action.id)
        result = ActionResult(action_id=action.id, success=True, output="done")

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)
        mock_pipeline_deps["gate"].evaluate_plan.return_value = [decision]
        mock_pipeline_deps["gate"].filter_approved.return_value = ([action], [decision])
        mock_pipeline_deps["executor"].execute_many = AsyncMock(return_value=[result])

        pipeline = Pipeline(**mock_pipeline_deps)
        r_intent, r_plan, r_results = await pipeline.run("test query")

        assert r_intent.intent_type == IntentType.FOCUS
        assert len(r_results) == 1
        assert r_results[0].success is True

    @pytest.mark.asyncio
    async def test_unknown_intent_returns_empty(self, mock_pipeline_deps):
        intent = _make_intent(intent_type=IntentType.UNKNOWN, confidence=0.2)

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)

        pipeline = Pipeline(**mock_pipeline_deps)
        r_intent, r_plan, r_results = await pipeline.run("tell me a joke")

        assert r_intent.intent_type == IntentType.UNKNOWN
        assert r_results == []

    @pytest.mark.asyncio
    async def test_empty_plan_returns_no_results(self, mock_pipeline_deps):
        intent = _make_intent()
        plan = _make_plan(actions=[])

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)

        pipeline = Pipeline(**mock_pipeline_deps)
        _, _, results = await pipeline.run("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_all_denied_raises_policy_error(self, mock_pipeline_deps):
        intent = _make_intent()
        action = _make_action()
        plan = _make_plan(actions=[action])
        decision = _make_decision(action_id=action.id, approved=False)

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)
        mock_pipeline_deps["gate"].evaluate_plan.return_value = [decision]
        mock_pipeline_deps["gate"].filter_approved.return_value = ([], [])

        pipeline = Pipeline(**mock_pipeline_deps)
        with pytest.raises(PolicyDeniedError):
            await pipeline.run("test")

    @pytest.mark.asyncio
    async def test_user_cancel_raises(self, mock_pipeline_deps):
        intent = _make_intent()
        action = _make_action()
        plan = _make_plan(actions=[action])
        decision = _make_decision(action_id=action.id, approved=True, risk=RiskLevel.MEDIUM)
        decision.requires_confirmation = True

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)
        mock_pipeline_deps["gate"].evaluate_plan.return_value = [decision]
        mock_pipeline_deps["gate"].filter_approved.return_value = ([action], [decision])

        pipeline = Pipeline(
            **mock_pipeline_deps,
            confirm_callback=lambda actions, decisions: False,
        )
        with pytest.raises(UserCancelledError):
            await pipeline.run("test")

    @pytest.mark.asyncio
    async def test_user_confirms_proceeds(self, mock_pipeline_deps):
        intent = _make_intent()
        action = _make_action()
        plan = _make_plan(actions=[action])
        decision = _make_decision(action_id=action.id, approved=True, risk=RiskLevel.MEDIUM)
        decision.requires_confirmation = True
        result = ActionResult(action_id=action.id, success=True, output="done")

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)
        mock_pipeline_deps["gate"].evaluate_plan.return_value = [decision]
        mock_pipeline_deps["gate"].filter_approved.return_value = ([action], [decision])
        mock_pipeline_deps["executor"].execute_many = AsyncMock(return_value=[result])

        pipeline = Pipeline(
            **mock_pipeline_deps,
            confirm_callback=lambda actions, decisions: True,
        )
        _, _, results = await pipeline.run("test")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, mock_pipeline_deps):
        intent = _make_intent()
        action = _make_action()
        plan = _make_plan(actions=[action])
        decision = _make_decision(action_id=action.id, approved=True)
        result = ActionResult(action_id=action.id, success=True, output="[DRY RUN]")

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)
        mock_pipeline_deps["gate"].evaluate_plan.return_value = [decision]
        mock_pipeline_deps["gate"].filter_approved.return_value = ([action], [decision])
        mock_pipeline_deps["executor"].execute_many = AsyncMock(return_value=[result])

        pipeline = Pipeline(**mock_pipeline_deps, dry_run=True)
        _, _, results = await pipeline.run("test")
        mock_pipeline_deps["executor"].execute_many.assert_called_once_with(
            [action], dry_run=True
        )

    @pytest.mark.asyncio
    async def test_no_confirmation_callback_skips_prompt(self, mock_pipeline_deps):
        intent = _make_intent()
        action = _make_action()
        plan = _make_plan(actions=[action])
        decision = _make_decision(action_id=action.id, approved=True, risk=RiskLevel.MEDIUM)
        decision.requires_confirmation = True
        result = ActionResult(action_id=action.id, success=True, output="done")

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)
        mock_pipeline_deps["gate"].evaluate_plan.return_value = [decision]
        mock_pipeline_deps["gate"].filter_approved.return_value = ([action], [decision])
        mock_pipeline_deps["executor"].execute_many = AsyncMock(return_value=[result])

        pipeline = Pipeline(**mock_pipeline_deps, confirm_callback=None)
        _, _, results = await pipeline.run("test")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_logs_request(self, mock_pipeline_deps):
        intent = _make_intent(intent_type=IntentType.UNKNOWN)

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)

        pipeline = Pipeline(**mock_pipeline_deps)
        await pipeline.run("test")
        mock_pipeline_deps["store"].log_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_policy_decisions(self, mock_pipeline_deps):
        intent = _make_intent()
        action = _make_action()
        plan = _make_plan(actions=[action])
        decision = _make_decision(action_id=action.id, approved=True)
        result = ActionResult(action_id=action.id, success=True, output="done")

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)
        mock_pipeline_deps["gate"].evaluate_plan.return_value = [decision]
        mock_pipeline_deps["gate"].filter_approved.return_value = ([action], [decision])
        mock_pipeline_deps["executor"].execute_many = AsyncMock(return_value=[result])

        pipeline = Pipeline(**mock_pipeline_deps)
        await pipeline.run("test")
        mock_pipeline_deps["store"].log_policy_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_actions(self, mock_pipeline_deps):
        intent = _make_intent()
        action = _make_action()
        plan = _make_plan(actions=[action])
        decision = _make_decision(action_id=action.id, approved=True)
        result = ActionResult(action_id=action.id, success=True, output="done")

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)
        mock_pipeline_deps["gate"].evaluate_plan.return_value = [decision]
        mock_pipeline_deps["gate"].filter_approved.return_value = ([action], [decision])
        mock_pipeline_deps["executor"].execute_many = AsyncMock(return_value=[result])

        pipeline = Pipeline(**mock_pipeline_deps)
        await pipeline.run("test")
        mock_pipeline_deps["store"].log_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_execution_results(self, mock_pipeline_deps):
        intent = _make_intent()
        action = _make_action()
        plan = _make_plan(actions=[action])
        decision = _make_decision(action_id=action.id, approved=True)
        result = ActionResult(action_id=action.id, success=True, output="done")

        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)
        mock_pipeline_deps["engine"].decide = AsyncMock(return_value=plan)
        mock_pipeline_deps["gate"].evaluate_plan.return_value = [decision]
        mock_pipeline_deps["gate"].filter_approved.return_value = ([action], [decision])
        mock_pipeline_deps["executor"].execute_many = AsyncMock(return_value=[result])

        pipeline = Pipeline(**mock_pipeline_deps)
        await pipeline.run("test")
        mock_pipeline_deps["store"].log_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_retriever_called(self, mock_pipeline_deps):
        intent = _make_intent(intent_type=IntentType.UNKNOWN)
        mock_pipeline_deps["parser"].parse = AsyncMock(return_value=intent)

        pipeline = Pipeline(**mock_pipeline_deps)
        await pipeline.run("test")
        mock_pipeline_deps["context_retriever"].format_context.assert_called_once_with("test")
