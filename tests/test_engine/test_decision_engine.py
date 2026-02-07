"""Brutal tests for decision engine and action registry."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agentic.engine.action_registry import ActionRegistry
from agentic.engine.decision_engine import DecisionEngine
from agentic.engine.strategies.base import IntentStrategy
from agentic.models.action import ActionCandidate, ActionType
from agentic.models.intent import IntentType, ParsedIntent


class TestActionRegistry:
    def test_default_strategies_registered(self):
        registry = ActionRegistry()
        assert registry.get(IntentType.FOCUS) is not None
        assert registry.get(IntentType.UPDATE) is not None
        assert registry.get(IntentType.CLEAN_MEMORY) is not None

    def test_unknown_returns_none(self):
        registry = ActionRegistry()
        assert registry.get(IntentType.UNKNOWN) is None

    def test_register_custom_strategy(self):
        registry = ActionRegistry()
        mock_strategy = AsyncMock(spec=IntentStrategy)
        registry.register(IntentType.UNKNOWN, mock_strategy)
        assert registry.get(IntentType.UNKNOWN) is mock_strategy

    def test_register_overwrites_existing(self):
        registry = ActionRegistry()
        original = registry.get(IntentType.FOCUS)
        new_strategy = AsyncMock(spec=IntentStrategy)
        registry.register(IntentType.FOCUS, new_strategy)
        assert registry.get(IntentType.FOCUS) is new_strategy
        assert registry.get(IntentType.FOCUS) is not original


class TestDecisionEngine:
    @pytest.mark.asyncio
    async def test_decide_unknown_returns_empty_plan(self, sample_unknown_intent):
        engine = DecisionEngine()
        plan = await engine.decide(sample_unknown_intent)
        assert plan.actions == []
        assert "UNKNOWN" in plan.reasoning

    @pytest.mark.asyncio
    async def test_decide_focus_generates_actions(self, sample_focus_intent):
        engine = DecisionEngine()
        plan = await engine.decide(sample_focus_intent)
        assert len(plan.actions) > 0
        assert plan.intent_id == sample_focus_intent.id

    @pytest.mark.asyncio
    async def test_decide_update_generates_actions(self, sample_update_intent):
        engine = DecisionEngine()
        plan = await engine.decide(sample_update_intent)
        assert len(plan.actions) > 0
        assert any(
            a.action_type in (ActionType.APT_INSTALL, ActionType.APT_UPGRADE)
            for a in plan.actions
        )

    @pytest.mark.asyncio
    async def test_decide_clean_memory_generates_actions(self, sample_clean_memory_intent):
        engine = DecisionEngine()
        plan = await engine.decide(sample_clean_memory_intent)
        assert len(plan.actions) > 0
        assert any(
            a.action_type in (ActionType.DROP_CACHES, ActionType.KILL_BY_MEMORY)
            for a in plan.actions
        )

    @pytest.mark.asyncio
    async def test_decide_no_matching_strategy(self):
        registry = ActionRegistry()
        # Remove FOCUS strategy
        registry._strategies.pop(IntentType.FOCUS, None)
        engine = DecisionEngine(registry)

        intent = ParsedIntent(
            raw_query="focus",
            intent_type=IntentType.FOCUS,
            confidence=0.9,
        )
        plan = await engine.decide(intent)
        assert plan.actions == []
        assert "No strategy" in plan.reasoning

    @pytest.mark.asyncio
    async def test_decide_reasoning_includes_count(self, sample_focus_intent):
        engine = DecisionEngine()
        plan = await engine.decide(sample_focus_intent)
        assert "action(s)" in plan.reasoning

    @pytest.mark.asyncio
    async def test_plan_has_intent_id(self, sample_focus_intent):
        engine = DecisionEngine()
        plan = await engine.decide(sample_focus_intent)
        assert plan.intent_id == sample_focus_intent.id
