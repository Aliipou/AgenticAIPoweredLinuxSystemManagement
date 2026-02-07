"""Brutal tests for all strategy implementations."""

from __future__ import annotations

import pytest

from agentic.engine.strategies.clean_memory import CleanMemoryStrategy
from agentic.engine.strategies.focus import DEFAULT_DISTRACTIONS, FocusStrategy
from agentic.engine.strategies.update import UpdateStrategy
from agentic.models.action import ActionType
from agentic.models.intent import Entity, IntentType, ParsedIntent


class TestFocusStrategy:
    @pytest.mark.asyncio
    async def test_generates_suspend_for_specific_process(self):
        intent = ParsedIntent(
            raw_query="close firefox",
            intent_type=IntentType.FOCUS,
            confidence=0.9,
            entities=[Entity(name="process", value="firefox", source="firefox")],
        )
        strategy = FocusStrategy()
        actions = await strategy.generate_actions(intent)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.SUSPEND_PROCESS
        assert actions[0].target == "firefox"

    @pytest.mark.asyncio
    async def test_uses_defaults_when_no_entities(self):
        intent = ParsedIntent(
            raw_query="help me focus",
            intent_type=IntentType.FOCUS,
            confidence=0.9,
            entities=[],
        )
        strategy = FocusStrategy()
        actions = await strategy.generate_actions(intent)
        assert len(actions) == len(DEFAULT_DISTRACTIONS)
        targets = [a.target for a in actions]
        for d in DEFAULT_DISTRACTIONS:
            assert d in targets

    @pytest.mark.asyncio
    async def test_ignores_non_process_entities(self):
        intent = ParsedIntent(
            raw_query="focus and install vim",
            intent_type=IntentType.FOCUS,
            confidence=0.9,
            entities=[Entity(name="package", value="vim", source="vim")],
        )
        strategy = FocusStrategy()
        actions = await strategy.generate_actions(intent)
        assert len(actions) == len(DEFAULT_DISTRACTIONS)

    @pytest.mark.asyncio
    async def test_multiple_process_entities(self):
        intent = ParsedIntent(
            raw_query="close slack and discord",
            intent_type=IntentType.FOCUS,
            confidence=0.9,
            entities=[
                Entity(name="process", value="slack", source="slack"),
                Entity(name="process", value="discord", source="discord"),
            ],
        )
        strategy = FocusStrategy()
        actions = await strategy.generate_actions(intent)
        assert len(actions) == 2
        assert actions[0].target == "slack"
        assert actions[1].target == "discord"

    @pytest.mark.asyncio
    async def test_rollback_command_set(self):
        intent = ParsedIntent(
            raw_query="close firefox",
            intent_type=IntentType.FOCUS,
            confidence=0.9,
            entities=[Entity(name="process", value="firefox", source="firefox")],
        )
        strategy = FocusStrategy()
        actions = await strategy.generate_actions(intent)
        assert actions[0].rollback_command != ""
        assert "CONT" in actions[0].rollback_command


class TestUpdateStrategy:
    @pytest.mark.asyncio
    async def test_specific_package_install(self):
        intent = ParsedIntent(
            raw_query="install vim",
            intent_type=IntentType.UPDATE,
            confidence=0.9,
            entities=[Entity(name="package", value="vim", source="vim")],
        )
        strategy = UpdateStrategy()
        actions = await strategy.generate_actions(intent)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.APT_INSTALL
        assert actions[0].target == "vim"
        assert "vim" in actions[0].command

    @pytest.mark.asyncio
    async def test_multiple_packages(self):
        intent = ParsedIntent(
            raw_query="install vim and git",
            intent_type=IntentType.UPDATE,
            confidence=0.9,
            entities=[
                Entity(name="package", value="vim", source="vim"),
                Entity(name="package", value="git", source="git"),
            ],
        )
        strategy = UpdateStrategy()
        actions = await strategy.generate_actions(intent)
        assert len(actions) == 2
        assert actions[0].target == "vim"
        assert actions[1].target == "git"

    @pytest.mark.asyncio
    async def test_no_packages_does_full_upgrade(self):
        intent = ParsedIntent(
            raw_query="update everything",
            intent_type=IntentType.UPDATE,
            confidence=0.9,
            entities=[],
        )
        strategy = UpdateStrategy()
        actions = await strategy.generate_actions(intent)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.APT_UPGRADE
        assert "upgrade" in actions[0].command

    @pytest.mark.asyncio
    async def test_install_has_rollback(self):
        intent = ParsedIntent(
            raw_query="install htop",
            intent_type=IntentType.UPDATE,
            confidence=0.9,
            entities=[Entity(name="package", value="htop", source="htop")],
        )
        strategy = UpdateStrategy()
        actions = await strategy.generate_actions(intent)
        assert "remove" in actions[0].rollback_command

    @pytest.mark.asyncio
    async def test_upgrade_has_no_rollback(self):
        intent = ParsedIntent(
            raw_query="upgrade system",
            intent_type=IntentType.UPDATE,
            confidence=0.9,
            entities=[],
        )
        strategy = UpdateStrategy()
        actions = await strategy.generate_actions(intent)
        assert actions[0].rollback_command == ""

    @pytest.mark.asyncio
    async def test_ignores_non_package_entities(self):
        intent = ParsedIntent(
            raw_query="update and close firefox",
            intent_type=IntentType.UPDATE,
            confidence=0.9,
            entities=[Entity(name="process", value="firefox", source="firefox")],
        )
        strategy = UpdateStrategy()
        actions = await strategy.generate_actions(intent)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.APT_UPGRADE


class TestCleanMemoryStrategy:
    @pytest.mark.asyncio
    async def test_always_drops_caches(self):
        intent = ParsedIntent(
            raw_query="free memory",
            intent_type=IntentType.CLEAN_MEMORY,
            confidence=0.9,
            entities=[],
        )
        strategy = CleanMemoryStrategy()
        actions = await strategy.generate_actions(intent)
        drop_cache = [a for a in actions if a.action_type == ActionType.DROP_CACHES]
        assert len(drop_cache) == 1

    @pytest.mark.asyncio
    async def test_kills_specific_process(self):
        intent = ParsedIntent(
            raw_query="kill electron to free RAM",
            intent_type=IntentType.CLEAN_MEMORY,
            confidence=0.9,
            entities=[Entity(name="process", value="electron", source="electron")],
        )
        strategy = CleanMemoryStrategy()
        actions = await strategy.generate_actions(intent)
        kill_actions = [a for a in actions if a.action_type == ActionType.KILL_BY_MEMORY]
        assert len(kill_actions) == 1
        assert kill_actions[0].target == "electron"

    @pytest.mark.asyncio
    async def test_generic_memory_hog_killing(self):
        intent = ParsedIntent(
            raw_query="free up RAM",
            intent_type=IntentType.CLEAN_MEMORY,
            confidence=0.9,
            entities=[],
        )
        strategy = CleanMemoryStrategy()
        actions = await strategy.generate_actions(intent)
        kill_actions = [a for a in actions if a.action_type == ActionType.KILL_BY_MEMORY]
        assert len(kill_actions) == 1
        assert kill_actions[0].target == "memory_hogs"

    @pytest.mark.asyncio
    async def test_multiple_memory_hog_targets(self):
        intent = ParsedIntent(
            raw_query="kill chrome and electron",
            intent_type=IntentType.CLEAN_MEMORY,
            confidence=0.9,
            entities=[
                Entity(name="process", value="chrome", source="chrome"),
                Entity(name="process", value="electron", source="electron"),
            ],
        )
        strategy = CleanMemoryStrategy()
        actions = await strategy.generate_actions(intent)
        kill_actions = [a for a in actions if a.action_type == ActionType.KILL_BY_MEMORY]
        assert len(kill_actions) == 2

    @pytest.mark.asyncio
    async def test_drop_caches_command(self):
        intent = ParsedIntent(
            raw_query="clear cache",
            intent_type=IntentType.CLEAN_MEMORY,
            confidence=0.9,
            entities=[],
        )
        strategy = CleanMemoryStrategy()
        actions = await strategy.generate_actions(intent)
        drop = [a for a in actions if a.action_type == ActionType.DROP_CACHES][0]
        assert "drop_caches" in drop.command
