"""Brutal tests for main entry point and dependency wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentic.engine.decision_engine import DecisionEngine
from agentic.executor.action_executor import ActionExecutor
from agentic.main import build_pipeline
from agentic.memory.context import ContextRetriever
from agentic.memory.store import MemoryStore
from agentic.parser.intent_parser import IntentParser
from agentic.pipeline import Pipeline
from agentic.policy.safety_gate import SafetyGate


class TestBuildPipeline:
    def test_returns_pipeline(self, mock_settings):
        pipeline = build_pipeline(settings=mock_settings)
        assert isinstance(pipeline, Pipeline)

    def test_wires_parser(self, mock_settings):
        pipeline = build_pipeline(settings=mock_settings)
        assert isinstance(pipeline._parser, IntentParser)

    def test_wires_engine(self, mock_settings):
        pipeline = build_pipeline(settings=mock_settings)
        assert isinstance(pipeline._engine, DecisionEngine)

    def test_wires_gate(self, mock_settings):
        pipeline = build_pipeline(settings=mock_settings)
        assert isinstance(pipeline._gate, SafetyGate)

    def test_wires_executor(self, mock_settings):
        pipeline = build_pipeline(settings=mock_settings)
        assert isinstance(pipeline._executor, ActionExecutor)

    def test_wires_store(self, mock_settings):
        pipeline = build_pipeline(settings=mock_settings)
        assert isinstance(pipeline._store, MemoryStore)

    def test_wires_context_retriever(self, mock_settings):
        pipeline = build_pipeline(settings=mock_settings)
        assert isinstance(pipeline._context, ContextRetriever)

    def test_dry_run_propagated(self, mock_settings):
        pipeline = build_pipeline(dry_run=True, settings=mock_settings)
        assert pipeline._dry_run is True

    def test_force_disables_confirmation(self, mock_settings):
        pipeline = build_pipeline(force=True, settings=mock_settings)
        # When force=True and require_confirmation=False (from mock_settings),
        # confirm_callback should be None
        assert pipeline._confirm_callback is None

    def test_settings_dry_run_override(self, monkeypatch):
        monkeypatch.setenv("AGENTIC_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AGENTIC_DRY_RUN", "true")
        monkeypatch.setenv("AGENTIC_REQUIRE_CONFIRMATION", "false")
        pipeline = build_pipeline()
        assert pipeline._dry_run is True

    def test_default_settings_loaded(self, monkeypatch):
        monkeypatch.setenv("AGENTIC_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("AGENTIC_REQUIRE_CONFIRMATION", "false")
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)
