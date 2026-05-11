"""Tests for SimulationEngine."""

from __future__ import annotations

import pytest

from agentic.executor.simulation_engine import SimulationEngine, _ACTION_SCOPES, _AVAILABILITY_IMPACT, _HIGH_IMPACT
from agentic.models.action import (
    ActionCandidate,
    ActionEffect,
    ActionPlan,
    ActionScope,
    ActionSimulation,
    ActionType,
)


class TestSimulationEngineDerived:
    """Tests for actions without a declared ActionEffect (derived from type)."""

    def test_process_action_has_process_scope(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="Kill", command="kill -9 123", target="firefox")
        sim = engine.simulate(action)
        assert sim.predicted_scope == ActionScope.PROCESS
        assert sim.action_id == action.id

    def test_package_action_has_package_scope(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.APT_INSTALL, description="Install")
        sim = engine.simulate(action)
        assert sim.predicted_scope == ActionScope.PACKAGE

    def test_memory_action_has_memory_scope(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.DROP_CACHES, description="Drop")
        sim = engine.simulate(action)
        assert sim.predicted_scope == ActionScope.MEMORY

    def test_service_action_has_service_scope(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.SYSTEMCTL_START, description="Start")
        sim = engine.simulate(action)
        assert sim.predicted_scope == ActionScope.SERVICE

    def test_low_impact_action_is_reversible(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.SUSPEND_PROCESS, description="Suspend")
        sim = engine.simulate(action)
        assert sim.reversible is True

    def test_high_impact_action_is_not_reversible(self):
        engine = SimulationEngine()
        for at in _HIGH_IMPACT:
            action = ActionCandidate(action_type=at, description="test")
            sim = engine.simulate(action)
            assert sim.reversible is False, f"{at} should be not reversible"

    def test_high_impact_adds_irreversibility_warning(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.APT_UPGRADE, description="Upgrade")
        sim = engine.simulate(action)
        assert any("reversible" in w.lower() for w in sim.warnings)

    def test_availability_impact_for_systemctl_stop(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.SYSTEMCTL_STOP, description="Stop", target="nginx")
        sim = engine.simulate(action)
        assert sim.availability_impact is True
        assert any("availability" in w.lower() for w in sim.warnings)

    def test_no_availability_impact_for_start(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.SYSTEMCTL_START, description="Start")
        sim = engine.simulate(action)
        assert sim.availability_impact is False

    def test_derived_has_no_data_loss_risk(self):
        engine = SimulationEngine()
        for at in ActionType:
            action = ActionCandidate(action_type=at, description="test")
            sim = engine.simulate(action)
            assert sim.data_loss_risk is False

    def test_simulated_output_includes_command(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="Kill", command="kill -9 999")
        sim = engine.simulate(action)
        assert "[SIMULATED]" in sim.simulated_output
        assert "kill -9 999" in sim.simulated_output

    def test_simulated_output_falls_back_to_action_type(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.SUSPEND_PROCESS, description="Suspend")
        sim = engine.simulate(action)
        assert "SUSPEND_PROCESS" in sim.simulated_output

    def test_simulated_output_includes_target(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="Kill", command="kill 1", target="chrome")
        sim = engine.simulate(action)
        assert "chrome" in sim.simulated_output

    def test_simulated_output_no_target_no_on_label(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.DROP_CACHES, description="Drop", command="sync")
        sim = engine.simulate(action)
        assert " on " not in sim.simulated_output

    def test_apt_upgrade_requires_sudo(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.APT_UPGRADE, description="Upgrade")
        sim = engine.simulate(action)
        assert sim.would_require_sudo is True

    def test_kill_process_no_sudo(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="Kill")
        sim = engine.simulate(action)
        assert sim.would_require_sudo is False

    def test_availability_warning_uses_target_name(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.SYSTEMCTL_RESTART, description="Restart", target="postgresql")
        sim = engine.simulate(action)
        assert any("postgresql" in w for w in sim.warnings)

    def test_availability_warning_generic_when_no_target(self):
        engine = SimulationEngine()
        action = ActionCandidate(action_type=ActionType.SYSTEMCTL_STOP, description="Stop")
        sim = engine.simulate(action)
        assert any("service" in w for w in sim.warnings)


class TestSimulationEngineDeclared:
    """Tests for actions with a declared ActionEffect."""

    def test_declared_effect_overrides_derived(self):
        engine = SimulationEngine()
        effect = ActionEffect(scope=ActionScope.FILESYSTEM, reversible=False, data_loss_risk=True)
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Custom",
            effect=effect,
        )
        sim = engine.simulate(action)
        assert sim.predicted_scope == ActionScope.FILESYSTEM
        assert sim.reversible is False
        assert sim.data_loss_risk is True

    def test_declared_availability_adds_warning(self):
        engine = SimulationEngine()
        effect = ActionEffect(scope=ActionScope.SERVICE, availability_impact=True)
        action = ActionCandidate(action_type=ActionType.SYSTEMCTL_START, description="test", effect=effect)
        sim = engine.simulate(action)
        assert sim.availability_impact is True
        assert any("availability" in w.lower() for w in sim.warnings)

    def test_declared_data_loss_adds_warning(self):
        engine = SimulationEngine()
        effect = ActionEffect(scope=ActionScope.FILESYSTEM, data_loss_risk=True)
        action = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="test", effect=effect)
        sim = engine.simulate(action)
        assert sim.data_loss_risk is True
        assert any("data loss" in w.lower() for w in sim.warnings)

    def test_declared_reversible_no_irreversibility_warning(self):
        engine = SimulationEngine()
        effect = ActionEffect(scope=ActionScope.PROCESS, reversible=True)
        action = ActionCandidate(action_type=ActionType.APT_UPGRADE, description="test", effect=effect)
        sim = engine.simulate(action)
        assert sim.reversible is True
        assert not any("reversible" in w.lower() for w in sim.warnings)


class TestSimulationEngineSimulatePlan:
    def test_empty_plan_returns_empty(self):
        engine = SimulationEngine()
        plan = ActionPlan(intent_id="i1", actions=[])
        sims = engine.simulate_plan(plan)
        assert sims == []

    def test_plan_with_actions(self):
        engine = SimulationEngine()
        actions = [
            ActionCandidate(action_type=ActionType.KILL_PROCESS, description="a"),
            ActionCandidate(action_type=ActionType.APT_INSTALL, description="b"),
        ]
        plan = ActionPlan(intent_id="i1", actions=actions)
        sims = engine.simulate_plan(plan)
        assert len(sims) == 2
        assert all(isinstance(s, ActionSimulation) for s in sims)
        assert sims[0].action_id == actions[0].id
        assert sims[1].action_id == actions[1].id


class TestSimulationLookupTables:
    def test_all_action_types_in_scopes(self):
        for at in ActionType:
            assert at in _ACTION_SCOPES, f"{at} missing from _ACTION_SCOPES"

    def test_high_impact_is_subset_of_action_types(self):
        for at in _HIGH_IMPACT:
            assert at in ActionType.__members__.values()

    def test_availability_impact_is_subset_of_action_types(self):
        for at in _AVAILABILITY_IMPACT:
            assert at in ActionType.__members__.values()
