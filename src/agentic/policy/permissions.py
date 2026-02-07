"""Permission matrix — maps ActionType to (RiskLevel, requires_sudo)."""

from __future__ import annotations

from agentic.models.action import ActionType
from agentic.models.policy import RiskLevel

# Each entry: (RiskLevel, requires_sudo)
PERMISSION_MATRIX: dict[ActionType, tuple[RiskLevel, bool]] = {
    ActionType.KILL_PROCESS: (RiskLevel.MEDIUM, False),
    ActionType.SUSPEND_PROCESS: (RiskLevel.LOW, False),
    ActionType.RENICE_PROCESS: (RiskLevel.LOW, False),
    ActionType.APT_INSTALL: (RiskLevel.MEDIUM, True),
    ActionType.APT_UPGRADE: (RiskLevel.HIGH, True),
    ActionType.DROP_CACHES: (RiskLevel.MEDIUM, True),
    ActionType.KILL_BY_MEMORY: (RiskLevel.HIGH, False),
    ActionType.SYSTEMCTL_START: (RiskLevel.MEDIUM, True),
    ActionType.SYSTEMCTL_STOP: (RiskLevel.HIGH, True),
    ActionType.SYSTEMCTL_RESTART: (RiskLevel.HIGH, True),
}
