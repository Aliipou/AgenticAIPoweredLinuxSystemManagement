"""Permission matrix — maps ActionType to (RiskLevel, requires_sudo)."""

from __future__ import annotations

from agentic.models.action import ActionType, RollbackSupport
from agentic.models.environment import Environment
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

# Risk ceiling per deployment environment.
# Operators set AGENTIC_ENVIRONMENT to enforce this at boot time.
ENVIRONMENT_RISK_CAPS: dict[Environment, str] = {
    Environment.PRODUCTION: "MEDIUM",      # APT_UPGRADE / SYSTEMCTL_STOP require explicit --force
    Environment.STAGING: "HIGH",           # mirrors current SafetyGate default
    Environment.DEVELOPMENT: "CRITICAL",   # unrestricted — dev only
}

# Honest rollback truth table — what can actually be undone after execution.
# FULL    → prior state fully restorable
# PARTIAL → rollback attempts but residual effects remain (config, in-flight requests)
# NONE    → no rollback path exists; execution is permanent
ROLLBACK_CAPABILITIES: dict[ActionType, RollbackSupport] = {
    ActionType.SUSPEND_PROCESS: RollbackSupport.FULL,      # kill -CONT reverses kill -STOP exactly
    ActionType.RENICE_PROCESS: RollbackSupport.FULL,       # renice back to original value
    ActionType.DROP_CACHES: RollbackSupport.FULL,          # caches refill naturally from memory pressure
    ActionType.SYSTEMCTL_START: RollbackSupport.FULL,      # can stop what was started
    ActionType.SYSTEMCTL_STOP: RollbackSupport.FULL,       # can start what was stopped
    ActionType.SYSTEMCTL_RESTART: RollbackSupport.PARTIAL, # service restarts; in-flight requests lost
    ActionType.APT_INSTALL: RollbackSupport.PARTIAL,       # apt-remove works but config files remain
    ActionType.KILL_PROCESS: RollbackSupport.NONE,         # process terminated — cannot be undone
    ActionType.KILL_BY_MEMORY: RollbackSupport.NONE,       # process terminated — cannot be undone
    ActionType.APT_UPGRADE: RollbackSupport.NONE,          # no safe general downgrade path
}

# Services whose availability is critical; stopping/restarting any of these
# escalates to CRITICAL risk regardless of the base matrix entry.
CRITICAL_SERVICES: frozenset[str] = frozenset({
    "postgresql", "postgres", "pg",
    "mysql", "mysqld", "mariadb",
    "mongodb", "mongod",
    "nginx", "apache2", "httpd", "lighttpd", "haproxy",
    "docker", "dockerd", "containerd", "kubelet",
    "elasticsearch", "redis", "redis-server",
    "rabbitmq", "rabbitmq-server",
    "kafka",
    "sshd", "ssh",
    "ufw", "iptables", "nftables",
})
