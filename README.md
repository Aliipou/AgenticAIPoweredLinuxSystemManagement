<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-447%20passing-brightgreen?style=flat)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen?style=flat)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

# Policy-Constrained Autonomous Operations Runtime

**A policy-gated autonomous agent for Linux system management.**  
Not an AI assistant. Not a chatbot. A runtime with formal authority boundaries.

</div>

---

## What Makes This Different

Most "agentic" systems are LLM wrappers over shell — they parse a request and run a command. This system enforces six independent safety gates before any command reaches the OS:

1. **Confidence Gate** — if the LLM isn't certain enough, the action is refused or downgraded to dry-run. Confidence ≠ correctness, so both are tracked.
2. **Environment Gate** — deployment context enforces a hard risk ceiling. `PRODUCTION` caps at MEDIUM; `STAGING` at HIGH; `DEVELOPMENT` is unrestricted. Set via `AGENTIC_ENVIRONMENT`.
3. **Capability Gate** — least-privilege enforcement. Each action type maps to exactly one required `Capability`. The runtime is constructed with a `frozenset` of granted capabilities; actions needing a missing capability are denied before policy evaluation.
4. **Policy Gate** — every action type carries a static risk level and sudo requirement. Actions above the configured threshold are blocked. Stopping or restarting a critical service auto-escalates to CRITICAL, blocking without `--force`.
5. **Command Validator** — two-tier deterministic scan of the generated command and target strings. Syntactic tier: `rm -rf /`, fork bombs, raw disk writes. Semantic tier: `find / -delete`, `chmod -R 777 /`, `chown -R` on `/etc`, deletion of `/etc/passwd` or `/boot/*`. Blocked unconditionally.
6. **Simulation Engine** — before execution, each approved action is run through a prediction model that computes scope, reversibility, data-loss risk, and availability impact. Results attach to the `ActionPlan` for inspection.

If all gates pass, actions execute through the **Transaction Manager**, which rolls back previous actions in reverse order if any step fails.

Every decision — approved or denied — is recorded in the audit log.

---

## Pipeline

```
User query
    │
    ▼
[ContextRetriever]   Retrieves recent history for grounding
    │
    ▼
[IntentParser]       LLM classification → IntentType + confidence + entities
    │
    ▼
[ConfidenceGate]     confidence < 0.70  → reject (LowConfidenceError)
                     confidence < 0.85  → approve, force dry-run
                     confidence ≥ 0.85  → full approval
    │
    ▼
[DecisionEngine]     Maps intent → ActionPlan via registered strategies
    │
    ▼
[EnvironmentGate]    PRODUCTION  → max MEDIUM cap
                     STAGING     → max HIGH cap
                     DEVELOPMENT → unrestricted
    │
    ▼
[CapabilityGate]     frozenset[Capability] — deny if required cap not granted
    │
    ▼
[SafetyGate]         Per-action risk scoring against PERMISSION_MATRIX
                     Critical service targets escalate to CRITICAL
    │
    ▼
[CommandValidator]   Syntactic + semantic pattern scan
                     Rejects regardless of upstream approval
    │
    ▼
[SimulationEngine]   Predicts effect of each approved action (non-blocking)
                     Attaches ActionSimulation list to ActionPlan
    │
    ▼
[TransactionManager] Executes actions sequentially
                     On failure: rolls back succeeded actions in reverse order
    │
    ▼
[MemoryStore]        Full audit trail: request → decisions → simulations → results
```

---

## Intent Types

| Intent | Description | Read-only |
|--------|-------------|-----------|
| `FOCUS` | Kill/suspend distracting processes | No |
| `UPDATE` | Install or upgrade packages | No |
| `CLEAN_MEMORY` | Drop caches, kill memory hogs | No |
| `OBSERVE` | Inspect system state — processes, CPU, memory, disk | Yes |
| `NETWORK` | Manage interfaces, firewall, connectivity | No |
| `STORAGE` | Disk usage, cleanup, mount management | No |
| `UNKNOWN` | Unrecognised — no action taken | — |

---

## Capability System

Each action type requires exactly one capability. The runtime is granted a `frozenset[Capability]` at construction time — actions needing a capability outside that set are denied.

| Capability | Action Types |
|-----------|-------------|
| `KILL_PROCESS` | `KILL_PROCESS`, `KILL_BY_MEMORY` |
| `SUSPEND_PROCESS` | `SUSPEND_PROCESS` |
| `RENICE_PROCESS` | `RENICE_PROCESS` |
| `PACKAGE_MANAGEMENT` | `APT_INSTALL`, `APT_UPGRADE` |
| `MEMORY_MANAGEMENT` | `DROP_CACHES` |
| `SERVICE_MANAGEMENT` | `SYSTEMCTL_START`, `SYSTEMCTL_STOP`, `SYSTEMCTL_RESTART` |

---

## Risk Matrix

| Action | Risk | Requires sudo |
|--------|------|---------------|
| `SUSPEND_PROCESS` | LOW | No |
| `RENICE_PROCESS` | LOW | No |
| `KILL_PROCESS` | MEDIUM | No |
| `APT_INSTALL` | MEDIUM | Yes |
| `DROP_CACHES` | MEDIUM | Yes |
| `SYSTEMCTL_START` | MEDIUM | Yes |
| `APT_UPGRADE` | HIGH | Yes |
| `KILL_BY_MEMORY` | HIGH | No |
| `SYSTEMCTL_STOP` | HIGH → **CRITICAL**¹ | Yes |
| `SYSTEMCTL_RESTART` | HIGH → **CRITICAL**¹ | Yes |

¹ Escalates to CRITICAL when the target is a member of `CRITICAL_SERVICES`. Blocked without `--force`.

### Environment Risk Caps

| Environment | Max allowed risk | Set via |
|------------|-----------------|---------|
| `PRODUCTION` | MEDIUM | `AGENTIC_ENVIRONMENT=PRODUCTION` |
| `STAGING` | HIGH | `AGENTIC_ENVIRONMENT=STAGING` |
| `DEVELOPMENT` | CRITICAL (unrestricted) | default |

### Critical Services

`SYSTEMCTL_STOP` or `SYSTEMCTL_RESTART` targeting any of these escalates to CRITICAL:

`postgresql` · `mysql` · `mariadb` · `mongodb` · `nginx` · `apache2` · `haproxy` · `docker` · `containerd` · `kubelet` · `elasticsearch` · `redis` · `rabbitmq` · `kafka` · `sshd` · `ufw` · `iptables`

---

## Semantic Safety Patterns

The Command Validator rejects commands that are **dangerous by effect**, not just by syntax:

| Pattern | Example | Reason |
|---------|---------|--------|
| `find / -delete` | `find / -type f -mtime +30 -delete` | Recursive filesystem wipe |
| `find / -exec rm` | `find / -name '*.bak' -exec rm {} \;` | Recursive deletion via exec |
| `chmod -R [0-7]*77 /` | `chmod -R 0777 /` | World-writable root filesystem |
| `chown -R` on critical paths | `chown -R nobody /etc` | Ownership takeover of system dirs |
| `rm /etc/passwd` etc. | `rm /etc/shadow` | Deletion of auth/boot critical files |

---

## Action IR

Every `ActionCandidate` carries a formal specification of its intent:

```python
ActionCandidate(
    action_type=ActionType.SUSPEND_PROCESS,
    description="Suspend chrome to reduce CPU",
    command="kill -STOP $(pgrep chrome)",
    target="chrome",
    rollback_command="kill -CONT $(pgrep chrome)",
    preconditions=["process exists", "not in ESSENTIAL_PROCESSES"],
    postconditions=["process suspended", "CPU load reduced"],
    required_capabilities=["SUSPEND_PROCESS"],
    effect=ActionEffect(
        scope=ActionScope.PROCESS,
        reversible=True,
        availability_impact=False,
    ),
)
```

The `SimulationEngine` uses `effect` when declared, or derives predictions from `action_type` when absent.

---

## Protected Processes

The executor maintains a blocklist of processes that can never be targeted, regardless of policy approval:

- PID 1 essentials: `init`, `systemd`, `kthreadd`
- systemd managers: `systemd-journald`, `systemd-logind`, `systemd-udevd`, `systemd-resolved`, `systemd-networkd`, `systemd-timesyncd`
- Auth & session: `sshd`, `login`, `sudo`, `polkit`, `auditd`
- IPC: `dbus`, `dbus-daemon`
- Display managers: `Xorg`, `gdm3`, `lightdm`, `sddm`
- Network: `NetworkManager`, `wpa_supplicant`, `dhclient`

---

## Setup

```bash
git clone https://github.com/Aliipou/AgenticAIPoweredLinuxSystemManagement.git
cd AgenticAIPoweredLinuxSystemManagement
cp .env.example .env           # fill in AGENTIC_OPENAI_API_KEY
pip install -e .
agentic --help
```

Or with Docker:

```bash
docker compose up
```

---

## Testing

```bash
# Run all tests with coverage
pytest

# Coverage is enforced at 100% — CI will fail if it drops
```

447 tests. 100% line and branch coverage on all production code.

---

## Project Structure

```
src/agentic/
  cli/                   Typer CLI application
  config/
    settings.py          Pydantic settings + AGENTIC_ENVIRONMENT
  engine/                Decision engine and intent strategies
  executor/
    action_executor.py   Dispatches to runners
    command_validator.py Syntactic + semantic safety patterns
    simulation_engine.py Pre-execution effect prediction
    transaction.py       Sequential execution with rollback
    runners/             process, package, memory, systemctl
  memory/                SQLite audit store
  models/
    action.py            ActionType, ActionScope, ActionEffect,
                         ActionSimulation, ActionCandidate, ActionPlan
    capability.py        Capability enum (6 values)
    environment.py       Environment enum (PRODUCTION/STAGING/DEVELOPMENT)
  parser/                OpenAI intent classifier
  policy/
    capability_gate.py   Least-privilege enforcement
    confidence_gate.py   LLM confidence gating
    environment_gate.py  Deployment-context risk ceiling
    permissions.py       PERMISSION_MATRIX + CRITICAL_SERVICES + ENVIRONMENT_RISK_CAPS
    safety_gate.py       Risk-level enforcement + critical service escalation
  pipeline.py            End-to-end orchestrator
```

---

## License

MIT
