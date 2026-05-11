<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-346%20passing-brightgreen?style=flat)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen?style=flat)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

# Policy-Constrained Autonomous Operations Runtime

**A policy-gated autonomous agent for Linux system management.**  
Not an AI assistant. Not a chatbot. A runtime with formal authority boundaries.

</div>

---

## What Makes This Different

Most "agentic" systems are LLM wrappers over shell — they parse a request and run a command. This system enforces three independent safety gates before any command reaches the OS:

1. **Confidence Gate** — if the LLM isn't certain enough, the action is refused or downgraded to dry-run. Confidence ≠ correctness, so both are tracked.
2. **Policy Gate** — every action type carries a static risk level and sudo requirement. Actions above the configured threshold are blocked regardless of confidence. Stopping or restarting a critical service (PostgreSQL, nginx, Docker, sshd, etc.) auto-escalates to CRITICAL, blocking without `--force`.
3. **Command Validator** — two-tier deterministic scan of the generated command and target strings. The syntactic tier catches `rm -rf /`, fork bombs, raw disk writes. The semantic tier catches commands that are safe-looking in isolation but catastrophic by effect: `find / -delete`, `chmod -R 777 /`, `chown -R` on `/etc`, deletion of `/etc/passwd` or `/boot/*`. All patterns are blocked unconditionally, regardless of policy approval.

If all three gates pass, the executor runs the action. Every decision — approved or denied — is recorded in the audit log.

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
[SafetyGate]         Per-action risk scoring against PERMISSION_MATRIX
                     CRITICAL actions blocked; HIGH requires confirmation
    │
    ▼
[CommandValidator]   Deterministic pattern scan — rejects dangerous strings
                     regardless of policy approval
    │
    ▼
[ActionExecutor]     Runs approved actions (or simulates in dry-run mode)
    │
    ▼
[MemoryStore]        Full audit trail: request → decisions → results
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

346 tests. 100% line and branch coverage on all production code.

---

## Project Structure

```
src/agentic/
  cli/             Typer CLI application
  config/          Pydantic settings
  engine/          Decision engine and intent strategies
  executor/        Action runners (process, package, memory, systemctl)
    command_validator.py   ← deterministic safety check
  memory/          SQLite audit store
  models/          Pydantic data models
  parser/          OpenAI intent classifier
  models/
    action.py              ← ActionType, ActionScope, ActionEffect, ActionCandidate
  policy/
    confidence_gate.py     ← LLM confidence gating
    permissions.py         ← PERMISSION_MATRIX + CRITICAL_SERVICES
    safety_gate.py         ← risk-level enforcement + critical service escalation
  pipeline.py      End-to-end orchestrator
```

---

## License

MIT
