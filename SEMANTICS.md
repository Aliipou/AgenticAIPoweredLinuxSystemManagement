# Formal Semantics — Agentic Runtime

This document defines what each component **is** and **does**. It does not describe
what components could do, should do, or might do in future versions.

---

## Core Types

### `ActionType`
A finite enumeration of 10 recognized OS operations. The set is closed.
Actions outside this set cannot be expressed and therefore cannot be executed.

```
KILL_PROCESS | SUSPEND_PROCESS | RENICE_PROCESS
APT_INSTALL  | APT_UPGRADE
DROP_CACHES  | KILL_BY_MEMORY
SYSTEMCTL_START | SYSTEMCTL_STOP | SYSTEMCTL_RESTART
```

### `ActionCandidate`
A **proposed** action — not a committed one. It carries:
- `action_type`: which of the 10 operations
- `command`: the shell string that would be executed
- `target`: the named resource (process name, service name, package)
- `effect`: declared effect metadata (optional — may be absent)
- `preconditions`: documentation strings; **not verified at runtime**
- `postconditions`: documentation strings; **not verified at runtime**
- `rollback_support`: honest classification of recoverability (see below)
- `rollback_command`: shell string for rollback; **executed only if rollback_support ≠ NONE**

`ActionCandidate` is not executable. It becomes executable only after passing all gates.

### `ActionEffect`
A **declared** (not observed) description of what an action does to system state:
- `scope`: which resource domain is affected
- `reversible`: caller's assertion that the action can be undone
- `data_loss_risk`: caller's assertion that data may be permanently lost
- `availability_impact`: caller's assertion that a service may become unavailable

These are assertions by the action author. The runtime does not verify them.
When absent (`effect is None`), the `SimulationEngine` derives predictions from
static lookup tables.

### `RollbackSupport`
An honest classification of what can be recovered after execution:

| Value | Meaning | What the runtime does |
|-------|---------|----------------------|
| `FULL` | Prior state completely restorable | Rollback is attempted |
| `PARTIAL` | Rollback works but residual effects remain | Rollback is attempted; warning is emitted |
| `NONE` | No rollback path exists; execution is permanent | Rollback is **skipped** regardless of `rollback_command` |
| `UNKNOWN` | Not declared | `SimulationEngine` falls back to `_HIGH_IMPACT` heuristic |

`NONE` is the honest declaration for `KILL_PROCESS`, `KILL_BY_MEMORY`, and `APT_UPGRADE`.
The system does not pretend these can be undone.

### `Capability`
A least-privilege token. The runtime is constructed with a `frozenset[Capability]`.
An action whose `required_capability` is not in that set is denied before any policy
evaluation. The six capabilities map to disjoint action subsets — they cannot be combined
to grant more authority than intended.

### `RiskLevel`
An ordered integer scale: `SAFE(1) < LOW(2) < MEDIUM(3) < HIGH(4) < CRITICAL(5)`.
Risk is a property of `ActionType`, not of the action instance, except when a
critical-service target escalates `SYSTEMCTL_STOP`/`SYSTEMCTL_RESTART` to `CRITICAL`.

### `Environment`
The deployment context, set at runtime via `AGENTIC_ENVIRONMENT`:

| Environment | Risk ceiling | Meaning |
|------------|-------------|---------|
| `PRODUCTION` | MEDIUM | `APT_UPGRADE`, `SYSTEMCTL_STOP` require `--force` |
| `STAGING` | HIGH | mirrors default `SafetyGate` threshold |
| `DEVELOPMENT` | CRITICAL | unrestricted |

---

## Components — What They Actually Do

### `IntentParser`
Calls the OpenAI API and classifies free text into one of 7 `IntentType` values with
a confidence float in [0, 1]. Output is probabilistic. The parser can be wrong.

### `ConfidenceGate`
A threshold filter on the float from `IntentParser`:
- `< 0.70` → reject (`LowConfidenceError`)
- `0.70 ≤ x < 0.85` → approve, force `dry_run=True`
- `≥ 0.85` → full approval

The gate enforces the threshold. It does not verify that the intent classification
is semantically correct — only that the model expressed sufficient confidence.

### `EnvironmentGate`
Reads `AGENTIC_ENVIRONMENT` and denies any action whose static `RiskLevel` exceeds
the environment's ceiling. This is a hard ceiling, not a warning.

### `CapabilityGate`
Checks `ACTION_CAPABILITIES[action_type]` against the runtime's `frozenset[Capability]`.
Denies if the required capability is absent. This is a membership check — O(1), no heuristics.

### `SafetyGate`
Looks up `(RiskLevel, requires_sudo)` from `PERMISSION_MATRIX[action_type]`. If the
action targets a member of `CRITICAL_SERVICES` and `action_type` is `SYSTEMCTL_STOP`
or `SYSTEMCTL_RESTART`, the risk is escalated to `CRITICAL`. CRITICAL actions are
denied without `--force`.

### `CommandValidator`
Applies two deterministic pattern scans to `command` and `target` strings:

**Syntactic tier** — character-level patterns:
- `rm -rf /` (and variants)
- `: () { :|: & }` fork bombs
- raw disk writes (`> /dev/sda`, `dd if=... of=/dev/sd`)

**Semantic tier** — effect-based patterns:
- `find / -delete` or `find / -exec rm`
- `chmod -R [0-7]*77 /`
- `chown -R` on `/etc`, `/boot`, `/usr`, `/bin`, `/sbin`, `/lib`
- deletion of `/etc/passwd`, `/etc/shadow`, `/boot/*`

The validator operates on strings. It cannot detect obfuscated equivalents (e.g.,
base64-encoded commands, aliases, or multi-stage pipelines that assemble dangerous
commands from benign parts).

### `SimulationEngine`
**This is a static lookup table, not a runtime predictor.**

It derives `(scope, reversible, data_loss_risk, availability_impact)` from either:
1. The declared `ActionEffect` on the candidate, if present, or
2. Static dictionaries keyed on `ActionType` (`_ACTION_SCOPES`, `_HIGH_IMPACT`, `_AVAILABILITY_IMPACT`)

It does not inspect actual system state. It does not observe running processes, service
health, disk usage, or network state. Its output is deterministic given the action's
declared metadata and type — not the actual system.

Use simulation output for: operator visibility, pre-flight warnings, audit logging.
Do not use it as: an execution guard, a security control, or a proof of effect.

### `TransactionManager`
Executes `ActionCandidate` objects sequentially via `ActionExecutor`. On the first
failure, it iterates the successfully-executed actions in **reverse order** and attempts
rollback for each action where:
1. `rollback_command` is non-empty, AND
2. `rollback_support ≠ RollbackSupport.NONE`

Rollback failures are recorded in `TransactionResult.rollback_errors` but do not
propagate as exceptions. The transaction result is `success=False` regardless of whether
rollback succeeds or fails.

### `ActionExecutor`
Dispatches to type-specific runners (process, package, memory, systemctl). Before
execution, checks the command against a blocklist of protected process names and PIDs.
Protected processes cannot be targeted regardless of upstream gate approval.

---

## What the System Does Not Do

| Capability | Status | Reason |
|-----------|--------|--------|
| Verify `preconditions` at runtime | **Not implemented** | Strings are documentation only |
| Verify `postconditions` at runtime | **Not implemented** | Strings are documentation only |
| Observe actual system state in simulation | **Not implemented** | Simulation is static lookup |
| Detect obfuscated dangerous commands | **Not implemented** | Validator operates on literal strings |
| Enforce process isolation (namespaces, seccomp, cgroups) | **Not implemented** | Execution runs in host process space |
| Guarantee idempotency | **Not implemented** | No deduplication or state check before execution |
| Verify that rollback actually restored prior state | **Not implemented** | Rollback is fire-and-check-exit-code only |
