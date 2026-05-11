# Runtime Invariants

This document lists invariants that are **actually enforced at runtime** — by code,
not by documentation or convention. Each entry names the enforcement mechanism and
the file where it lives.

An invariant not listed here is either a design intention, a documentation claim,
or a test assertion. It is not an enforcement guarantee.

---

## Enforced Invariants

### INV-01: Protected processes cannot be targeted
**What:** A fixed set of process names (init, systemd, sshd, login, dbus, etc.) cannot
be killed or suspended regardless of policy approval.
**Mechanism:** Blocklist check in `ActionExecutor.execute()` before dispatch.
**File:** `src/agentic/executor/action_executor.py`
**Bypass:** No override path exists. The check runs after all gates.

---

### INV-02: `PRODUCTION` environment cannot execute HIGH or CRITICAL actions
**What:** `AGENTIC_ENVIRONMENT=PRODUCTION` caps actionable risk at MEDIUM.
**Mechanism:** `EnvironmentGate.evaluate()` denies any action with `RiskLevel > MEDIUM`.
**File:** `src/agentic/policy/environment_gate.py`, `src/agentic/policy/permissions.py`
**Bypass:** None within the gate chain. Requires constructing a pipeline without an `EnvironmentGate`.

---

### INV-03: Actions above the configured risk threshold are denied
**What:** `SafetyGate` rejects any action whose static risk exceeds `max_risk_level`.
**Mechanism:** Integer comparison of `RiskLevel` values in `SafetyGate.evaluate()`.
**File:** `src/agentic/policy/safety_gate.py`
**Bypass:** `--force` flag bypasses the CRITICAL block for operators who explicitly override.

---

### INV-04: Critical service STOP/RESTART escalates to CRITICAL risk
**What:** `SYSTEMCTL_STOP` or `SYSTEMCTL_RESTART` targeting any service in `CRITICAL_SERVICES`
is treated as CRITICAL regardless of the base matrix entry.
**Mechanism:** Target name lookup in `CRITICAL_SERVICES` frozenset (case-insensitive) in
`SafetyGate.evaluate()`.
**File:** `src/agentic/policy/safety_gate.py`, `src/agentic/policy/permissions.py`
**Bypass:** `--force` flag. Escalation itself cannot be bypassed — only the block after escalation.

---

### INV-05: Actions without the required capability are denied
**What:** Each of the 10 `ActionType` values maps to exactly one `Capability`. An action
cannot execute if the runtime's `frozenset[Capability]` does not contain that capability.
**Mechanism:** `CapabilityGate.evaluate()` performs a frozenset membership check.
**File:** `src/agentic/policy/capability_gate.py`
**Bypass:** None within the gate. Requires constructing a `CapabilityGate` with a broader
frozenset at startup.

---

### INV-06: Low-confidence intents cannot execute live actions
**What:** LLM confidence < 0.70 produces a rejection. Confidence in [0.70, 0.85) forces
`dry_run=True`.
**Mechanism:** `ConfidenceGate.evaluate()` raises `LowConfidenceError` or sets the
dry-run flag.
**File:** `src/agentic/policy/confidence_gate.py`
**Bypass:** None within the gate. Confidence thresholds are hard-coded constants.

---

### INV-07: Syntactically and semantically dangerous commands are unconditionally blocked
**What:** Commands matching known destructive patterns (`rm -rf /`, fork bombs, raw disk
writes, `find / -delete`, `chmod -R 777 /`, etc.) are rejected regardless of what
upstream gates approved.
**Mechanism:** `CommandValidator.validate()` applies regex patterns to `command` and
`target` strings. Returns `CommandValidationError` on match.
**File:** `src/agentic/executor/command_validator.py`
**Bypass:** None. The validator runs after all policy gates.

---

### INV-08: `RollbackSupport.NONE` actions are never rolled back
**What:** Actions declared with `RollbackSupport.NONE` are skipped during rollback
even when a `rollback_command` is set.
**Mechanism:** Explicit guard in `TransactionManager._rollback()`.
**File:** `src/agentic/executor/transaction.py`
**Bypass:** None. The guard runs unconditionally.

---

### INV-09: Every decision is recorded in the audit log
**What:** Every gate decision (approved or denied), every simulation, and every
execution result is written to the SQLite audit store before the pipeline returns.
**Mechanism:** `MemoryStore.record_*()` calls in `Pipeline.run()`.
**File:** `src/agentic/pipeline.py`, `src/agentic/memory/store.py`
**Bypass:** Bypassing the pipeline bypasses the audit log.

---

## Declared But Not Enforced

The following are documented in code but **not verified at runtime**:

| Declaration | Location | What it is | What it is NOT |
|------------|----------|-----------|----------------|
| `preconditions: list[str]` | `ActionCandidate` | Documentation strings | Verified at runtime |
| `postconditions: list[str]` | `ActionCandidate` | Documentation strings | Verified at runtime |
| `ActionEffect.reversible` | `ActionEffect` | Caller assertion | Checked against actual state |
| `ActionEffect.data_loss_risk` | `ActionEffect` | Caller assertion | Verified before execution |
| Simulation scope/reversibility | `ActionSimulation` | Static prediction | Observed system state |
| `ROLLBACK_CAPABILITIES` table | `permissions.py` | Design truth table | Used by executor at runtime |

`ROLLBACK_CAPABILITIES` documents the correct rollback classification for each
`ActionType`. Individual `ActionCandidate` instances carry their own `rollback_support`
field; the table is not automatically applied — it exists as the authoritative reference.
