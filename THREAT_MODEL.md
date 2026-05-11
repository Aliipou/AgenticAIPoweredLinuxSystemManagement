# Threat Model

This document defines what this system is designed to prevent, what it is not
designed to prevent, and where the trust boundaries lie.

---

## What This System Is

A **policy-gated execution runtime** for a predefined set of Linux system operations.
It is not a general-purpose sandbox, an IPC firewall, or a kernel security module.
Its security surface is the pipeline from LLM intent to shell command execution.

---

## Trust Boundaries

```
┌─────────────────────────────────────────────────────────┐
│  UNTRUSTED: LLM output (IntentParser, DecisionEngine)  │
│  Reason: probabilistic, can hallucinate intent/commands │
└───────────────────────────┬─────────────────────────────┘
                            │ gates filter before execution
┌───────────────────────────▼─────────────────────────────┐
│  TRUSTED: Gate chain (Confidence → Environment →        │
│           Capability → Safety → CommandValidator)       │
│  Reason: deterministic code; covered by test suite      │
└───────────────────────────┬─────────────────────────────┘
                            │ approved actions only
┌───────────────────────────▼─────────────────────────────┐
│  TRUSTED: ActionExecutor + runners                      │
│  Reason: dispatches only approved, validated commands   │
└─────────────────────────────────────────────────────────┘
```

The operator who constructs the `Pipeline` instance determines the trust level.
A pipeline with all capabilities granted, `max_risk_level="CRITICAL"`, `force=True`,
and no `EnvironmentGate` has no meaningful restrictions.

---

## Threats Mitigated

### T1 — LLM hallucination of destructive commands
**Threat:** The LLM generates a dangerous command string (e.g., `rm -rf /`, fork bomb).
**Mitigation:** `CommandValidator` applies deterministic pattern matching unconditionally
after all other gates. A matched pattern is a hard rejection regardless of upstream approval.
**Residual risk:** Obfuscated equivalents (base64, aliased commands, multi-stage pipelines)
bypass the validator.

---

### T2 — LLM low-confidence intent classification
**Threat:** The LLM is uncertain but the system executes anyway.
**Mitigation:** `ConfidenceGate` refuses execution below 0.70 and forces dry-run below 0.85.
**Residual risk:** A high-confidence wrong classification still reaches execution.

---

### T3 — Exceeding deployment authority
**Threat:** A high-risk operation runs in a production environment where it should not.
**Mitigation:** `EnvironmentGate` enforces a hard risk ceiling per environment.
**Residual risk:** The environment is set via an environment variable; a misconfigured
deployment sets the wrong environment.

---

### T4 — Accidental privilege escalation
**Threat:** An action type acquires more capability than intended through configuration.
**Mitigation:** `CapabilityGate` uses a closed `frozenset[Capability]` set at construction
time. Each action type maps to exactly one capability; there is no wildcard.
**Residual risk:** A `frozenset` with all capabilities granted has no restriction.

---

### T5 — Targeting critical infrastructure services
**Threat:** The LLM instructs stopping or restarting a database, web server, or
security-critical service.
**Mitigation:** `CRITICAL_SERVICES` frozenset escalates `SYSTEMCTL_STOP`/`SYSTEMCTL_RESTART`
to CRITICAL risk, which is blocked without explicit `--force`.
**Residual risk:** Services not in `CRITICAL_SERVICES` are not escalated. The list is
static and must be kept up to date.

---

### T6 — Cascading failure leaving system in partial state
**Threat:** A multi-action plan partially executes and fails mid-way, leaving the system
in an inconsistent state.
**Mitigation:** `TransactionManager` executes sequentially and rolls back completed actions
in reverse order on failure.
**Residual risk:** `RollbackSupport.NONE` actions cannot be rolled back. `PARTIAL` rollbacks
leave residual effects. The system is not atomic in the database sense.

---

### T7 — Targeting essential OS processes
**Threat:** The LLM targets `init`, `systemd`, `sshd`, or other processes whose termination
would break the host.
**Mitigation:** `ActionExecutor` maintains a static blocklist that rejects these targets
unconditionally after all gate approvals.
**Residual risk:** The blocklist is static; a critical process not in the list is not protected.

---

## Threats NOT Mitigated

### N1 — Malicious operator
A human operator who constructs the pipeline with all capabilities, no environment gate,
and `force=True` faces no restrictions from this system. The runtime does not protect
against its own authorized users.

### N2 — Process isolation
Commands execute in the host process space. There are no namespaces, cgroups, seccomp
filters, or capability dropping applied at execution time. An approved command has full
OS access within the executor's privilege level.

### N3 — Semantic intent bypass
An LLM that expresses a harmful intent through a sequence of individually-approved
low-risk actions is not detected. The gates evaluate each action independently.

### N4 — Obfuscated commands
The `CommandValidator` operates on literal command strings. A command that assembles
a dangerous operation from benign-looking parts (variable expansion, heredocs, eval,
encoded payloads) bypasses the validator.

### N5 — Side-channel effects of approved actions
An approved `APT_INSTALL` could install a package with malicious post-install scripts.
An approved `RENICE_PROCESS` on the right PID could degrade a security monitor.
The gates evaluate declared risk level, not second-order effects.

### N6 — Audit log integrity
The SQLite audit log has no cryptographic integrity protection. A compromised host
can modify or delete audit records without detection.

### N7 — Race conditions
There is no locking on process targets. A process can change state (exit, fork, change
PID) between gate evaluation and command execution. The system does not verify that the
target observed at planning time still exists at execution time.

---

## Security Properties That Hold

| Property | Enforcement | Strength |
|---------|-------------|---------|
| No high-risk action in PRODUCTION | `EnvironmentGate` | Hard ceiling; bypassed only by not using the gate |
| No execution without capability token | `CapabilityGate` | frozenset membership; no wildcard |
| Known destructive patterns blocked | `CommandValidator` | Deterministic regex; no ML |
| Essential processes unreachable | `ActionExecutor` blocklist | Static list; runs last |
| Low-confidence intents forced to dry-run | `ConfidenceGate` | Numeric threshold; hard-coded |
| NONE rollback never attempted | `TransactionManager` | Explicit guard; no bypass |

## Security Properties That Do Not Hold

| Claimed property | Reality |
|-----------------|---------|
| "Predicts action effects" | Static lookup table, not runtime observation |
| "Preconditions verified" | Documentation strings only |
| "Rollback restores prior state" | Exit-code check only; state not verified |
| "Isolated execution" | Runs in host process space |
