# Agentic — AI-Powered Linux System Management

An intelligent system management layer that understands natural language, proposes safe OS actions, and logs everything for audit.

## Architecture

```
User Query → Parser → Engine → Policy Gate → Executor → Memory
              (NLP)   (Strategy)  (Risk)      (Runner)   (SQLite)
```

**6-layer pipeline:**

| Layer | Module | Role |
|-------|--------|------|
| **Parser** | `parser/intent_parser.py` | OpenAI-based NLP classification into FOCUS/UPDATE/CLEAN_MEMORY/UNKNOWN |
| **Engine** | `engine/decision_engine.py` | Strategy pattern — maps intents to concrete action plans |
| **Policy** | `policy/safety_gate.py` | Risk evaluation, permission matrix, confirmation gating |
| **Executor** | `executor/action_executor.py` | Runner pattern — dispatches to process/package/memory/systemctl runners |
| **Memory** | `memory/store.py` | SQLite audit log with full request lifecycle tracking |
| **CLI** | `cli/app.py` | Typer + Rich terminal interface |

## Quick Start

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your OpenAI API key

# Use
agentic ask "help me focus by closing distracting apps"
agentic ask "update all system packages" --dry-run
agentic ask "free up some RAM" --force
agentic status
agentic history
agentic config
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `agentic ask "query"` | Parse, plan, confirm, and execute |
| `agentic ask "query" --dry-run` | Simulate without executing |
| `agentic ask "query" --force` | Skip confirmation prompts |
| `agentic history` | Show recent action history |
| `agentic history -n 5` | Limit history results |
| `agentic rollback <id>` | Rollback a previous action |
| `agentic status` | Show CPU/memory/top processes |
| `agentic config` | Show current settings |

## Intent Types

| Intent | Description | Example Queries |
|--------|-------------|-----------------|
| `FOCUS` | Suspend distracting processes | "help me focus", "close chrome and slack" |
| `UPDATE` | Install/upgrade packages | "update my system", "install vim" |
| `CLEAN_MEMORY` | Free RAM, kill memory hogs | "free up memory", "kill chrome to save RAM" |
| `UNKNOWN` | Unrecognized requests | "tell me a joke" |

## Risk Levels

| Level | Value | Confirmation | Example Actions |
|-------|-------|-------------|-----------------|
| SAFE | 1 | No | — |
| LOW | 2 | No | Suspend process, renice |
| MEDIUM | 3 | Yes | Kill process, apt install, drop caches |
| HIGH | 4 | Yes | apt upgrade, kill by memory, systemctl stop |
| CRITICAL | 5 | Blocked | Requires `--force` |

## Configuration

All settings use the `AGENTIC_` env prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTIC_OPENAI_API_KEY` | (required) | OpenAI API key |
| `AGENTIC_OPENAI_MODEL` | `gpt-4o` | Model for intent classification |
| `AGENTIC_DB_PATH` | `~/.agentic/history.db` | SQLite database location |
| `AGENTIC_DRY_RUN` | `false` | Global dry-run mode |
| `AGENTIC_LOG_LEVEL` | `INFO` | Logging verbosity |
| `AGENTIC_MAX_RISK_LEVEL` | `HIGH` | Maximum allowed risk level |
| `AGENTIC_REQUIRE_CONFIRMATION` | `true` | Prompt for MEDIUM+ risk |

## Development

```bash
# Install dev dependencies
pip install -e .
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Run tests (100% coverage enforced)
pytest

# Run with verbose coverage report
pytest --cov=agentic --cov-report=term-missing --cov-branch
```

## Project Structure

```
src/agentic/
├── main.py                  # Entry point, dependency wiring
├── pipeline.py              # 6-layer orchestrator
├── exceptions.py            # Error hierarchy
├── config/settings.py       # Pydantic Settings
├── models/                  # Pydantic data models
│   ├── intent.py            # IntentType, Entity, ParsedIntent
│   ├── action.py            # ActionType, ActionCandidate, ActionPlan, ActionResult
│   └── policy.py            # RiskLevel, PolicyDecision
├── cli/                     # Typer + Rich interface
│   ├── app.py               # Commands: ask, history, rollback, status, config
│   ├── output.py            # Rich display helpers
│   └── prompts.py           # Confirmation dialogs
├── parser/                  # OpenAI NLP classification
│   ├── intent_parser.py     # Async parse() with structured output
│   ├── schemas.py           # JSON schema for OpenAI
│   └── prompt_templates.py  # System/user prompts
├── engine/                  # Decision engine + strategies
│   ├── decision_engine.py   # Intent → ActionPlan
│   ├── action_registry.py   # IntentType → Strategy mapping
│   └── strategies/          # FOCUS, UPDATE, CLEAN_MEMORY
├── policy/                  # Safety gate
│   ├── safety_gate.py       # Risk evaluation + filtering
│   ├── permissions.py       # ActionType → (RiskLevel, sudo) matrix
│   └── risk_levels.py       # Risk helpers
├── executor/                # Action execution
│   ├── action_executor.py   # Dispatch to runners
│   └── runners/             # process, package, memory, systemctl
└── memory/                  # Audit logging
    ├── store.py             # SQLite CRUD
    ├── models.py            # DB record models
    ├── context.py           # History retrieval
    └── migrations.py        # Schema definitions
```

## Design Decisions

- **Async internally, sync CLI boundary** — All layers use `async def`, Typer commands wrap with `asyncio.run()`
- **OpenAI structured output** — JSON schema for deterministic parsing
- **Strategy pattern** — Each IntentType maps to an IntentStrategy subclass
- **Runner pattern** — Each ActionType maps to a BaseRunner subclass
- **Dry-run first-class** — Threaded through entire pipeline
- **No action without confirmation** — MEDIUM+ risk always prompts user
- **Full audit trail** — Every request, action, policy decision, and result logged to SQLite
