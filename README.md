<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&amp;color=gradient&amp;customColorList=12,20,24&amp;height=180&amp;section=header&amp;text=AgenticAI%20Linux%20Mgmt&amp;fontSize=38&amp;fontColor=fff&amp;animation=twinkling&amp;fontAlignY=38" />

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&amp;logo=python&amp;logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=flat&amp;logo=openai&amp;logoColor=white)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen?style=flat)](tests/)

**An intelligent, agentic AI layer that understands natural language and safely manages Linux systems.**

</div>

## What It Does

Instead of memorizing commands, you describe what you want in plain English. The system interprets your intent, evaluates risk, and either executes safely or asks for confirmation before touching anything critical.

```
You: "Clean up logs older than 30 days to free disk space"

System: Parsing intent...  CLEAN_MEMORY (confidence: 0.94)
         Risk assessment... LOW (no system files affected)
         Proposed action:   find /var/log -mtime +30 -name "*.log" -delete
         Executing...       Freed 2.3 GB across 847 files
         Logged to audit trail.
```

## Architecture

The system processes every request through a 6-layer pipeline:

```
Input
  |
  v
[Parser]        NLP classification via GPT-4 (FOCUS / UPDATE / CLEAN_MEMORY / UNKNOWN)
  |
  v
[Engine]        Strategy pattern — selects execution plan based on intent
  |
  v
[Policy Gate]   Risk scoring — blocks destructive actions, flags uncertain ones
  |
  v
[Executor]      Runs approved shell commands with timeout and sandboxing
  |
  v
[Memory]        SQLite audit log — every action, decision, and outcome recorded
  |
  v
Output + Explanation
```

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Parser | `parser/intent_parser.py` | OpenAI intent classification |
| Engine | `engine/decision_engine.py` | Strategy-based plan selection |
| Policy Gate | `policy/risk_gate.py` | Risk scoring and approval |
| Executor | `executor/runner.py` | Sandboxed shell execution |
| Memory | `memory/audit_log.py` | SQLite persistence |

## Quick Start

```bash
git clone https://github.com/Aliipou/AgenticAIPoweredLinuxSystemManagement.git
cd AgenticAIPoweredLinuxSystemManagement
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
python main.py
```

## Supported Intent Types

| Intent | Example Request |
|--------|----------------|
| `FOCUS` | "Show me top CPU consumers" |
| `UPDATE` | "Update all outdated packages" |
| `CLEAN_MEMORY` | "Free up disk space safely" |
| `UNKNOWN` | Asks for clarification |

## Safety Design

Every action passes through the policy gate before execution. High-risk operations (removing system files, modifying boot config, network changes) are blocked by default and require explicit override. All decisions are logged with timestamp, reasoning, and outcome.

## License

MIT
