# Contributing to AgenticAIPoweredLinuxSystemManagement

## Setup

```bash
git clone https://github.com/Aliipou/AgenticAIPoweredLinuxSystemManagement.git
cd AgenticAIPoweredLinuxSystemManagement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
```

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

## Safety Guidelines

This project executes shell commands on the host system. Any contribution that touches the executor, policy gate, or risk scoring **must** include:
- Tests with both safe and unsafe command examples
- Clear documentation of the risk scoring logic
- A review of the impact on the policy gate

Never submit code that bypasses the policy gate or reduces risk thresholds without strong justification.

## Code Style

- Python 3.10+, type hints required on all public functions
- `ruff` for linting, `black` for formatting
- Docstrings on all public classes and methods

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `test:`, `chore:`

## Pull Requests

1. Fork the repo and create a descriptive branch
2. Write tests for new behavior
3. Run `make lint && make test`
4. Open a PR explaining the change and its safety implications
