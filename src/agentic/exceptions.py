"""Custom exception hierarchy for Agentic."""

from __future__ import annotations


class AgenticError(Exception):
    """Base exception for all Agentic errors."""


class ParseError(AgenticError):
    """Raised when intent parsing fails."""


class PolicyDeniedError(AgenticError):
    """Raised when the safety gate blocks an action."""


class ExecutionError(AgenticError):
    """Raised when action execution fails."""

    def __init__(self, message: str, action_id: str = "") -> None:
        super().__init__(message)
        self.action_id = action_id


class UserCancelledError(AgenticError):
    """Raised when the user cancels a confirmation prompt."""
