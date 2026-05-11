"""Brutal tests for exception hierarchy."""

from __future__ import annotations

import pytest

from agentic.exceptions import (
    AgenticError,
    ExecutionError,
    LowConfidenceError,
    ParseError,
    PolicyDeniedError,
    UnsafeCommandError,
    UserCancelledError,
)


class TestExceptionHierarchy:
    def test_agentic_error_is_exception(self):
        assert issubclass(AgenticError, Exception)

    def test_parse_error_inherits(self):
        assert issubclass(ParseError, AgenticError)

    def test_policy_denied_inherits(self):
        assert issubclass(PolicyDeniedError, AgenticError)

    def test_execution_error_inherits(self):
        assert issubclass(ExecutionError, AgenticError)

    def test_user_cancelled_inherits(self):
        assert issubclass(UserCancelledError, AgenticError)


class TestExceptionInstantiation:
    def test_agentic_error(self):
        err = AgenticError("base error")
        assert str(err) == "base error"

    def test_parse_error(self):
        err = ParseError("bad parse")
        assert str(err) == "bad parse"

    def test_policy_denied(self):
        err = PolicyDeniedError("denied!")
        assert str(err) == "denied!"

    def test_execution_error_with_action_id(self):
        err = ExecutionError("failed", action_id="act-123")
        assert str(err) == "failed"
        assert err.action_id == "act-123"

    def test_execution_error_default_action_id(self):
        err = ExecutionError("failed")
        assert err.action_id == ""

    def test_user_cancelled(self):
        err = UserCancelledError("cancelled")
        assert str(err) == "cancelled"

    def test_catch_specific_as_agentic(self):
        with pytest.raises(AgenticError):
            raise ParseError("test")

    def test_catch_execution_as_agentic(self):
        with pytest.raises(AgenticError):
            raise ExecutionError("test", action_id="x")

    def test_low_confidence_inherits(self):
        assert issubclass(LowConfidenceError, AgenticError)

    def test_unsafe_command_inherits(self):
        assert issubclass(UnsafeCommandError, AgenticError)


class TestNewExceptions:
    def test_low_confidence_error(self):
        err = LowConfidenceError("confidence too low")
        assert str(err) == "confidence too low"

    def test_low_confidence_as_agentic(self):
        with pytest.raises(AgenticError):
            raise LowConfidenceError("low")

    def test_unsafe_command_error_with_action_id(self):
        err = UnsafeCommandError("rm -rf / detected", action_id="act-99")
        assert str(err) == "rm -rf / detected"
        assert err.action_id == "act-99"

    def test_unsafe_command_error_default_action_id(self):
        err = UnsafeCommandError("dangerous")
        assert err.action_id == ""

    def test_unsafe_command_as_agentic(self):
        with pytest.raises(AgenticError):
            raise UnsafeCommandError("unsafe", action_id="x")
