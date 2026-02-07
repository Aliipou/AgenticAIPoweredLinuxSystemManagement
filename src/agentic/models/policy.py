"""Policy models — output of the safety gate."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class RiskLevel(int, enum.Enum):
    SAFE = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class PolicyDecision(BaseModel):
    action_id: str
    risk_level: RiskLevel
    approved: bool
    requires_sudo: bool = False
    reason: str = ""
    requires_confirmation: bool = Field(default=False)
