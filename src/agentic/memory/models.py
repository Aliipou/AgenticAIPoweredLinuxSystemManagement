"""Pydantic models for database records."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class RequestRecord(BaseModel):
    id: str
    raw_query: str
    intent_type: str
    confidence: float
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ActionRecord(BaseModel):
    id: str
    request_id: str
    action_type: str
    description: str
    command: str = ""
    risk_level: int = 1
    approved: bool = False


class ExecutionRecord(BaseModel):
    id: str
    action_id: str
    success: bool
    output: str = ""
    error: str = ""
    rolled_back: bool = False
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
