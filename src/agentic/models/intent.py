"""Intent models — output of the NLP parser."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class IntentType(str, enum.Enum):
    FOCUS = "FOCUS"
    UPDATE = "UPDATE"
    CLEAN_MEMORY = "CLEAN_MEMORY"
    UNKNOWN = "UNKNOWN"


class Entity(BaseModel):
    name: str
    value: str
    source: str = ""


class ParsedIntent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    raw_query: str
    intent_type: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: list[Entity] = Field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
