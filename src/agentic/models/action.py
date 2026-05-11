"""Action models — output of the decision engine."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ActionType(str, enum.Enum):
    KILL_PROCESS = "KILL_PROCESS"
    SUSPEND_PROCESS = "SUSPEND_PROCESS"
    RENICE_PROCESS = "RENICE_PROCESS"
    APT_INSTALL = "APT_INSTALL"
    APT_UPGRADE = "APT_UPGRADE"
    DROP_CACHES = "DROP_CACHES"
    KILL_BY_MEMORY = "KILL_BY_MEMORY"
    SYSTEMCTL_START = "SYSTEMCTL_START"
    SYSTEMCTL_STOP = "SYSTEMCTL_STOP"
    SYSTEMCTL_RESTART = "SYSTEMCTL_RESTART"


class ActionScope(str, enum.Enum):
    PROCESS = "PROCESS"
    FILESYSTEM = "FILESYSTEM"
    PACKAGE = "PACKAGE"
    SERVICE = "SERVICE"
    MEMORY = "MEMORY"
    NETWORK = "NETWORK"
    SYSTEM = "SYSTEM"


class ActionEffect(BaseModel):
    """Formal description of what an action does to system state."""

    scope: ActionScope
    reversible: bool = True
    data_loss_risk: bool = False
    availability_impact: bool = False


class ActionCandidate(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    action_type: ActionType
    description: str
    command: str = ""
    target: str = ""
    parameters: dict[str, str] = Field(default_factory=dict)
    rollback_command: str = ""
    effect: ActionEffect | None = None


class ActionPlan(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    intent_id: str
    actions: list[ActionCandidate] = Field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ActionResult(BaseModel):
    action_id: str
    success: bool
    output: str = ""
    error: str = ""
    rolled_back: bool = False
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
