"""Capability enum — least-privilege action classification."""

from __future__ import annotations

import enum


class Capability(str, enum.Enum):
    KILL_PROCESS = "KILL_PROCESS"
    SUSPEND_PROCESS = "SUSPEND_PROCESS"
    RENICE_PROCESS = "RENICE_PROCESS"
    PACKAGE_MANAGEMENT = "PACKAGE_MANAGEMENT"
    MEMORY_MANAGEMENT = "MEMORY_MANAGEMENT"
    SERVICE_MANAGEMENT = "SERVICE_MANAGEMENT"
