"""Deployment environment — constrains the default risk ceiling."""

from __future__ import annotations

import enum


class Environment(str, enum.Enum):
    PRODUCTION = "PRODUCTION"
    STAGING = "STAGING"
    DEVELOPMENT = "DEVELOPMENT"
