"""Risk classification helpers."""

from __future__ import annotations

from agentic.models.policy import RiskLevel


def risk_from_string(value: str) -> RiskLevel:
    mapping = {
        "SAFE": RiskLevel.SAFE,
        "LOW": RiskLevel.LOW,
        "MEDIUM": RiskLevel.MEDIUM,
        "HIGH": RiskLevel.HIGH,
        "CRITICAL": RiskLevel.CRITICAL,
    }
    result = mapping.get(value.upper())
    if result is None:
        raise ValueError(f"Unknown risk level: {value}")
    return result


def is_above_threshold(level: RiskLevel, threshold: RiskLevel) -> bool:
    return level.value > threshold.value


def requires_user_confirmation(level: RiskLevel) -> bool:
    return level.value >= RiskLevel.MEDIUM.value
