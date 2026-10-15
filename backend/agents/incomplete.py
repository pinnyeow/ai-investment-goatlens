"""Fail-closed result helpers shared by investor agents."""

import math
from typing import Any, Dict, Iterable, Optional


def metric_is_available(value: Any) -> bool:
    """Only finite, non-boolean numeric values satisfy strategy inputs."""
    return type(value) in (int, float) and math.isfinite(value)


def incomplete_if_missing(
    *,
    agent: str,
    style: str,
    ticker: str,
    financials: Dict[str, Any],
    required_metrics: Iterable[str],
) -> Optional[Dict[str, Any]]:
    """Return an explicit incomplete result when required inputs are absent."""
    unavailable = [
        metric
        for metric in required_metrics
        if not metric_is_available(financials.get(metric))
    ]
    if not unavailable:
        return None
    return {
        "agent": agent,
        "style": style,
        "ticker": ticker,
        "score": None,
        "verdict": None,
        "status": "incomplete",
        "unavailable_metrics": unavailable,
        "metrics": {},
        "insights": [],
        "concerns": ["Required financial inputs are unavailable."],
    }
