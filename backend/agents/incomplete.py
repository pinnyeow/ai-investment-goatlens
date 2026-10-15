"""Fail-closed result helpers shared by investor agents."""

from typing import Any, Dict, Iterable, Optional


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
        metric for metric in required_metrics if financials.get(metric) is None
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
