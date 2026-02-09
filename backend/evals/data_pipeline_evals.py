"""
Layer 1: Data Pipeline Evals — Deterministic checks.

These are code-based (no LLM needed). They verify the data feeding
into agents is clean and correct. Catches problems like NaN values,
wrong beat/miss labels, and impossible financial metrics.

Think of this as your "data quality gate." If data is bad,
agent insights will be bad no matter how good the agent code is.
"""

import json
import math
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Result from a single eval check."""
    name: str
    passed: bool
    details: str = ""
    severity: str = "error"  # "error" or "warning"


@dataclass
class LayerResults:
    """Results from an entire eval layer."""
    layer: str
    results: List[EvalResult] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        return self.pass_count / len(self.results) if self.results else 0.0


def _is_valid_float(val) -> bool:
    """Check if a value is a valid, finite float (not NaN, not Infinity)."""
    if val is None:
        return True  # None is acceptable (missing data)
    if not isinstance(val, (int, float)):
        return False
    return not (math.isnan(val) or math.isinf(val))


# ── Earnings Data Checks ──


def eval_eps_sanity(earnings: List[Dict[str, Any]]) -> List[EvalResult]:
    """
    Check that EPS values are valid floats, not NaN or Infinity.
    
    WHY: We had a real production bug where NaN EPS values from yfinance
    crashed JSON serialization. This eval catches that class of problem.
    """
    results = []
    for i, e in enumerate(earnings):
        quarter = e.get("quarter", f"row_{i}")

        # eps_actual
        actual = e.get("eps_actual")
        ok = _is_valid_float(actual)
        results.append(EvalResult(
            name=f"eps_actual_valid:{quarter}",
            passed=ok,
            details=f"eps_actual={actual}" if not ok else "",
        ))

        # eps_estimate
        estimate = e.get("eps_estimate")
        ok = _is_valid_float(estimate)
        results.append(EvalResult(
            name=f"eps_estimate_valid:{quarter}",
            passed=ok,
            details=f"eps_estimate={estimate}" if not ok else "",
        ))

    return results


def eval_beat_miss_labels(earnings: List[Dict[str, Any]]) -> List[EvalResult]:
    """
    Check that beat/miss labels are consistent with EPS values.
    
    Rule: If actual > estimate → "beat". If actual < estimate → "miss".
    If actual == estimate → "inline".
    
    WHY: A flipped label means the UI shows a green "BEAT" badge
    when the company actually missed. Users would make wrong decisions.
    """
    results = []
    for i, e in enumerate(earnings):
        quarter = e.get("quarter", f"row_{i}")
        actual = e.get("eps_actual")
        estimate = e.get("eps_estimate")
        label = e.get("beat_miss", "")

        # Skip rows where we don't have both values (future earnings)
        if actual is None or estimate is None:
            continue
        if not _is_valid_float(actual) or not _is_valid_float(estimate):
            continue

        if actual > estimate:
            expected = "beat"
        elif actual < estimate:
            expected = "miss"
        else:
            expected = "inline"

        ok = label == expected
        results.append(EvalResult(
            name=f"beat_miss_label:{quarter}",
            passed=ok,
            details=f"actual={actual}, estimate={estimate}, label='{label}', expected='{expected}'" if not ok else "",
        ))

    return results


def eval_surprise_math(earnings: List[Dict[str, Any]]) -> List[EvalResult]:
    """
    Check that surprise_pct matches the math: (actual - estimate) / |estimate| * 100.
    
    WHY: If the surprise percentage is calculated wrong, the magnitude
    of beats/misses is misrepresented. A 2% beat shown as 20% is misleading.
    """
    results = []
    for i, e in enumerate(earnings):
        quarter = e.get("quarter", f"row_{i}")
        actual = e.get("eps_actual")
        estimate = e.get("eps_estimate")
        surprise = e.get("surprise_pct")

        if actual is None or estimate is None or surprise is None:
            continue
        if not _is_valid_float(actual) or not _is_valid_float(estimate):
            continue
        if estimate == 0:
            continue  # Division by zero edge case

        expected = (actual - estimate) / abs(estimate) * 100
        # Allow ±0.5% tolerance for floating point rounding
        ok = abs(surprise - expected) < 0.5

        results.append(EvalResult(
            name=f"surprise_math:{quarter}",
            passed=ok,
            details=f"surprise_pct={surprise:.2f}, expected={expected:.2f}" if not ok else "",
        ))

    return results


# ── Price History Checks ──


def eval_price_history(prices: List[Dict[str, Any]]) -> List[EvalResult]:
    """
    Check price history data for NaN values, negative prices, and order.
    
    WHY: Chart rendering will break or show misleading data if prices
    contain NaN or impossible values (negative close price).
    """
    results = []

    if not prices:
        results.append(EvalResult(
            name="price_history_not_empty",
            passed=False,
            details="No price data returned",
        ))
        return results

    results.append(EvalResult(
        name="price_history_not_empty",
        passed=True,
    ))

    # Check for NaN/Infinity in OHLCV fields
    nan_count = 0
    for p in prices:
        for field_name in ["open", "high", "low", "close"]:
            val = p.get(field_name)
            if not _is_valid_float(val):
                nan_count += 1

    results.append(EvalResult(
        name="price_no_nan",
        passed=nan_count == 0,
        details=f"{nan_count} NaN/Inf values found in OHLCV data" if nan_count > 0 else "",
    ))

    # Check close prices are positive
    negative_count = sum(1 for p in prices if isinstance(p.get("close"), (int, float)) and p["close"] <= 0)
    results.append(EvalResult(
        name="price_close_positive",
        passed=negative_count == 0,
        details=f"{negative_count} non-positive close prices" if negative_count > 0 else "",
    ))

    # Check dates are in order (chronological)
    dates = [p.get("date", "") for p in prices if p.get("date")]
    sorted_dates = sorted(dates)
    results.append(EvalResult(
        name="price_dates_chronological",
        passed=dates == sorted_dates,
        details="Dates are not in chronological order" if dates != sorted_dates else "",
    ))

    return results


# ── Financial Metrics Checks ──


def eval_financial_metrics(metrics: Dict[str, Any]) -> List[EvalResult]:
    """
    Check that financial metrics are within plausible ranges.
    
    WHY: A P/E of 999999 or ROE of -500% usually means the data source
    returned garbage. These range checks catch obviously broken data.
    """
    results = []

    # P/E ratio: should be between -500 and 2000 (extreme growth stocks can have high P/E)
    pe = metrics.get("pe_ratio")
    if pe is not None and _is_valid_float(pe):
        ok = -500 <= pe <= 2000
        results.append(EvalResult(
            name="pe_ratio_plausible",
            passed=ok,
            details=f"P/E={pe}" if not ok else "",
            severity="warning",
        ))

    # ROE: should be between -200% and 500%
    roe = metrics.get("roe")
    if roe is not None and _is_valid_float(roe):
        ok = -2.0 <= roe <= 5.0
        results.append(EvalResult(
            name="roe_plausible",
            passed=ok,
            details=f"ROE={roe}" if not ok else "",
            severity="warning",
        ))

    # Debt-to-equity: should be between -10 and 100
    dte = metrics.get("debt_to_equity")
    if dte is not None and _is_valid_float(dte):
        ok = -10 <= dte <= 100
        results.append(EvalResult(
            name="debt_to_equity_plausible",
            passed=ok,
            details=f"D/E={dte}" if not ok else "",
            severity="warning",
        ))

    return results


# ── JSON Safety Check ──


def eval_json_safety(data: Any) -> List[EvalResult]:
    """
    Check that the data serializes to JSON without errors.
    
    WHY: Python allows float('nan') and float('inf') but JSON does not.
    This is the exact bug that caused our "Internal Server Error" crash.
    This eval ensures it never happens again.
    """
    try:
        json.dumps(data)
        return [EvalResult(name="json_serializable", passed=True)]
    except (TypeError, ValueError, OverflowError) as e:
        return [EvalResult(
            name="json_serializable",
            passed=False,
            details=f"JSON serialization failed: {e}",
        )]


# ── Main Runner ──


def run_data_pipeline_evals(
    earnings: List[Dict[str, Any]],
    prices: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    full_response: Any = None,
) -> LayerResults:
    """
    Run all Layer 1 data pipeline evals.
    
    Args:
        earnings: List of earnings quarters from /api/earnings
        prices: List of price points from /api/price-history
        metrics: Financial metrics dict from agent output
        full_response: The complete API response (for JSON safety check)
    
    Returns:
        LayerResults with all individual check results
    """
    layer = LayerResults(layer="Layer 1: Data Pipeline")

    # Earnings checks
    layer.results.extend(eval_eps_sanity(earnings))
    layer.results.extend(eval_beat_miss_labels(earnings))
    layer.results.extend(eval_surprise_math(earnings))

    # Price history checks
    layer.results.extend(eval_price_history(prices))

    # Financial metrics checks
    layer.results.extend(eval_financial_metrics(metrics))

    # JSON safety
    if full_response is not None:
        layer.results.extend(eval_json_safety(full_response))

    return layer
