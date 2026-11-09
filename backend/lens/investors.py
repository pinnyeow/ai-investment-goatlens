"""The five investor lenses as plain checklists.

Each investor is a list of criteria. A criterion looks at one or two metrics and
awards points (positive or negative), with a sentence explaining why. When a
metric is missing the criterion is skipped and reported as "no data". It never
counts against the company.

Score = points earned / points available on the criteria that could be checked,
scaled to -100..100. Coverage = share of the checklist that could be checked.
Thresholds come from the original GOATlens agents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Dict, List, Optional, Tuple

Metrics = Dict[str, Optional[float]]
# A rule returns (points, status, sentence) or None when its inputs are missing.
Rule = Callable[[Metrics], Optional[Tuple[float, str, str]]]

MIN_COVERAGE = 0.5  # an investor needs at least half its checklist to give a verdict


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    max_points: float  # 0 for penalty-only checks (red flags) and bonuses
    rule: Rule
    needs: Tuple[str, ...] = ()  # metrics the rule reads, used to explain "no data"


METRIC_NAMES = {
    "roe": "return on equity", "profit_margin": "profit margin", "debt_to_equity": "debt-to-equity",
    "net_debt_to_ebitda": "net debt / EBITDA", "free_cash_flow": "free cash flow", "pe_ratio": "P/E (needs positive earnings)",
    "earnings_growth": "earnings growth (needs positive earnings)", "institutional_ownership": "institutional ownership",
    "eps": "earnings per share", "book_value_per_share": "book value", "pb_ratio": "price-to-book",
    "current_ratio": "current ratio", "consecutive_dividend_years": "dividend history",
    "current_assets": "current assets", "total_liabilities": "total liabilities",
    "shares_outstanding": "shares outstanding", "gross_margin": "gross margin",
    "operating_margin": "operating margin", "roic": "return on invested capital",
    "capex_to_revenue": "capital spending", "revenue_growth": "revenue history",
    "margin_trend": "multi-year margins", "insider_ownership": "insider ownership",
    "realized_total_return_252d": "a full year of prices", "volatility_252d": "a full year of prices",
    "correlation_sp500": "a full year of prices", "debt_growth_3y": "multi-year debt history", "beta": "beta",
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def have(m: Metrics, *keys: str) -> bool:
    return all(m.get(k) is not None for k in keys)


def tiered(value: float, tiers: List[Tuple[Callable[[float], bool], float]], default: float) -> float:
    for test, points in tiers:
        if test(value):
            return points
    return default


def status_for(points: float) -> str:
    return "pass" if points > 0 else "fail" if points < 0 else "neutral"


def leverage(m, good: float, ok: float):
    """Return (tier, sentence) where tier is "good", "ok" or "high", or None.

    Uses debt-to-equity normally. When shareholders' equity is negative (common
    after years of buybacks, e.g. McDonald's or Starbucks) debt-to-equity is
    undefined, so fall back to net debt / EBITDA: years of earnings to repay debt.
    """
    if have(m, "debt_to_equity"):
        de = m["debt_to_equity"]
        tier = "good" if de <= good else "ok" if de <= ok else "high"
        return tier, f"Debt-to-equity {de:.2f}"
    if have(m, "net_debt_to_ebitda"):
        nd = m["net_debt_to_ebitda"]
        tier = "good" if nd <= 1.0 else "ok" if nd <= 2.5 else "high"
        return tier, f"Net debt is {nd:.1f}× yearly EBITDA (equity is negative, so debt-to-equity doesn't apply)"
    return None


# ── Warren Buffett ──────────────────────────────────────────────────────────

def _buffett_roe(m):
    if have(m, "roe"):
        roe, text = m["roe"], f"Return on equity {pct(m['roe'])} (Buffett wants 15%+)"
    elif have(m, "roic"):  # ROE is undefined when equity is negative
        roe, text = m["roic"], f"Return on invested capital {pct(m['roic'])} (ROE doesn't apply; Buffett wants 15%+)"
    else:
        return None
    points = min(25, roe / 0.15 * 15) if roe >= 0.15 else -15
    return points, status_for(points), text


def _buffett_margin(m):
    if not have(m, "profit_margin"):
        return None
    pm = m["profit_margin"]
    points = min(20, pm / 0.10 * 10) if pm >= 0.10 else -10
    return points, status_for(points), f"Profit margin {pct(pm)} (10%+ signals pricing power)"


def _buffett_debt(m):
    result = leverage(m, good=0.5, ok=1.0)
    if result is None:
        return None
    tier, text = result
    points = {"good": 20, "ok": 10, "high": -20}[tier]
    return points, status_for(points), f"{text} (Buffett prefers little debt)"


def _buffett_fcf(m):
    if not have(m, "free_cash_flow"):
        return None
    fcf = m["free_cash_flow"]
    points = 15 if fcf > 0 else -20
    word = "positive" if fcf > 0 else "negative"
    return points, status_for(points), f"Free cash flow is {word} (${fcf / 1e9:,.1f}B last fiscal year)"


BUFFETT = [
    Criterion("roe", "Return on equity", 25, _buffett_roe, ('roe', 'roic')),
    Criterion("profit_margin", "Profit margin", 20, _buffett_margin, ('profit_margin',)),
    Criterion("debt_to_equity", "Low debt", 20, _buffett_debt, ('debt_to_equity', 'net_debt_to_ebitda')),
    Criterion("free_cash_flow", "Free cash flow", 15, _buffett_fcf, ('free_cash_flow',)),
]


# ── Peter Lynch ─────────────────────────────────────────────────────────────

def _lynch_peg(m):
    if not have(m, "pe_ratio", "earnings_growth") or m["earnings_growth"] <= 0 or m["pe_ratio"] <= 0:
        return None
    peg = m["pe_ratio"] / (m["earnings_growth"] * 100)
    points = tiered(peg, [(lambda v: v < 0.5, 35), (lambda v: v < 1.0, 25), (lambda v: v < 1.5, 10),
                          (lambda v: v < 2.0, -10)], -25)
    return points, status_for(points), f"PEG ratio {peg:.2f} (under 1.0 is Lynch's sweet spot)"


def _lynch_category(m):
    if m.get("earnings_turnaround") == 1.0:
        return 15, "pass", "Turnaround: earnings swung from loss to profit"
    if not have(m, "earnings_growth"):
        latest, oldest = m.get("eps_latest_annual"), m.get("eps_oldest_annual")
        if latest is not None and latest <= 0:
            if oldest is not None and oldest > 0:
                return -15, "fail", "Latest fiscal year was a loss after earlier profits (Lynch avoided these)"
            return -15, "fail", "No profits yet: Lynch avoided paying for a story without earnings"
        return None
    g = m["earnings_growth"]
    if g >= 0.20:
        points, kind = 25, "Fast grower"
    elif g >= 0.05:
        points, kind = 15, "Stalwart"
    elif g >= 0:
        points, kind = -15, "Slow grower"
    else:
        return -15, "fail", f"Shrinking: earnings per share falling {pct(abs(g))} a year"
    return points, status_for(points), f"{kind}: earnings per share growing {pct(g)} a year"


def _lynch_institutions(m):
    if not have(m, "institutional_ownership"):
        return None
    io = m["institutional_ownership"]
    points = 15 if io < 0.30 else 10 if io < 0.50 else -10 if io > 0.80 else 0
    return points, status_for(points), f"Institutions own {pct(io)} (Lynch liked under-followed stocks)"


def _lynch_balance(m):
    result = leverage(m, good=0.5, ok=1.0)
    if result is None:
        return None
    tier, text = result
    points = {"good": 5, "ok": 0, "high": -5}[tier]
    return points, status_for(points), f"{text} (a strong balance sheet survives mistakes)"


LYNCH = [
    Criterion("peg_ratio", "PEG ratio", 35, _lynch_peg, ('pe_ratio', 'earnings_growth')),
    Criterion("growth_category", "Growth category", 25, _lynch_category, ('earnings_growth',)),
    Criterion("institutional_ownership", "Under-discovered", 15, _lynch_institutions, ('institutional_ownership',)),
    Criterion("balance_sheet", "Balance sheet", 5, _lynch_balance, ('debt_to_equity', 'net_debt_to_ebitda')),
]


# ── Benjamin Graham ─────────────────────────────────────────────────────────

def _graham_margin_of_safety(m):
    if not have(m, "eps", "book_value_per_share", "current_price"):
        return None
    eps, bvps, price = m["eps"], m["book_value_per_share"], m["current_price"]
    if eps <= 0 or bvps <= 0:
        return -30, "fail", "No Graham Number: earnings or book value is negative"
    graham_number = math.sqrt(22.5 * eps * bvps)
    margin = (graham_number - price) / graham_number
    points = tiered(margin, [(lambda v: v >= 0.33, 40), (lambda v: v >= 0.20, 25), (lambda v: v >= 0.10, 10),
                             (lambda v: v >= 0, 0)], -30)
    verb = "below" if margin >= 0 else "above"
    return points, status_for(points), (
        f"Price ${price:,.2f} is {abs(margin) * 100:.0f}% {verb} the Graham Number ${graham_number:,.2f}"
    )


def _graham_pe(m):
    if not have(m, "pe_ratio") or m["pe_ratio"] <= 0:
        return None
    pe = m["pe_ratio"]
    points = 20 if pe <= 10 else 15 if pe <= 15 else 5 if pe <= 20 else -15
    return points, status_for(points), f"P/E {pe:.1f} (Graham's limit is 15)"


def _graham_pe_pb(m):
    if not have(m, "pe_ratio", "pb_ratio") or m["pe_ratio"] <= 0 or m["pb_ratio"] <= 0:
        return None
    product = m["pe_ratio"] * m["pb_ratio"]
    points = 10 if product <= 22.5 else -10
    return points, status_for(points), f"P/E × P/B = {product:.1f} (Graham's limit is 22.5)"


def _graham_current_ratio(m):
    if not have(m, "current_ratio"):
        return None
    cr = m["current_ratio"]
    points = 15 if cr >= 2.0 else 5 if cr >= 1.5 else -15
    return points, status_for(points), f"Current ratio {cr:.2f} (Graham wants 2.0+)"


def _graham_dividends(m):
    if not have(m, "consecutive_dividend_years"):
        return None
    years = int(m["consecutive_dividend_years"])
    points = 15 if years >= 20 else 10 if years >= 10 else 5 if years >= 5 else 0
    text = f"Paid dividends {years} years in a row (Graham wants 20)" if years else "Pays no dividend"
    return points, status_for(points), text


def _graham_net_net(m):
    if not have(m, "current_assets", "total_liabilities", "shares_outstanding", "current_price"):
        return None
    ncav = (m["current_assets"] - m["total_liabilities"]) / m["shares_outstanding"]
    if ncav > m["current_price"]:
        return 20, "pass", f"Net-net bargain: trades below liquidation value (${ncav:,.2f}/share)"
    return 0, "neutral", "Not a net-net (price is above current assets minus all liabilities)"


GRAHAM = [
    Criterion("margin_of_safety", "Margin of safety", 40, _graham_margin_of_safety, ('eps', 'book_value_per_share')),
    Criterion("pe_ratio", "Moderate P/E", 20, _graham_pe, ('pe_ratio',)),
    Criterion("pe_x_pb", "P/E × P/B", 10, _graham_pe_pb, ('pe_ratio', 'pb_ratio')),
    Criterion("current_ratio", "Current ratio", 15, _graham_current_ratio, ('current_ratio',)),
    Criterion("dividend_record", "Dividend record", 15, _graham_dividends, ('consecutive_dividend_years',)),
    Criterion("net_net", "Net-net bonus", 0, _graham_net_net, ('current_assets', 'total_liabilities', 'shares_outstanding')),
]


# ── Charlie Munger ──────────────────────────────────────────────────────────

def _munger_gross(m):
    if not have(m, "gross_margin"):
        return None
    gm = m["gross_margin"]
    points = 25 if gm >= 0.40 else -20 if gm < 0.20 else 0
    return points, status_for(points), f"Gross margin {pct(gm)} (40%+ shows pricing power)"


def _munger_operating(m):
    if not have(m, "operating_margin"):
        return None
    om = m["operating_margin"]
    points = 20 if om >= 0.15 else 0
    return points, status_for(points), f"Operating margin {pct(om)} (Munger likes 15%+)"


def _munger_roic(m):
    if not have(m, "roic"):
        return None
    r = m["roic"]
    points = 25 if r >= 0.15 else -20 if r < 0.08 else 0
    return points, status_for(points), f"Return on invested capital {pct(r)} (15%+ is a great business)"


def _munger_capex(m):
    if not have(m, "capex_to_revenue"):
        return None
    c = m["capex_to_revenue"]
    points = 15 if c < 0.05 else -20 if c > 0.15 else 0
    return points, status_for(points), f"Capital spending is {pct(c)} of revenue (capital-light is better)"


def _munger_growth(m):
    if not have(m, "revenue_growth"):
        return None
    g = m["revenue_growth"]
    points = 15 if g > 0.05 else 0
    return points, status_for(points), f"Revenue growing {pct(g)} a year"


def _munger_debt_flag(m):
    result = leverage(m, good=1.0, ok=1.0)
    if result is None:
        return None
    tier, text = result
    if tier == "high":
        return -20, "fail", f"Red flag: heavy debt. {text}"
    return 0, "neutral", f"No debt red flag. {text}"


def _munger_margin_flag(m):
    if not have(m, "margin_trend"):
        return None
    t = m["margin_trend"]
    if t < -0.02:
        return -20, "fail", f"Red flag: operating margin fell {abs(t) * 100:.1f} points (moat erosion?)"
    return 0, "neutral", f"Operating margin trend {t * 100:+.1f} points over the period"


def _munger_insiders(m):
    if not have(m, "insider_ownership"):
        return None
    io = m["insider_ownership"]
    if io >= 0.10:
        return 15, "pass", f"Insiders own {pct(io)} (skin in the game)"
    if io >= 0.05:
        return 10, "pass", f"Insiders own {pct(io)}"
    if io < 0.01:
        return -20, "fail", f"Insiders own only {pct(io)} (incentives may not be aligned)"
    return 0, "neutral", f"Insiders own {pct(io)}"


MUNGER = [
    Criterion("gross_margin", "Pricing power", 25, _munger_gross, ('gross_margin',)),
    Criterion("operating_margin", "Operating efficiency", 20, _munger_operating, ('operating_margin',)),
    Criterion("roic", "Return on capital", 25, _munger_roic, ('roic',)),
    Criterion("capex_to_revenue", "Capital-light", 15, _munger_capex, ('capex_to_revenue',)),
    Criterion("revenue_growth", "Sustainable growth", 15, _munger_growth, ('revenue_growth',)),
    Criterion("debt_flag", "Debt red flag", 0, _munger_debt_flag, ('debt_to_equity', 'net_debt_to_ebitda')),
    Criterion("margin_flag", "Margin red flag", 0, _munger_margin_flag, ('margin_trend',)),
    Criterion("insider_ownership", "Skin in the game", 15, _munger_insiders, ('insider_ownership',)),
]


# ── Ray Dalio ───────────────────────────────────────────────────────────────

def _dalio_risk_adjusted(m):
    if not have(m, "realized_total_return_252d", "volatility_252d") or m["volatility_252d"] <= 0:
        return None
    ratio = m["realized_total_return_252d"] / m["volatility_252d"]
    points = tiered(ratio, [(lambda v: v >= 1.0, 30), (lambda v: v >= 0.5, 20), (lambda v: v >= 0.25, 10),
                            (lambda v: v >= 0, 0)], -20)
    return points, status_for(points), (
        f"Past-year return {pct(m['realized_total_return_252d'])} for {pct(m['volatility_252d'])} volatility "
        f"(return per unit of risk {ratio:.2f})"
    )


def _dalio_diversification(m):
    if not have(m, "correlation_sp500"):
        return None
    c = m["correlation_sp500"]
    points = 20 * (1 - abs(c))
    return points, "pass" if abs(c) < 0.5 else "neutral", f"Correlation with the S&P 500: {c:.2f} (lower diversifies more)"


def _dalio_leverage(m):
    result = leverage(m, good=1.0, ok=1.0)
    if result is None:
        return None
    tier, text = result
    points = 15 if tier == "good" else -20
    return points, status_for(points), f"{text} (Dalio watches whether debt is sustainable)"


def _dalio_debt_cycle(m):
    if not have(m, "debt_growth_3y"):
        return None
    g = m["debt_growth_3y"]
    if g > 0.15:
        return 5, "neutral", f"Debt grew {pct(g)} over the period (expansion phase)"
    if g > 0:
        return 10, "pass", f"Debt grew a moderate {pct(g)} over the period"
    return 0, "neutral", f"Debt shrank {pct(abs(g))} over the period (deleveraging)"


def _dalio_beta(m):
    if not have(m, "beta"):
        return None
    b = m["beta"]
    points = 15 if 0.8 <= b <= 1.2 else 10 if 0.5 <= b <= 1.5 else -10
    return points, status_for(points), f"Beta {b:.2f} (balanced risk is near 1.0)"


DALIO = [
    Criterion("risk_adjusted_return", "Return per unit of risk", 30, _dalio_risk_adjusted, ('realized_total_return_252d', 'volatility_252d')),
    Criterion("diversification", "Diversification", 20, _dalio_diversification, ('correlation_sp500',)),
    Criterion("leverage", "Sustainable leverage", 15, _dalio_leverage, ('debt_to_equity', 'net_debt_to_ebitda')),
    Criterion("debt_cycle", "Debt cycle", 10, _dalio_debt_cycle, ('debt_growth_3y',)),
    Criterion("beta", "Market sensitivity", 15, _dalio_beta, ('beta',)),
]


INVESTORS = {
    "buffett": ("Warren Buffett", "Value Investing with Quality Focus", BUFFETT),
    "lynch": ("Peter Lynch", "Growth at Reasonable Price (GARP)", LYNCH),
    "graham": ("Benjamin Graham", "Deep Value / Margin of Safety", GRAHAM),
    "munger": ("Charlie Munger", "Quality-Focused Mental Models", MUNGER),
    "dalio": ("Ray Dalio", "Macro & Risk Parity", DALIO),
}


def verdict_for(score: float) -> str:
    if score >= 60:
        return "strong_buy"
    if score >= 20:
        return "buy"
    if score >= -20:
        return "hold"
    if score >= -60:
        return "sell"
    return "strong_sell"


def _no_data_reason(criterion: Criterion, metrics: Metrics, not_applicable: set) -> str:
    if not_applicable.intersection(criterion.needs):
        return "Not meaningful for banks and insurers, so not scored"
    missing = [METRIC_NAMES.get(k, k) for k in criterion.needs if metrics.get(k) is None]
    if missing:
        return f"Not scored: no {', '.join(dict.fromkeys(missing))} for this company"
    return "Not scored: the inputs don't apply here (for example, negative earnings)"


def evaluate_investor(investor_id: str, metrics: Metrics, not_applicable: Optional[set] = None) -> Dict:
    not_applicable = not_applicable or set()
    name, style, criteria = INVESTORS[investor_id]
    checks = []
    earned = available = 0.0
    total_weight = sum(max(c.max_points, 5) for c in criteria)
    checked_weight = 0.0
    for criterion in criteria:
        outcome = criterion.rule(metrics)
        if outcome is None:
            checks.append({"key": criterion.key, "label": criterion.label, "status": "no_data",
                           "points": None, "detail": _no_data_reason(criterion, metrics, not_applicable)})
            continue
        points, status, detail = outcome
        earned += points
        available += criterion.max_points
        checked_weight += max(criterion.max_points, 5)
        checks.append({"key": criterion.key, "label": criterion.label, "status": status,
                       "points": round(points, 1), "detail": detail})

    coverage = checked_weight / total_weight
    scored = coverage >= MIN_COVERAGE and available > 0
    score = round(max(-100.0, min(100.0, earned / available * 100)), 1) if scored else None
    missing = [c["key"] for c in checks if c["status"] == "no_data"]
    return {
        "id": investor_id,
        "agent": name,
        "style": style,
        "score": score,
        "verdict": verdict_for(score) if score is not None else None,
        "status": "complete" if scored else "insufficient_data",
        "coverage": round(coverage, 2),
        "criteria_checked": len(checks) - len(missing),
        "criteria_total": len(checks),
        "unavailable_metrics": missing,
        "criteria": checks,
        "insights": [c["detail"] for c in checks if c["status"] == "pass"],
        "concerns": [c["detail"] for c in checks if c["status"] == "fail"],
    }


VERDICT_ORDER = ["strong_sell", "sell", "hold", "buy", "strong_buy"]


def consensus(results: List[Dict]) -> Dict:
    scored = [r for r in results if r["score"] is not None]
    if not scored:
        return {
            "consensus_verdict": "insufficient_data",
            "consensus_score": None,
            "agreement_score": None,
            "consensus_points": [],
            "divergence_points": ["Not enough data for any investor to form a view."],
        }
    scores = [r["score"] for r in scored]
    average = round(mean(scores), 1)
    verdict = verdict_for(average)
    agreement = round(max(0.0, 1 - pstdev(scores) / 100), 2) if len(scores) > 1 else 1.0

    bullish = [r["agent"] for r in scored if r["verdict"] in ("buy", "strong_buy")]
    bearish = [r["agent"] for r in scored if r["verdict"] in ("sell", "strong_sell")]
    points, divergences = [], []
    if bullish:
        points.append(f"{len(bullish)} of {len(scored)} investors lean positive: {', '.join(bullish)}")
    if bearish:
        points.append(f"{len(bearish)} of {len(scored)} investors lean negative: {', '.join(bearish)}")
    if bullish and bearish:
        divergences.append(
            f"{', '.join(bullish)} vs {', '.join(bearish)}: their frameworks weigh this company very differently"
        )
    high = max(scored, key=lambda r: r["score"])
    low = min(scored, key=lambda r: r["score"])
    if high["score"] - low["score"] >= 60:
        divergences.append(
            f"Widest gap: {high['agent']} ({high['score']:+.0f}) vs {low['agent']} ({low['score']:+.0f})"
        )
    return {
        "consensus_verdict": verdict,
        "consensus_score": average,
        "agreement_score": agreement,
        "consensus_points": points,
        "divergence_points": divergences,
    }


def moat_trend(metrics: Metrics) -> str:
    trend = metrics.get("margin_trend")
    if trend is None:
        return "unknown"
    if trend > 0.02:
        return "strengthening"
    if trend < -0.02:
        return "weakening"
    return "stable"
