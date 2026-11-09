"""Fetch one company's data from Yahoo Finance and turn it into plain metrics.

Runs in the nightly GitHub Action, never inside a user request. Every metric is
either a number or None. None means "Yahoo does not report it" or "it does not
apply" (for example a current ratio for a bank). It never means zero.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf


def _num(value: Any) -> Optional[float]:
    """Return a finite float, or None for anything missing or non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row(frame: Optional[pd.DataFrame], *names: str) -> List[Optional[float]]:
    """Annual values for the first matching row name, newest first."""
    if frame is None or frame.empty:
        return []
    for name in names:
        if name in frame.index:
            return [_num(v) for v in frame.loc[name].tolist()]
    return []


def _filled(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Drop report columns that are mostly empty. Yahoo sometimes adds a column
    for a fiscal year before its numbers are filled in."""
    if frame is None or frame.empty:
        return frame
    return frame.loc[:, frame.notna().mean() >= 0.5]


def _scaled(value: Optional[float], factor: Optional[float]) -> Optional[float]:
    return value * factor if value is not None and factor else None


def _first(values: List[Optional[float]]) -> Optional[float]:
    return values[0] if values else None


def _cagr(values: List[Optional[float]]) -> Optional[float]:
    """Compound annual growth from the oldest to the newest positive value."""
    points = [v for v in values if v is not None]
    if len(points) < 2:
        return None
    newest, oldest = points[0], points[-1]
    years = len(points) - 1
    if newest <= 0 or oldest <= 0:
        return None
    return (newest / oldest) ** (1 / years) - 1


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _dividend_streak(dividends: pd.Series) -> int:
    """Consecutive calendar years with a dividend, counting back from last year."""
    if dividends is None or dividends.empty:
        return 0
    years = {d.year for d in dividends[dividends > 0].index}
    year = date.today().year - 1
    streak = 0
    while year in years:
        streak += 1
        year -= 1
    return streak


def _daily_returns(history: pd.DataFrame) -> pd.Series:
    closes = history["Adj Close"] if "Adj Close" in history else history["Close"]
    closes.index = closes.index.tz_localize(None).normalize()
    return closes.pct_change().dropna()


_FX_CACHE: Dict[str, Optional[float]] = {}


def fx_rate(from_currency: str, to_currency: str) -> Optional[float]:
    """Latest exchange rate from Yahoo (e.g. TWD -> USD), cached for the build."""
    pair = f"{from_currency}{to_currency}=X"
    if pair not in _FX_CACHE:
        try:
            closes = yf.Ticker(pair).history(period="5d")["Close"].dropna()
            _FX_CACHE[pair] = float(closes.iloc[-1]) if len(closes) else None
        except Exception:
            _FX_CACHE[pair] = None
    return _FX_CACHE[pair]


def fetch_market_returns() -> pd.Series:
    """One year of S&P 500 daily returns, shared by every company in a build."""
    history = yf.Ticker("^GSPC").history(period="1y", auto_adjust=False)
    return _daily_returns(history)


def fetch_company(ticker: str, market_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Fetch everything GOATlens shows for one ticker.

    Raises ValueError when Yahoo has no usable quote for the ticker.
    """
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    if price is None:
        raise ValueError(f"No price for {ticker}")

    income = _filled(tk.financials)
    balance = _filled(tk.balance_sheet)
    cashflow = _filled(tk.cashflow)
    history = tk.history(period="5y", auto_adjust=False)
    dividends = tk.dividends

    revenue = _row(income, "Total Revenue", "Operating Revenue")
    eps_history = _row(income, "Diluted EPS", "Basic EPS")
    operating_income = _row(income, "Operating Income", "EBIT")
    ebit = _first(_row(income, "EBIT", "Operating Income"))
    tax_rate = _first(_row(income, "Tax Rate For Calcs"))
    interest = _first(_row(income, "Interest Expense", "Interest Expense Non Operating"))
    capex = _first(_row(cashflow, "Capital Expenditure"))
    total_debt = _row(balance, "Total Debt")
    net_debt = _first(_row(balance, "Net Debt"))
    ebitda = _first(_row(income, "EBITDA", "Normalized EBITDA"))
    invested_capital = _first(_row(balance, "Invested Capital"))
    equity = _first(_row(balance, "Stockholders Equity", "Common Stock Equity"))
    # implied shares counts every share class in one class's units (Berkshire's A
    # and B shares); plain sharesOutstanding can cover a single class only.
    shares = _num(info.get("impliedSharesOutstanding")) or _num(info.get("sharesOutstanding"))
    currency = info.get("currency") or "USD"
    financial_currency = info.get("financialCurrency") or currency
    foreign_statements = financial_currency != currency
    # Foreign filers (ADRs) report in their home currency. Convert at today's rate,
    # but only if the converted figures reproduce Yahoo's own P/E: that proves the
    # share count is in ADR units (it sometimes counts ordinary shares instead).
    fx = 1.0
    if foreign_statements:
        rate = fx_rate(financial_currency, currency)
        net_income = _first(_row(income, "Net Income", "Net Income Common Stockholders"))
        trailing_pe = _num(info.get("trailingPE"))
        fx = None
        if rate and net_income and net_income > 0 and shares and trailing_pe:
            implied_pe = price * shares / (net_income * rate)
            if 0.6 <= implied_pe / trailing_pe <= 1.6:
                fx = rate
    # Book value per share from the balance sheet; Yahoo's own figure is sometimes
    # per a different share class (Berkshire) or a different currency (ADRs).
    book_value_per_share = (
        equity * fx / shares if equity is not None and shares and fx else None
    )

    op_margins = [
        _ratio(op, rev) for op, rev in zip(operating_income, revenue)
    ]
    op_margins = [m for m in op_margins if m is not None]

    # Earnings growth: annual diluted EPS growth across the years Yahoo reports.
    eps_points = [v for v in eps_history if v is not None]
    earnings_growth = _cagr(eps_points)
    earnings_turnaround = (
        1.0 if len(eps_points) >= 2 and eps_points[-1] <= 0 < eps_points[0] else None
    )

    # Yahoo reports debt-to-equity as a percentage (84.3 means 0.843).
    debt_to_equity = _num(info.get("debtToEquity"))
    if debt_to_equity is not None:
        debt_to_equity /= 100

    dividend_rate = _num(info.get("dividendRate"))
    returns = _daily_returns(history.tail(253)) if not history.empty else pd.Series(dtype=float)
    volatility = float(returns.std() * math.sqrt(252)) if len(returns) > 100 else None
    one_year_return = None
    if len(history) > 252:
        closes = history["Adj Close"]
        one_year_return = float(closes.iloc[-1] / closes.iloc[-253] - 1)
    correlation = None
    if market_returns is not None and len(returns) > 100:
        joined = pd.concat([returns, market_returns], axis=1, join="inner").dropna()
        if len(joined) > 100:
            correlation = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))

    metrics: Dict[str, Optional[float]] = {
        "current_price": price,
        "pe_ratio": _num(info.get("trailingPE")),
        "pb_ratio": _ratio(price, book_value_per_share) if book_value_per_share and book_value_per_share > 0 else None,
        "eps": _num(info.get("trailingEps")),
        "book_value_per_share": book_value_per_share,
        "roe": _num(info.get("returnOnEquity")),
        "profit_margin": _num(info.get("profitMargins")),
        "gross_margin": _num(info.get("grossMargins")),
        "operating_margin": _num(info.get("operatingMargins")),
        "debt_to_equity": debt_to_equity,
        "current_ratio": _num(info.get("currentRatio")),
        "free_cash_flow": _first(_row(cashflow, "Free Cash Flow")),
        "earnings_growth": earnings_growth,
        "eps_latest_annual": eps_points[0] if eps_points else None,
        "eps_oldest_annual": eps_points[-1] if len(eps_points) >= 2 else None,
        # Used instead of debt-to-equity when equity is negative (e.g. after big buybacks).
        "net_debt_to_ebitda": _ratio(net_debt, ebitda) if ebitda and ebitda > 0 else None,
        "earnings_turnaround": earnings_turnaround,
        "revenue_growth": _cagr(revenue),
        "margin_trend": (op_margins[0] - op_margins[-1]) if len(op_margins) >= 2 else None,
        "capex_to_revenue": _ratio(abs(capex) if capex is not None else None, _first(revenue)),
        "roic": (  # undefined when invested capital is zero or negative
            _ratio(ebit * (1 - (tax_rate if tax_rate is not None else 0.21)), invested_capital)
            if ebit is not None and invested_capital and invested_capital > 0 else None
        ),
        "interest_coverage": _ratio(ebit, abs(interest) if interest else None),
        "debt_growth_3y": (
            _ratio(total_debt[0] - total_debt[-1], total_debt[-1])
            if len(total_debt) >= 3 and total_debt[0] is not None and total_debt[-1] else None
        ),
        "current_assets": _scaled(_first(_row(balance, "Current Assets")), fx),
        "total_liabilities": _scaled(_first(_row(balance, "Total Liabilities Net Minority Interest")), fx),
        "shares_outstanding": shares,
        "institutional_ownership": _num(info.get("heldPercentInstitutions")),
        "insider_ownership": _num(info.get("heldPercentInsiders")),
        "dividend_yield": _ratio(dividend_rate, price) if dividend_rate else 0.0,
        "consecutive_dividend_years": float(_dividend_streak(dividends)),
        "beta": _num(info.get("beta")),
        "volatility_252d": volatility,
        "realized_total_return_252d": one_year_return,
        "correlation_sp500": correlation,
    }

    # Metrics that exist but would mislead, with the reason shown to users.
    not_applicable: Dict[str, str] = {}
    industry = (info.get("industry") or "")
    is_financial = any(word in industry for word in ("Bank", "Insurance", "Capital Markets", "Mortgage"))
    if is_financial:
        # Deposits and insurance float flow through cash flow and debt.
        for key in ("free_cash_flow", "gross_margin", "capex_to_revenue", "current_ratio",
                    "debt_to_equity", "net_debt_to_ebitda", "debt_growth_3y", "roic",
                    "current_assets", "total_liabilities"):
            not_applicable[key] = "Not meaningful for banks and insurers"
    if foreign_statements and fx is None:
        # Couldn't convert reliably, so per-share and absolute amounts can't be
        # compared with the US share price. Ratios and margins are unaffected.
        for key in ("book_value_per_share", "pb_ratio", "current_assets", "total_liabilities"):
            not_applicable.setdefault(
                key, f"Reported in {financial_currency}, so it can't be compared with the US share price"
            )
    for key in not_applicable:
        metrics[key] = None

    fiscal_year_end = None
    if income is not None and not income.empty:
        fiscal_year_end = str(pd.Timestamp(income.columns[0]).date())

    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector") or "Other",
        "industry": info.get("industry"),
        "currency": currency,
        "financial_currency": financial_currency,
        "fx_rate_used": fx if foreign_statements else None,
        "is_financial": is_financial,
        "not_applicable": not_applicable,
        "metrics": metrics,
        "statement_years": len([v for v in revenue if v is not None]),
        "fiscal_year_end": fiscal_year_end,
        "price_as_of": _price_time(info, history),
        "prices": _prices(history),
        **_earnings(tk),
    }


def _price_time(info: Dict[str, Any], history: pd.DataFrame) -> Optional[str]:
    stamp = info.get("regularMarketTime")
    if isinstance(stamp, (int, float)):
        return datetime.fromtimestamp(stamp, tz=timezone.utc).date().isoformat()
    if not history.empty:
        return str(history.index[-1].date())
    return None


def _prices(history: pd.DataFrame) -> List[List[Any]]:
    """Compact rows (date, open, high, low, close, adj close, volume): daily for the
    last year, weekly before that, so five years fits in a small file."""
    if history.empty:
        return []
    cutoff = history.index[-1] - pd.Timedelta(days=372)
    older = history[history.index < cutoff]
    if not older.empty:  # last trading day of each week
        weeks = older.index.tz_localize(None).to_period("W")
        older = older[~weeks.duplicated(keep="last")]
    trimmed = pd.concat([older, history[history.index >= cutoff]])
    rows = []
    for stamp, row in trimmed.iterrows():
        values = [_num(row.get(k)) for k in ("Open", "High", "Low", "Close", "Adj Close")]
        if any(v is None for v in values):
            continue
        rows.append([str(stamp.date()), *[round(v, 4) for v in values], int(_num(row.get("Volume")) or 0)])
    return rows


def _earnings(tk: yf.Ticker) -> Dict[str, Any]:
    """Past earnings announcements (by announcement date) and the next one."""
    empty: Dict[str, Any] = {"earnings_history": [], "next_earnings_date": None}
    try:
        frame = tk.get_earnings_dates(limit=12)
    except Exception:
        return empty
    if frame is None or frame.empty:
        return empty
    today = pd.Timestamp.now(tz="UTC").normalize()
    quarters, upcoming = [], []
    for stamp, row in frame.sort_index(ascending=False).iterrows():
        when = pd.Timestamp(stamp)
        when_utc = when.tz_convert("UTC") if when.tzinfo else when.tz_localize("UTC")
        actual, estimate = _num(row.get("Reported EPS")), _num(row.get("EPS Estimate"))
        if actual is None:
            if when_utc >= today:
                upcoming.append(str(when.date()))
            continue
        surprise = _num(row.get("Surprise(%)"))
        if surprise is None:
            beat_miss = "unknown"
        elif abs(surprise) < 1:
            beat_miss = "inline"
        else:
            beat_miss = "beat" if surprise > 0 else "miss"
        quarters.append({
            "quarter": f"Reported {when.strftime('%b %Y')}",
            "date": str(when.date()),
            "eps_actual": actual,
            "eps_estimate": estimate,
            "eps_surprise": round(actual - estimate, 4) if estimate is not None else None,
            "surprise_pct": round(surprise, 2) if surprise is not None else None,
            "beat_miss": beat_miss,
        })
    return {"earnings_history": quarters[:8], "next_earnings_date": min(upcoming) if upcoming else None}
