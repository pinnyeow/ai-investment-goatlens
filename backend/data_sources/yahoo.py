"""
Yahoo Finance Data Client

Free data source for GOATlens using yfinance library.
Provides comprehensive financial data including:
- Company info and profile
- Income statements
- Balance sheets
- Cash flow statements
- Key ratios and metrics
- Historical prices
- Earnings history (actual vs. consensus estimates)

No API key required!

Performance:
    Module-level TTL cache prevents redundant Yahoo scrapes when the
    same ticker is requested by multiple endpoints within a short window
    (default 5 min).  Cache is keyed by (ticker, method_name, args_hash).
"""

import math
import time
import hashlib
from datetime import datetime, timedelta

import yfinance as yf
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor


# ─── Module-level TTL cache ───────────────────────────────────────────────
# Avoids re-scraping Yahoo Finance when /api/analyze, /api/price-history,
# and /api/earnings all request the same ticker within the TTL window.
# Format: { cache_key: (timestamp, data) }
_cache: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL = 300  # 5 minutes


def _cache_key(namespace: str, *args) -> str:
    """Build a deterministic cache key from namespace + args."""
    raw = f"{namespace}:" + ":".join(str(a) for a in args)
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str):
    """Return cached value if within TTL, else None."""
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    # Expired — remove to free memory
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value):
    """Store value with current timestamp."""
    _cache[key] = (time.time(), value)


def clear_cache():
    """Clear all cached data (useful for testing)."""
    _cache.clear()


def _safe_float(val, default=0) -> Optional[float]:
    """
    Sanitize a float value for JSON serialization.

    Python's json module cannot serialize NaN or Infinity — they crash
    FastAPI's response builder. This helper converts any NaN/inf to a
    safe default (0 or None) so the pipeline never produces poison values.

    Args:
        val: The value to check. May be float, numpy.float64, int, None, etc.
        default: Value to return when val is NaN/inf/None. Use 0 for required
                 numeric fields, None for optional fields.

    Returns:
        A JSON-safe float (or the default).
    """
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


class YahooFinanceError(Exception):
    """Exception raised for Yahoo Finance errors."""
    pass


@dataclass
class FinancialData:
    """Normalized financial data structure."""
    ticker: str
    company_name: str
    sector: str
    industry: str
    
    # Valuation
    market_cap: float
    current_price: float
    pe_ratio: float
    pb_ratio: float
    ps_ratio: float
    peg_ratio: float
    
    # Profitability
    gross_margin: float
    operating_margin: float
    profit_margin: float
    roe: float
    roa: float
    roic: float
    
    # Growth
    revenue_growth: float
    earnings_growth: float
    revenue_growth_5y: float
    
    # Financial Health
    current_ratio: float
    quick_ratio: float
    debt_to_equity: float
    interest_coverage: float
    
    # Per Share Data
    eps: float
    book_value_per_share: float
    free_cash_flow_per_share: float
    dividend_yield: float
    
    # Risk
    beta: float
    volatility_252d: float
    
    # Ownership
    insider_ownership: float
    institutional_ownership: float


class YahooFinanceClient:
    """
    Client for Yahoo Finance data via yfinance.
    
    Usage:
        client = YahooFinanceClient()
        data = await client.get_company_data("AAPL")
    """
    
    def __init__(self):
        """Initialize Yahoo Finance client."""
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def close(self):
        """Close the client."""
        self._executor.shutdown(wait=False)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get yfinance Ticker object."""
        return yf.Ticker(symbol)
    
    async def _run_sync(self, func, *args):
        """Run synchronous function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, func, *args)
    
    def _fetch_all_data_sync(self, ticker: str) -> Dict[str, Any]:
        """Synchronously fetch all company data (cached for TTL)."""
        key = _cache_key("company", ticker)
        cached = _cache_get(key)
        if cached is not None:
            return cached

        try:
            stock = self._get_ticker(ticker)
            
            # Get basic info
            info = stock.info
            if not info or info.get('regularMarketPrice') is None:
                raise YahooFinanceError(f"No data found for ticker: {ticker}")
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Get historical income statements (quarterly)
            income_quarterly = stock.quarterly_income_stmt
            
            # Get earnings history (actual vs. consensus estimates)
            earnings_dates = None
            try:
                earnings_dates = stock.earnings_dates
            except Exception:
                pass  # Some tickers may not have earnings dates
            
            result = {
                "ticker": ticker,
                "info": info,
                "income_statements": income_stmt,
                "income_quarterly": income_quarterly,
                "balance_sheets": balance_sheet,
                "cash_flow_statements": cash_flow,
                "earnings_dates": earnings_dates,
                "fetched_at": datetime.now().isoformat(),
            }
            _cache_set(key, result)
            return result
        except YahooFinanceError:
            raise
        except Exception as e:
            raise YahooFinanceError(f"Failed to fetch data for {ticker}: {str(e)}")
    
    async def get_company_data(self, ticker: str, years: int = 10) -> Dict[str, Any]:
        """
        Fetch all company data.
        
        This is the main entry point for getting comprehensive
        company data for GOAT analysis.
        
        Args:
            ticker: Stock ticker symbol
            years: Number of years of historical data (not all may be available)
            
        Returns:
            Comprehensive company data dictionary
        """
        return await self._run_sync(self._fetch_all_data_sync, ticker)
    
    def normalize_data(self, raw_data: Dict[str, Any]) -> FinancialData:
        """
        Normalize raw Yahoo Finance data into standardized FinancialData structure.
        
        Args:
            raw_data: Raw data from get_company_data()
            
        Returns:
            Normalized FinancialData object
        """
        info = raw_data.get("info", {})
        
        # Safe getter — also catches NaN/inf values from Yahoo Finance
        def get_val(key, default=0):
            val = info.get(key)
            if isinstance(val, (int, float)):
                return _safe_float(val, default)
            return val if val is not None else default
        
        return FinancialData(
            ticker=raw_data["ticker"],
            company_name=get_val("longName", get_val("shortName", "")),
            sector=get_val("sector", "Unknown"),
            industry=get_val("industry", "Unknown"),
            
            # Valuation
            market_cap=get_val("marketCap", 0),
            current_price=get_val("regularMarketPrice", get_val("currentPrice", 0)),
            pe_ratio=get_val("trailingPE", get_val("forwardPE", 0)),
            pb_ratio=get_val("priceToBook", 0),
            ps_ratio=get_val("priceToSalesTrailing12Months", 0),
            peg_ratio=get_val("pegRatio", 0),
            
            # Profitability
            gross_margin=get_val("grossMargins", 0),
            operating_margin=get_val("operatingMargins", 0),
            profit_margin=get_val("profitMargins", 0),
            roe=get_val("returnOnEquity", 0),
            roa=get_val("returnOnAssets", 0),
            roic=get_val("returnOnCapitalEmployed", 0),
            
            # Growth
            revenue_growth=get_val("revenueGrowth", 0),
            earnings_growth=get_val("earningsGrowth", get_val("earningsQuarterlyGrowth", 0)),
            revenue_growth_5y=self._calculate_revenue_cagr(raw_data),
            
            # Financial Health
            current_ratio=get_val("currentRatio", 0),
            quick_ratio=get_val("quickRatio", 0),
            debt_to_equity=get_val("debtToEquity", 0) / 100 if get_val("debtToEquity", 0) else 0,
            interest_coverage=self._calculate_interest_coverage(raw_data),
            
            # Per Share Data
            eps=get_val("trailingEps", 0),
            book_value_per_share=get_val("bookValue", 0),
            free_cash_flow_per_share=get_val("freeCashflow", 0) / get_val("sharesOutstanding", 1) if get_val("sharesOutstanding") else 0,
            dividend_yield=get_val("dividendYield", 0),
            
            # Risk
            beta=get_val("beta", 1.0),
            volatility_252d=0,  # Would need price history to calculate
            
            # Ownership
            insider_ownership=get_val("heldPercentInsiders", 0),
            institutional_ownership=get_val("heldPercentInstitutions", 0),
        )
    
    def normalize_earnings(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize earnings dates into a clean list of quarterly earnings records.
        
        Each record contains EPS actual vs. estimate and beat/miss classification.
        
        Args:
            raw_data: Raw data from get_company_data()
            
        Returns:
            List of earnings records sorted by date (most recent first),
            up to the last 8 quarters.
        """
        earnings_dates = raw_data.get("earnings_dates")
        if earnings_dates is None or (hasattr(earnings_dates, 'empty') and earnings_dates.empty):
            return []
        
        records = []
        try:
            df = earnings_dates
            
            # yfinance earnings_dates columns vary; find the right ones
            # Common columns: "EPS Estimate", "Reported EPS", "Surprise(%)"
            eps_est_col = None
            eps_act_col = None
            surprise_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if "estimate" in col_lower and "eps" in col_lower:
                    eps_est_col = col
                elif "reported" in col_lower and "eps" in col_lower:
                    eps_act_col = col
                elif "surprise" in col_lower:
                    surprise_col = col
            
            if eps_est_col is None or eps_act_col is None:
                return []
            
            for idx, row in df.iterrows():
                eps_est = row.get(eps_est_col)
                eps_act = row.get(eps_act_col)
                
                # Skip rows without both actual and estimate (future dates).
                # IMPORTANT: Yahoo returns NaN (not None) for missing values,
                # and NaN is NOT None, so we must use _safe_float to catch it.
                eps_est = _safe_float(eps_est, default=None)
                eps_act = _safe_float(eps_act, default=None)
                if eps_act is None or eps_est is None:
                    continue
                
                # Calculate surprise
                surprise = eps_act - eps_est
                surprise_pct = 0.0
                if eps_est != 0:
                    surprise_pct = _safe_float((surprise / abs(eps_est)) * 100, default=0.0)
                
                # Determine beat/miss (inline if within 1% of estimate)
                if surprise_pct > 1.0:
                    beat_miss = "beat"
                elif surprise_pct < -1.0:
                    beat_miss = "miss"
                else:
                    beat_miss = "inline"
                
                # Determine quarter label from the date
                if hasattr(idx, 'month'):
                    date_obj = idx
                else:
                    date_obj = None
                
                quarter_label = ""
                date_str = ""
                if date_obj:
                    month = date_obj.month
                    year = date_obj.year
                    if month <= 3:
                        quarter_label = f"Q1 {year}"
                    elif month <= 6:
                        quarter_label = f"Q2 {year}"
                    elif month <= 9:
                        quarter_label = f"Q3 {year}"
                    else:
                        quarter_label = f"Q4 {year}"
                    date_str = date_obj.strftime("%Y-%m-%d")
                
                records.append({
                    "quarter": quarter_label,
                    "date": date_str,
                    "eps_actual": round(eps_act, 2),
                    "eps_estimate": round(eps_est, 2),
                    "eps_surprise": round(surprise, 2),
                    "surprise_pct": round(surprise_pct, 1),
                    "beat_miss": beat_miss,
                })
        except Exception:
            return []
        
        # Sort by date descending (most recent first) and limit to 8 quarters
        records.sort(key=lambda x: x["date"], reverse=True)
        return records[:8]
    
    def get_earnings_streak(self, earnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate earnings beat/miss streak and summary stats.
        
        Args:
            earnings: Normalized earnings list from normalize_earnings()
            
        Returns:
            Dictionary with streak info and summary statistics.
        """
        if not earnings:
            return {
                "streak_type": "unknown",
                "streak_count": 0,
                "beats": 0,
                "misses": 0,
                "inline": 0,
                "total": 0,
                "summary": "No earnings data available",
            }
        
        beats = sum(1 for e in earnings if e["beat_miss"] == "beat")
        misses = sum(1 for e in earnings if e["beat_miss"] == "miss")
        inline = sum(1 for e in earnings if e["beat_miss"] == "inline")
        total = len(earnings)
        
        # Calculate consecutive streak from most recent
        streak_type = earnings[0]["beat_miss"] if earnings else "unknown"
        streak_count = 0
        for e in earnings:
            if e["beat_miss"] == streak_type:
                streak_count += 1
            else:
                break
        
        # Build human-readable summary
        if streak_count >= 4 and streak_type == "beat":
            summary = f"Strong momentum: Beat {streak_count} consecutive quarters"
        elif streak_count >= 2 and streak_type == "beat":
            summary = f"Positive trend: Beat last {streak_count} quarters"
        elif streak_count >= 2 and streak_type == "miss":
            summary = f"Concerning: Missed last {streak_count} quarters"
        else:
            summary = f"Beat {beats} of last {total} quarters"
        
        return {
            "streak_type": streak_type,
            "streak_count": streak_count,
            "beats": beats,
            "misses": misses,
            "inline": inline,
            "total": total,
            "summary": summary,
        }
    
    def get_price_history_sync(self, ticker: str, period: str = "2y") -> List[Dict[str, Any]]:
        """
        Fetch daily OHLCV price history for a ticker (cached for TTL).

        Args:
            ticker: Stock ticker symbol
            period: yfinance period string (1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max)

        Returns:
            List of {date, open, high, low, close, volume} dicts sorted by date ascending.
        """
        key = _cache_key("prices", ticker, period)
        cached = _cache_get(key)
        if cached is not None:
            return cached

        try:
            stock = self._get_ticker(ticker)
            hist = stock.history(period=period)
            if hist is None or hist.empty:
                return []

            records = []
            for idx, row in hist.iterrows():
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                close_val = _safe_float(row.get("Close"), default=None)
                if close_val is None:
                    continue  # Skip days with no close price
                records.append({
                    "date": date_str,
                    "open": round(_safe_float(row.get("Open", 0)), 2),
                    "high": round(_safe_float(row.get("High", 0)), 2),
                    "low": round(_safe_float(row.get("Low", 0)), 2),
                    "close": round(close_val, 2),
                    "volume": int(_safe_float(row.get("Volume", 0))),
                })
            _cache_set(key, records)
            return records
        except Exception:
            return []

    async def get_price_history(self, ticker: str, period: str = "2y") -> List[Dict[str, Any]]:
        """Async wrapper for get_price_history_sync."""
        return await self._run_sync(self.get_price_history_sync, ticker, period)

    def calculate_earnings_reactions(
        self,
        price_history: List[Dict[str, Any]],
        earnings_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For each earnings date, calculate the stock price reaction.

        Finds the closing price the day before earnings, and 1-day / 5-day
        after. Adds price_reaction_1d and price_reaction_5d percentage fields
        to each earnings record.

        Args:
            price_history: Daily OHLCV list from get_price_history()
            earnings_data: Normalized earnings list from normalize_earnings()

        Returns:
            Enriched earnings records with price reaction data.
        """
        if not price_history or not earnings_data:
            return earnings_data

        # Build a lookup: date string -> index in price_history
        date_to_idx: Dict[str, int] = {}
        for i, p in enumerate(price_history):
            date_to_idx[p["date"]] = i

        # Sorted date list for finding nearest trading day
        sorted_dates = [p["date"] for p in price_history]

        def find_nearest_idx(target_date: str, direction: int = -1) -> Optional[int]:
            """Find the nearest trading day index to target_date.
            direction: -1 = before/on, +1 = on/after."""
            if target_date in date_to_idx:
                return date_to_idx[target_date]
            # Linear scan (small list, fine for <=500 trading days)
            if direction == -1:
                best = None
                for d in sorted_dates:
                    if d <= target_date:
                        best = date_to_idx[d]
                    else:
                        break
                return best
            else:
                for d in sorted_dates:
                    if d >= target_date:
                        return date_to_idx[d]
                return None

        enriched = []
        for e in earnings_data:
            record = dict(e)
            earnings_date = e.get("date", "")
            if not earnings_date:
                record["price_reaction_1d"] = None
                record["price_reaction_5d"] = None
                enriched.append(record)
                continue

            # Find day-before index (the trading day on or before earnings date)
            before_idx = find_nearest_idx(earnings_date, direction=-1)
            if before_idx is not None and before_idx > 0:
                # Use the day *before* earnings if the earnings date itself is in data
                if sorted_dates[before_idx] == earnings_date and before_idx > 0:
                    before_idx -= 1
                pre_price = price_history[before_idx]["close"]
            else:
                record["price_reaction_1d"] = None
                record["price_reaction_5d"] = None
                enriched.append(record)
                continue

            # 1-day after
            after_1d_idx = find_nearest_idx(earnings_date, direction=1)
            if after_1d_idx is not None:
                # Move at least 1 trading day after the earnings date
                if sorted_dates[after_1d_idx] == earnings_date and after_1d_idx + 1 < len(price_history):
                    after_1d_idx += 1
                elif sorted_dates[after_1d_idx] > earnings_date:
                    pass  # already the next trading day
                post_1d_price = _safe_float(price_history[after_1d_idx]["close"], default=None)
                if post_1d_price is not None and pre_price:
                    record["price_reaction_1d"] = round(_safe_float(((post_1d_price - pre_price) / pre_price) * 100, default=None), 2) if pre_price else None
                else:
                    record["price_reaction_1d"] = None
            else:
                record["price_reaction_1d"] = None

            # 5-day after
            if after_1d_idx is not None and after_1d_idx + 4 < len(price_history):
                post_5d_price = _safe_float(price_history[after_1d_idx + 4]["close"], default=None)
                if post_5d_price is not None and pre_price:
                    record["price_reaction_5d"] = round(_safe_float(((post_5d_price - pre_price) / pre_price) * 100, default=None), 2) if pre_price else None
                else:
                    record["price_reaction_5d"] = None
            else:
                record["price_reaction_5d"] = None

            enriched.append(record)

        return enriched

    def get_next_earnings_date_sync(self, ticker: str) -> Optional[str]:
        """
        Get the next upcoming earnings date for a ticker (cached for TTL).

        Returns:
            Date string (YYYY-MM-DD) or None if not available.
        """
        key = _cache_key("next_earnings", ticker)
        cached = _cache_get(key)
        if cached is not None:
            return cached

        try:
            stock = self._get_ticker(ticker)
            # Try stock.calendar first
            cal = stock.calendar
            if cal is not None:
                # calendar can be a dict or DataFrame
                date_val = None
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        if isinstance(ed, list) and len(ed) > 0:
                            date_val = str(ed[0])[:10]
                        else:
                            date_val = str(ed)[:10]
                elif hasattr(cal, "loc"):
                    # DataFrame
                    if "Earnings Date" in cal.index:
                        val = cal.loc["Earnings Date"]
                        if hasattr(val, "iloc"):
                            date_val = str(val.iloc[0])[:10]
                        else:
                            date_val = str(val)[:10]
                if date_val:
                    _cache_set(key, date_val)
                    return date_val

            # Fallback: look at earnings_dates for future dates
            ed = stock.earnings_dates
            if ed is not None and not ed.empty:
                today = datetime.now().strftime("%Y-%m-%d")
                for idx in ed.index:
                    date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                    if date_str >= today:
                        _cache_set(key, date_str)
                        return date_str
            return None
        except Exception:
            return None

    async def get_next_earnings_date(self, ticker: str) -> Optional[str]:
        """Async wrapper for get_next_earnings_date_sync."""
        return await self._run_sync(self.get_next_earnings_date_sync, ticker)

    # ------------------------------------------------------------------
    # Earnings Reaction Insights — new data fetchers
    # ------------------------------------------------------------------

    def get_analyst_reactions_sync(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Fetch recent analyst upgrades/downgrades for a ticker (cached for TTL).

        Returns the most recent 100 analyst actions, each with:
        date, firm, to_grade, from_grade, action.
        """
        key = _cache_key("analysts", ticker)
        cached = _cache_get(key)
        if cached is not None:
            return cached

        try:
            stock = self._get_ticker(ticker)
            ud = stock.upgrades_downgrades
            if ud is None or (hasattr(ud, "empty") and ud.empty):
                return []

            records = []
            for idx, row in ud.head(100).iterrows():
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                records.append({
                    "date": date_str,
                    "firm": str(row.get("Firm", "")),
                    "to_grade": str(row.get("ToGrade", "")),
                    "from_grade": str(row.get("FromGrade", "")),
                    "action": str(row.get("Action", "")),
                })
            _cache_set(key, records)
            return records
        except Exception:
            return []

    async def get_analyst_reactions(self, ticker: str) -> List[Dict[str, Any]]:
        """Async wrapper for get_analyst_reactions_sync."""
        return await self._run_sync(self.get_analyst_reactions_sync, ticker)

    def get_quarterly_revenue_sync(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Fetch quarterly revenue from the income statement (cached for TTL).

        Returns a list of {date, revenue} sorted most-recent first.
        """
        key = _cache_key("revenue", ticker)
        cached = _cache_get(key)
        if cached is not None:
            return cached

        try:
            stock = self._get_ticker(ticker)
            qi = stock.quarterly_income_stmt
            if qi is None or qi.empty:
                return []

            rev_row = None
            if "Total Revenue" in qi.index:
                rev_row = qi.loc["Total Revenue"]
            elif "Revenue" in qi.index:
                rev_row = qi.loc["Revenue"]

            if rev_row is None:
                return []

            records = []
            for col in rev_row.index:
                date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
                val = _safe_float(rev_row[col], default=None)
                if val is not None:
                    records.append({"date": date_str, "revenue": val})

            records.sort(key=lambda x: x["date"], reverse=True)
            _cache_set(key, records)
            return records
        except Exception:
            return []

    async def get_quarterly_revenue(self, ticker: str) -> List[Dict[str, Any]]:
        """Async wrapper for get_quarterly_revenue_sync."""
        return await self._run_sync(self.get_quarterly_revenue_sync, ticker)

    def get_forward_estimates_sync(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch forward-looking analyst estimates for revenue, EPS, and price targets (cached for TTL).

        Used to gauge whether forward guidance is strong/weak relative to
        the most recent quarter.
        """
        key = _cache_key("estimates", ticker)
        cached = _cache_get(key)
        if cached is not None:
            return cached

        result: Dict[str, Any] = {
            "revenue_estimates": [],
            "eps_estimates": [],
            "analyst_price_targets": None,
        }
        try:
            stock = self._get_ticker(ticker)

            # Revenue estimates (current quarter "0q", next quarter "+1q", etc.)
            try:
                rev_est = stock.revenue_estimate
                if rev_est is not None and not rev_est.empty:
                    for period, row in rev_est.iterrows():
                        result["revenue_estimates"].append({
                            "period": str(period),
                            "avg": _safe_float(row.get("avg"), default=None),
                            "low": _safe_float(row.get("low"), default=None),
                            "high": _safe_float(row.get("high"), default=None),
                            "year_ago_revenue": _safe_float(row.get("yearAgoRevenue"), default=None),
                            "growth": _safe_float(row.get("growth"), default=None),
                            "num_analysts": int(_safe_float(row.get("numberOfAnalysts"), default=0)),
                        })
            except Exception:
                pass

            # EPS estimates
            try:
                eps_est = stock.earnings_estimate
                if eps_est is not None and not eps_est.empty:
                    for period, row in eps_est.iterrows():
                        result["eps_estimates"].append({
                            "period": str(period),
                            "avg": _safe_float(row.get("avg"), default=None),
                            "low": _safe_float(row.get("low"), default=None),
                            "high": _safe_float(row.get("high"), default=None),
                            "year_ago_eps": _safe_float(row.get("yearAgoEps"), default=None),
                            "growth": _safe_float(row.get("growth"), default=None),
                            "num_analysts": int(_safe_float(row.get("numberOfAnalysts"), default=0)),
                        })
            except Exception:
                pass

            # Analyst price targets
            try:
                pt = stock.analyst_price_targets
                if pt is not None and isinstance(pt, dict):
                    result["analyst_price_targets"] = {
                        "current": _safe_float(pt.get("current"), default=None),
                        "high": _safe_float(pt.get("high"), default=None),
                        "low": _safe_float(pt.get("low"), default=None),
                        "mean": _safe_float(pt.get("mean"), default=None),
                        "median": _safe_float(pt.get("median"), default=None),
                    }
            except Exception:
                pass

            _cache_set(key, result)
            return result
        except Exception:
            return result

    async def get_forward_estimates(self, ticker: str) -> Dict[str, Any]:
        """Async wrapper for get_forward_estimates_sync."""
        return await self._run_sync(self.get_forward_estimates_sync, ticker)

    # ------------------------------------------------------------------
    # Earnings Reaction Insights — analysis engine
    # ------------------------------------------------------------------

    def build_earnings_insights(
        self,
        earnings_data: List[Dict[str, Any]],
        price_history: List[Dict[str, Any]],
        analyst_reactions: List[Dict[str, Any]],
        quarterly_revenue: List[Dict[str, Any]],
        fmp_guidance: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build per-quarter earnings reaction insights.

        For each historical earnings quarter, compute:
        - pre_earnings_runup (20-trading-day price change before earnings)
        - analyst reaction summary (upgrades / downgrades / maintains within 3 days)
        - revenue_growth_yoy from quarterly income statement
        - guidance_signal inferred from analyst behavior after the result
        - insight: a 1-2 sentence human-readable explanation
        - revenue_actual / capex / capex_pct_revenue (from FMP, if available)
        """
        if not earnings_data:
            return []

        # Build price-date helpers
        price_dates: List[str] = [p["date"] for p in price_history]
        price_by_date: Dict[str, int] = {p["date"]: i for i, p in enumerate(price_history)}

        # Revenue lookup: quarter-end date -> revenue
        rev_lookup: Dict[str, float] = {r["date"]: r["revenue"] for r in quarterly_revenue}

        # FMP data lookups (revenue actuals + CapEx by quarter-end date)
        fmp_rev = (fmp_guidance or {}).get("revenue_by_quarter", {})
        fmp_capex = (fmp_guidance or {}).get("capex_by_quarter", {})
        # Sorted quarter-end dates for matching to earnings announcement dates
        fmp_quarter_dates = sorted(fmp_rev.keys()) if fmp_rev else []

        insights: List[Dict[str, Any]] = []

        for e in earnings_data:
            rec: Dict[str, Any] = {
                "quarter": e.get("quarter", ""),
                "date": e.get("date", ""),
                "eps_actual": _safe_float(e.get("eps_actual"), default=None),
                "eps_estimate": _safe_float(e.get("eps_estimate"), default=None),
                "eps_beat_miss": e.get("beat_miss", "unknown"),
                "eps_surprise_pct": _safe_float(e.get("surprise_pct"), default=0),
                "price_reaction_1d": e.get("price_reaction_1d"),
                "price_reaction_5d": e.get("price_reaction_5d"),
                "pre_earnings_runup": None,
                "analyst_upgrades": 0,
                "analyst_downgrades": 0,
                "analyst_maintains": 0,
                "revenue_growth_yoy": None,
                "guidance_signal": "unknown",
                "insight": "",
                # FMP guidance fields (None if FMP unavailable)
                "revenue_actual": None,
                "revenue_yoy_pct": None,
                "capex": None,
                "capex_prev_quarter": None,
                "capex_qoq_pct": None,
                "capex_pct_revenue": None,
                "fcf": None,
            }

            earnings_date = e.get("date", "")
            if not earnings_date:
                insights.append(rec)
                continue

            # ── 0. FMP guidance data (revenue + CapEx) ──
            # Match earnings announcement date → quarter-end date.
            # e.g. earnings on 2026-02-06 maps to Q4 ending 2025-12-31.
            # The quarter-end is always 0-90 days before the announcement.
            if fmp_quarter_dates:
                matched_q = None
                try:
                    ear_dt_fmp = datetime.strptime(earnings_date, "%Y-%m-%d")
                    for qd in reversed(fmp_quarter_dates):
                        qd_dt = datetime.strptime(qd, "%Y-%m-%d")
                        diff_days = (ear_dt_fmp - qd_dt).days
                        if 0 <= diff_days <= 90:
                            matched_q = qd
                            break
                except Exception:
                    pass

                if matched_q:
                    # Revenue
                    fmp_rev_rec = fmp_rev.get(matched_q, {})
                    rev_actual = fmp_rev_rec.get("revenue")
                    if rev_actual:
                        rec["revenue_actual"] = round(rev_actual, 0)
                        # YoY: find same quarter previous year in FMP data
                        try:
                            mq_dt = datetime.strptime(matched_q, "%Y-%m-%d")
                            yoy_q = mq_dt.replace(year=mq_dt.year - 1).strftime("%Y-%m-%d")
                            yoy_rev_rec = fmp_rev.get(yoy_q, {})
                            yoy_rev = yoy_rev_rec.get("revenue")
                            if yoy_rev and yoy_rev > 0:
                                rec["revenue_yoy_pct"] = round(
                                    ((rev_actual - yoy_rev) / yoy_rev) * 100, 1
                                )
                        except Exception:
                            pass

                    # CapEx
                    capex_rec = fmp_capex.get(matched_q, {})
                    capex_val = capex_rec.get("capex")
                    if capex_val is not None:
                        rec["capex"] = round(capex_val, 0)
                        rec["fcf"] = round(capex_rec.get("fcf") or 0, 0)

                        # CapEx as % of revenue
                        if rev_actual and rev_actual > 0:
                            rec["capex_pct_revenue"] = round(
                                (capex_val / rev_actual) * 100, 1
                            )

                        # QoQ: find previous quarter's CapEx
                        try:
                            mq_dt = datetime.strptime(matched_q, "%Y-%m-%d")
                            # Approximate prev quarter: go back ~90 days
                            prev_candidates = [
                                d for d in fmp_quarter_dates
                                if d < matched_q
                            ]
                            if prev_candidates:
                                prev_q = prev_candidates[-1]
                                prev_capex = fmp_capex.get(prev_q, {}).get("capex")
                                if prev_capex is not None:
                                    rec["capex_prev_quarter"] = round(prev_capex, 0)
                                    if prev_capex > 0:
                                        rec["capex_qoq_pct"] = round(
                                            ((capex_val - prev_capex) / prev_capex) * 100, 1
                                        )
                        except Exception:
                            pass

            # ── 1. Pre-earnings run-up (20 trading days) ──
            nearest_idx = None
            for i, d in enumerate(price_dates):
                if d <= earnings_date:
                    nearest_idx = i
                else:
                    break
            if nearest_idx is not None and nearest_idx >= 20:
                pre_price = price_history[nearest_idx - 20]["close"]
                ear_price = price_history[nearest_idx]["close"]
                if pre_price > 0:
                    rec["pre_earnings_runup"] = round(
                        ((ear_price - pre_price) / pre_price) * 100, 1
                    )

            # ── 2. Analyst reactions within 3 calendar days ──
            try:
                ear_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
                window_end = (ear_dt + timedelta(days=4)).strftime("%Y-%m-%d")
                ups = downs = holds = 0
                for ar in analyst_reactions:
                    if earnings_date <= ar["date"] <= window_end:
                        action = ar.get("action", "").lower()
                        if action in ("upgrade", "up"):
                            ups += 1
                        elif action in ("downgrade", "down"):
                            downs += 1
                        else:
                            holds += 1
                rec["analyst_upgrades"] = ups
                rec["analyst_downgrades"] = downs
                rec["analyst_maintains"] = holds
            except Exception:
                pass

            # ── 3. Revenue growth YoY ──
            try:
                ear_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
                # Find revenue for the quarter that just ended (within 90 days before earnings)
                best_date = None
                best_rev = None
                for rd, rv in rev_lookup.items():
                    rd_dt = datetime.strptime(rd, "%Y-%m-%d")
                    diff = (ear_dt - rd_dt).days
                    if 0 <= diff <= 90:
                        if best_date is None or rd > best_date:
                            best_date = rd
                            best_rev = rv

                if best_rev and best_date:
                    # Find same quarter prior year
                    best_dt = datetime.strptime(best_date, "%Y-%m-%d")
                    yoy_target = best_dt.replace(year=best_dt.year - 1)
                    yoy_rev = None
                    for rd, rv in rev_lookup.items():
                        rd_dt = datetime.strptime(rd, "%Y-%m-%d")
                        if abs((rd_dt - yoy_target).days) <= 45:
                            yoy_rev = rv
                            break
                    if yoy_rev and yoy_rev > 0:
                        rec["revenue_growth_yoy"] = round(
                            ((best_rev - yoy_rev) / yoy_rev) * 100, 1
                        )
            except Exception:
                pass

            # ── 4. Guidance signal ──
            total_a = rec["analyst_upgrades"] + rec["analyst_downgrades"] + rec["analyst_maintains"]
            if rec["eps_beat_miss"] == "beat":
                if rec["analyst_downgrades"] > rec["analyst_upgrades"]:
                    rec["guidance_signal"] = "weak"
                elif total_a > 0 and rec["analyst_upgrades"] == 0:
                    rec["guidance_signal"] = "cautious"
                elif rec["analyst_upgrades"] > 2:
                    rec["guidance_signal"] = "strong"
                else:
                    rec["guidance_signal"] = "inline"
            elif rec["eps_beat_miss"] == "miss":
                if rec["analyst_downgrades"] > 0:
                    rec["guidance_signal"] = "weak"
                else:
                    rec["guidance_signal"] = "inline"
            else:
                rec["guidance_signal"] = "inline"

            # ── 5. Generate insight text ──
            rec["insight"] = self._generate_insight_text(rec)
            insights.append(rec)

        return insights

    @staticmethod
    def _generate_insight_text(rec: Dict[str, Any]) -> str:
        """Generate a concise human-readable insight for one earnings quarter."""
        parts: List[str] = []
        bm = rec["eps_beat_miss"]
        surprise = abs(rec.get("eps_surprise_pct") or 0)
        r1d = rec.get("price_reaction_1d")
        runup = rec.get("pre_earnings_runup")
        ups = rec["analyst_upgrades"]
        downs = rec["analyst_downgrades"]
        maint = rec["analyst_maintains"]
        total_a = ups + downs + maint
        guidance = rec["guidance_signal"]

        # Core result + reaction
        if bm == "beat" and r1d is not None and r1d < -1:
            parts.append(f"Beat EPS by +{surprise:.1f}% but stock fell {r1d:.1f}%.")
        elif bm == "beat" and r1d is not None and r1d > 1:
            parts.append(f"Beat EPS by +{surprise:.1f}% and stock rose +{r1d:.1f}%.")
        elif bm == "beat":
            parts.append(f"Beat EPS by +{surprise:.1f}%.")
        elif bm == "miss" and r1d is not None:
            parts.append(f"Missed EPS by {surprise:.1f}%. Stock moved {r1d:+.1f}%.")
        else:
            parts.append("EPS came in roughly inline with estimates.")

        # Analyst reactions
        if total_a > 0:
            if ups == 0 and downs == 0:
                parts.append(f"All {maint} analysts maintained ratings.")
            elif ups > downs:
                parts.append(f"{ups} upgrades vs {downs} downgrades from {total_a} analysts.")
            elif downs > ups:
                parts.append(f"{downs} downgrades vs {ups} upgrades — analysts cautious.")

        # Pre-earnings run-up
        if runup is not None and runup > 10:
            parts.append(f"Stock ran up +{runup:.1f}% before earnings, beat may have been priced in.")
        elif runup is not None and runup > 5:
            parts.append(f"Modest pre-earnings run-up of +{runup:.1f}%.")

        # CapEx / investment story (from FMP)
        capex_qoq = rec.get("capex_qoq_pct")
        capex_pct_rev = rec.get("capex_pct_revenue")
        if capex_qoq is not None and abs(capex_qoq) > 10:
            if capex_qoq > 0:
                parts.append(f"CapEx surged +{capex_qoq:.0f}% QoQ — heavy investment cycle.")
            else:
                parts.append(f"CapEx dropped {capex_qoq:.0f}% QoQ — pulling back on spending.")
        elif capex_pct_rev is not None and capex_pct_rev > 15:
            parts.append(f"CapEx is {capex_pct_rev:.0f}% of revenue — high capital intensity.")

        # Guidance signal
        if guidance == "weak":
            parts.append("Forward guidance appears weak based on analyst reactions.")
        elif guidance == "cautious":
            parts.append("No upgrades despite the beat suggests cautious forward outlook.")
        elif guidance == "strong":
            parts.append("Multiple upgrades signal strong forward guidance.")

        return " ".join(parts) if parts else ""

    def build_forward_guidance_summary(
        self,
        forward_estimates: Dict[str, Any],
        current_price: float = 0,
    ) -> Dict[str, Any]:
        """
        Build a forward-looking guidance summary from analyst estimates.

        Computes growth trend (accelerating/decelerating/stable), price-target
        upside, estimate ranges (low/avg/high), and a human-readable summary.
        """
        result: Dict[str, Any] = {
            "current_q_eps_growth": None,
            "next_q_eps_growth": None,
            "current_q_rev_growth": None,
            "next_q_rev_growth": None,
            "growth_trend": "unknown",
            "analyst_price_target_mean": None,
            "analyst_price_target_upside_pct": None,
            "num_analysts": None,
            # Full estimate ranges for analyst vs company guidance comparison
            "estimate_ranges": [],
            "consensus_confidence": "unknown",
            "price_targets": None,
            "summary": "No forward estimate data available.",
        }

        rev_ests = forward_estimates.get("revenue_estimates", [])
        eps_ests = forward_estimates.get("eps_estimates", [])
        pt = forward_estimates.get("analyst_price_targets")

        # ── Build estimate ranges (analyst low → consensus → high) ──
        estimate_ranges: List[Dict[str, Any]] = []

        for i, label in enumerate(["Current Q", "Next Q"]):
            row: Dict[str, Any] = {"period": label}

            # EPS range
            if i < len(eps_ests):
                e = eps_ests[i]
                row["eps_low"] = _safe_float(e.get("low"), default=None)
                row["eps_avg"] = _safe_float(e.get("avg"), default=None)
                row["eps_high"] = _safe_float(e.get("high"), default=None)
                row["eps_year_ago"] = _safe_float(e.get("year_ago_eps"), default=None)
                row["eps_growth"] = _safe_float(e.get("growth"), default=None)
                row["eps_num_analysts"] = int(_safe_float(e.get("num_analysts"), default=0))

            # Revenue range
            if i < len(rev_ests):
                r = rev_ests[i]
                row["rev_low"] = _safe_float(r.get("low"), default=None)
                row["rev_avg"] = _safe_float(r.get("avg"), default=None)
                row["rev_high"] = _safe_float(r.get("high"), default=None)
                row["rev_year_ago"] = _safe_float(r.get("year_ago_revenue"), default=None)
                row["rev_growth"] = _safe_float(r.get("growth"), default=None)
                row["rev_num_analysts"] = int(_safe_float(r.get("num_analysts"), default=0))

            if len(row) > 1:  # More than just "period"
                estimate_ranges.append(row)

        result["estimate_ranges"] = estimate_ranges

        # ── Consensus confidence (EPS spread relative to avg) ──
        if estimate_ranges and estimate_ranges[0].get("eps_avg"):
            cq = estimate_ranges[0]
            eps_low = cq.get("eps_low")
            eps_high = cq.get("eps_high")
            eps_avg = cq.get("eps_avg")
            if eps_low is not None and eps_high is not None and eps_avg and eps_avg != 0:
                spread_pct = abs(eps_high - eps_low) / abs(eps_avg) * 100
                if spread_pct < 10:
                    result["consensus_confidence"] = "high"
                elif spread_pct < 25:
                    result["consensus_confidence"] = "moderate"
                else:
                    result["consensus_confidence"] = "low"

        # ── Price target range ──
        if pt:
            result["price_targets"] = {
                "current": _safe_float(pt.get("current"), default=None),
                "low": _safe_float(pt.get("low"), default=None),
                "mean": _safe_float(pt.get("mean"), default=None),
                "median": _safe_float(pt.get("median"), default=None),
                "high": _safe_float(pt.get("high"), default=None),
            }

        # Revenue growth: current quarter vs next quarter
        if len(rev_ests) >= 2:
            cg = _safe_float(rev_ests[0].get("growth"), default=None)
            ng = _safe_float(rev_ests[1].get("growth"), default=None)
            if cg is not None:
                result["current_q_rev_growth"] = round(cg * 100, 1)
            if ng is not None:
                result["next_q_rev_growth"] = round(ng * 100, 1)

        # EPS growth: current quarter vs next quarter
        if len(eps_ests) >= 2:
            cg = _safe_float(eps_ests[0].get("growth"), default=None)
            ng = _safe_float(eps_ests[1].get("growth"), default=None)
            if cg is not None:
                result["current_q_eps_growth"] = round(cg * 100, 1)
            if ng is not None:
                result["next_q_eps_growth"] = round(ng * 100, 1)
            result["num_analysts"] = eps_ests[0].get("num_analysts")

        # Growth trend
        curr_rev = result["current_q_rev_growth"]
        next_rev = result["next_q_rev_growth"]
        if curr_rev is not None and next_rev is not None:
            diff = next_rev - curr_rev
            if diff > 2:
                result["growth_trend"] = "accelerating"
            elif diff < -2:
                result["growth_trend"] = "decelerating"
            else:
                result["growth_trend"] = "stable"

        # Price target (backward compat)
        if pt:
            mean_pt = pt.get("mean")
            result["analyst_price_target_mean"] = _safe_float(mean_pt, default=None)
            if mean_pt and current_price > 0:
                upside = ((mean_pt - current_price) / current_price) * 100
                result["analyst_price_target_upside_pct"] = round(upside, 1)

        # Summary text
        parts: List[str] = []
        if result["growth_trend"] == "decelerating" and curr_rev is not None and next_rev is not None:
            parts.append(f"Revenue growth expected to decelerate from {curr_rev:.1f}% to {next_rev:.1f}%.")
        elif result["growth_trend"] == "accelerating" and curr_rev is not None and next_rev is not None:
            parts.append(f"Revenue growth expected to accelerate from {curr_rev:.1f}% to {next_rev:.1f}%.")
        elif result["growth_trend"] == "stable" and curr_rev is not None:
            parts.append(f"Revenue growth expected to remain stable around {curr_rev:.1f}%.")

        if result["analyst_price_target_mean"] and result["analyst_price_target_upside_pct"] is not None:
            upside = result["analyst_price_target_upside_pct"]
            direction = "upside" if upside > 0 else "downside"
            parts.append(f"Mean analyst PT ${result['analyst_price_target_mean']:.0f} ({abs(upside):.1f}% {direction}).")

        conf = result["consensus_confidence"]
        if conf == "high":
            parts.append("Tight analyst spread signals high confidence in guidance.")
        elif conf == "low":
            parts.append("Wide analyst spread signals uncertainty around company guidance.")

        if result["num_analysts"]:
            parts.append(f"{result['num_analysts']} analysts covering.")

        result["summary"] = " ".join(parts) if parts else "Limited forward estimate data available."
        return result

    def _calculate_revenue_cagr(self, raw_data: Dict[str, Any]) -> float:
        """Calculate 5-year revenue CAGR from income statements."""
        try:
            income_stmt = raw_data.get("income_statements")
            if income_stmt is None or income_stmt.empty:
                return 0
            
            # Get total revenue row
            if "Total Revenue" in income_stmt.index:
                revenues = income_stmt.loc["Total Revenue"]
            elif "Revenue" in income_stmt.index:
                revenues = income_stmt.loc["Revenue"]
            else:
                return 0
            
            # Get first and last available values
            revenues = revenues.dropna()
            if len(revenues) < 2:
                return 0
            
            end_value = float(revenues.iloc[0])  # Most recent
            start_value = float(revenues.iloc[-1])  # Oldest
            years = len(revenues) - 1
            
            if start_value <= 0 or end_value <= 0 or years <= 0:
                return 0
            
            return (end_value / start_value) ** (1 / years) - 1
        except:
            return 0
    
    def _calculate_interest_coverage(self, raw_data: Dict[str, Any]) -> float:
        """Calculate interest coverage ratio."""
        try:
            income_stmt = raw_data.get("income_statements")
            if income_stmt is None or income_stmt.empty:
                return 0
            
            # Get EBIT and interest expense
            ebit = None
            interest = None
            
            if "EBIT" in income_stmt.index:
                ebit = float(income_stmt.loc["EBIT"].iloc[0])
            elif "Operating Income" in income_stmt.index:
                ebit = float(income_stmt.loc["Operating Income"].iloc[0])
            
            if "Interest Expense" in income_stmt.index:
                interest = abs(float(income_stmt.loc["Interest Expense"].iloc[0]))
            
            if ebit and interest and interest > 0:
                return ebit / interest
            return 0
        except:
            return 0
    
    def extract_anchor_year_data(
        self,
        raw_data: Dict[str, Any],
        anchor_years: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract financial data for specific anchor years.
        
        Used for time-travel analysis comparing company
        at different points in time.
        
        Args:
            raw_data: Raw data from get_company_data()
            anchor_years: List of years to extract (e.g., [2014, 2019, 2024])
            
        Returns:
            Dictionary mapping year to financial data
        """
        result = {}
        
        income_stmt = raw_data.get("income_statements")
        balance_sheet = raw_data.get("balance_sheets")
        cash_flow = raw_data.get("cash_flow_statements")
        
        for year in anchor_years:
            year_data = {
                "income": self._extract_year_from_df(income_stmt, year),
                "balance": self._extract_year_from_df(balance_sheet, year),
                "cash_flow": self._extract_year_from_df(cash_flow, year),
            }
            
            # Only include if we have at least some data
            if any(v is not None for v in year_data.values()):
                result[year] = year_data
        
        return result
    
    def _extract_year_from_df(self, df, year: int) -> Optional[Dict[str, Any]]:
        """Extract data for a specific year from a DataFrame."""
        if df is None or df.empty:
            return None
        
        try:
            for col in df.columns:
                if hasattr(col, 'year') and col.year == year:
                    return df[col].to_dict()
            return None
        except:
            return None
