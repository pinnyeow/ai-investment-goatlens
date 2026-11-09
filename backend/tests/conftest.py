import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUALITY_COMPANY = {
    "current_price": 100.0, "pe_ratio": 12.0, "pb_ratio": 1.2, "eps": 8.0, "book_value_per_share": 80.0,
    "roe": 0.25, "profit_margin": 0.20, "gross_margin": 0.55, "operating_margin": 0.30,
    "debt_to_equity": 0.3, "current_ratio": 2.5, "free_cash_flow": 5e9, "earnings_growth": 0.22,
    "earnings_turnaround": None, "revenue_growth": 0.12, "margin_trend": 0.03, "capex_to_revenue": 0.03,
    "roic": 0.22, "interest_coverage": 20.0, "debt_growth_3y": 0.05, "current_assets": 5e10,
    "total_liabilities": 3e10, "shares_outstanding": 1e9, "institutional_ownership": 0.45,
    "insider_ownership": 0.12, "dividend_yield": 0.02, "consecutive_dividend_years": 25.0, "beta": 1.0,
    "volatility_252d": 0.20, "realized_total_return_252d": 0.25, "correlation_sp500": 0.4,
}


def company(metrics=None, **overrides):
    """A fetched-company dict like lens.data.fetch_company returns, without the network."""
    base = {
        "ticker": "TEST", "company_name": "Test Corp", "sector": "Technology", "industry": "Software",
        "currency": "USD", "is_financial": False, "not_applicable": [],
        "metrics": dict(QUALITY_COMPANY if metrics is None else metrics),
        "statement_years": 4, "fiscal_year_end": "2025-12-31", "price_as_of": "2026-11-06",
        "prices": [["2026-11-05", 99, 101, 98, 100, 100, 1000], ["2026-11-06", 100, 102, 99, 101, 101, 1200]],
        "earnings_history": [
            {"quarter": "Q3 2026", "date": "2026-11-05", "eps_actual": 2.1, "eps_estimate": 2.0,
             "eps_surprise": 0.1, "surprise_pct": 5.0, "beat_miss": "beat"},
        ],
        "next_earnings_date": None,
    }
    base.update(overrides)
    return base
