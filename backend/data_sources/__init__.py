"""
GOATlens - Data Sources

Pluggable data source adapters for financial data.
Primary: Yahoo Finance (free, no API key required)
Fallback: FMP API (Financial Modeling Prep)
"""

from .yahoo import YahooFinanceClient, YahooFinanceError, FinancialData
from .fmp import FMPClient, FMPError
from .news import NewsClient, NewsError

__all__ = [
    "YahooFinanceClient",
    "YahooFinanceError",
    "FinancialData",
    "FMPClient",
    "FMPError",
    "NewsClient",
    "NewsError",
]
