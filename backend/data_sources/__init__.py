"""
GOATlens - Data Sources

Data source adapters with explicit roles.
SEC supplies authoritative filing fundamentals; Yahoo supplies best-effort market data.
"""

from .yahoo import YahooFinanceClient, YahooFinanceError, FinancialData

from .news import NewsClient, NewsError

__all__ = [
    "YahooFinanceClient",
    "YahooFinanceError",
    "FinancialData",

    "NewsClient",
    "NewsError",
]
