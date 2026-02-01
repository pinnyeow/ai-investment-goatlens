"""
GOATlens - Data Sources

Pluggable data source adapters for financial data.
Primary: FMP API (Financial Modeling Prep)
"""

from .fmp import FMPClient, FMPError

__all__ = ["FMPClient", "FMPError"]
