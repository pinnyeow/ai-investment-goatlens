"""
Financial Modeling Prep (FMP) API Client

Primary data source for GOATlens.
FMP provides comprehensive financial data including:
- Income statements
- Balance sheets
- Cash flow statements
- Key ratios
- Historical prices
- Earnings data

Free tier: 250 API calls/day
"""

import os
import httpx
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio


class FMPError(Exception):
    """Exception raised for FMP API errors."""
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


class FMPClient:
    """
    Async client for Financial Modeling Prep API.
    
    Usage:
        client = FMPClient(api_key="your_key")
        data = await client.get_company_data("AAPL", years=10)
    """
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP client.
        
        Args:
            api_key: FMP API key. If not provided, reads from FMP_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise FMPError("FMP API key required. Set FMP_API_KEY environment variable.")
        
        self._client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """
        Make authenticated request to FMP API.
        
        Args:
            endpoint: API endpoint path
            params: Additional query parameters
            
        Returns:
            JSON response data
        """
        params = params or {}
        params["apikey"] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API error messages
            if isinstance(data, dict) and "Error Message" in data:
                raise FMPError(data["Error Message"])
            
            return data
            
        except httpx.HTTPStatusError as e:
            raise FMPError(f"HTTP error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise FMPError(f"Request failed: {str(e)}") from e
    
    async def get_company_profile(self, ticker: str) -> Dict[str, Any]:
        """
        Get company profile and basic info.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company profile data
        """
        data = await self._request(f"profile/{ticker}")
        if data and len(data) > 0:
            return data[0]
        raise FMPError(f"No data found for ticker: {ticker}")
    
    async def get_income_statements(
        self,
        ticker: str,
        limit: int = 10,
        period: str = "annual",
    ) -> List[Dict[str, Any]]:
        """
        Get historical income statements.
        
        Args:
            ticker: Stock ticker symbol
            limit: Number of years to fetch
            period: "annual" or "quarter"
            
        Returns:
            List of income statements
        """
        return await self._request(
            f"income-statement/{ticker}",
            {"limit": limit, "period": period},
        )
    
    async def get_balance_sheets(
        self,
        ticker: str,
        limit: int = 10,
        period: str = "annual",
    ) -> List[Dict[str, Any]]:
        """
        Get historical balance sheets.
        
        Args:
            ticker: Stock ticker symbol
            limit: Number of years to fetch
            period: "annual" or "quarter"
            
        Returns:
            List of balance sheets
        """
        return await self._request(
            f"balance-sheet-statement/{ticker}",
            {"limit": limit, "period": period},
        )
    
    async def get_cash_flow_statements(
        self,
        ticker: str,
        limit: int = 10,
        period: str = "annual",
    ) -> List[Dict[str, Any]]:
        """
        Get historical cash flow statements.
        
        Args:
            ticker: Stock ticker symbol
            limit: Number of years to fetch
            period: "annual" or "quarter"
            
        Returns:
            List of cash flow statements
        """
        return await self._request(
            f"cash-flow-statement/{ticker}",
            {"limit": limit, "period": period},
        )
    
    async def get_key_metrics(
        self,
        ticker: str,
        limit: int = 10,
        period: str = "annual",
    ) -> List[Dict[str, Any]]:
        """
        Get key financial metrics (ratios, per-share data).
        
        Args:
            ticker: Stock ticker symbol
            limit: Number of years to fetch
            period: "annual" or "quarter"
            
        Returns:
            List of key metrics by year
        """
        return await self._request(
            f"key-metrics/{ticker}",
            {"limit": limit, "period": period},
        )
    
    async def get_financial_ratios(
        self,
        ticker: str,
        limit: int = 10,
        period: str = "annual",
    ) -> List[Dict[str, Any]]:
        """
        Get financial ratios (profitability, liquidity, leverage).
        
        Args:
            ticker: Stock ticker symbol
            limit: Number of years to fetch
            period: "annual" or "quarter"
            
        Returns:
            List of ratios by year
        """
        return await self._request(
            f"ratios/{ticker}",
            {"limit": limit, "period": period},
        )
    
    async def get_historical_prices(
        self,
        ticker: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get historical stock prices.
        
        Args:
            ticker: Stock ticker symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of historical prices
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        data = await self._request(f"historical-price-full/{ticker}", params)
        return data.get("historical", [])
    
    async def get_earnings_history(
        self,
        ticker: str,
        limit: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Get historical earnings (actual vs estimates).
        
        Args:
            ticker: Stock ticker symbol
            limit: Number of quarters to fetch
            
        Returns:
            List of earnings data
        """
        return await self._request(
            f"historical/earning_calendar/{ticker}",
            {"limit": limit},
        )
    
    async def get_company_data(
        self,
        ticker: str,
        years: int = 10,
    ) -> Dict[str, Any]:
        """
        Fetch all company data in parallel.
        
        This is the main entry point for getting comprehensive
        company data for GOAT analysis.
        
        Args:
            ticker: Stock ticker symbol
            years: Number of years of historical data
            
        Returns:
            Comprehensive company data dictionary
        """
        # Fetch all data in parallel
        profile, income, balance, cash_flow, metrics, ratios = await asyncio.gather(
            self.get_company_profile(ticker),
            self.get_income_statements(ticker, limit=years),
            self.get_balance_sheets(ticker, limit=years),
            self.get_cash_flow_statements(ticker, limit=years),
            self.get_key_metrics(ticker, limit=years),
            self.get_financial_ratios(ticker, limit=years),
        )
        
        return {
            "ticker": ticker,
            "profile": profile,
            "income_statements": income,
            "balance_sheets": balance,
            "cash_flow_statements": cash_flow,
            "key_metrics": metrics,
            "ratios": ratios,
            "fetched_at": datetime.now().isoformat(),
        }
    
    def normalize_data(self, raw_data: Dict[str, Any]) -> FinancialData:
        """
        Normalize raw FMP data into standardized FinancialData structure.
        
        Args:
            raw_data: Raw data from get_company_data()
            
        Returns:
            Normalized FinancialData object
        """
        profile = raw_data["profile"]
        latest_metrics = raw_data["key_metrics"][0] if raw_data["key_metrics"] else {}
        latest_ratios = raw_data["ratios"][0] if raw_data["ratios"] else {}
        latest_income = raw_data["income_statements"][0] if raw_data["income_statements"] else {}
        
        return FinancialData(
            ticker=raw_data["ticker"],
            company_name=profile.get("companyName", ""),
            sector=profile.get("sector", ""),
            industry=profile.get("industry", ""),
            
            # Valuation
            market_cap=profile.get("mktCap", 0),
            current_price=profile.get("price", 0),
            pe_ratio=latest_ratios.get("priceEarningsRatio", 0) or 0,
            pb_ratio=latest_ratios.get("priceToBookRatio", 0) or 0,
            ps_ratio=latest_ratios.get("priceToSalesRatio", 0) or 0,
            peg_ratio=latest_ratios.get("priceEarningsToGrowthRatio", 0) or 0,
            
            # Profitability
            gross_margin=latest_ratios.get("grossProfitMargin", 0) or 0,
            operating_margin=latest_ratios.get("operatingProfitMargin", 0) or 0,
            profit_margin=latest_ratios.get("netProfitMargin", 0) or 0,
            roe=latest_ratios.get("returnOnEquity", 0) or 0,
            roa=latest_ratios.get("returnOnAssets", 0) or 0,
            roic=latest_ratios.get("returnOnCapitalEmployed", 0) or 0,
            
            # Growth
            revenue_growth=latest_metrics.get("revenueGrowth", 0) or 0,
            earnings_growth=latest_metrics.get("netIncomeGrowth", 0) or 0,
            revenue_growth_5y=self._calculate_cagr(
                raw_data["income_statements"], "revenue", 5
            ),
            
            # Financial Health
            current_ratio=latest_ratios.get("currentRatio", 0) or 0,
            quick_ratio=latest_ratios.get("quickRatio", 0) or 0,
            debt_to_equity=latest_ratios.get("debtEquityRatio", 0) or 0,
            interest_coverage=latest_ratios.get("interestCoverage", 0) or 0,
            
            # Per Share Data
            eps=latest_income.get("eps", 0) or 0,
            book_value_per_share=latest_metrics.get("bookValuePerShare", 0) or 0,
            free_cash_flow_per_share=latest_metrics.get("freeCashFlowPerShare", 0) or 0,
            dividend_yield=profile.get("lastDiv", 0) / profile.get("price", 1) if profile.get("price") else 0,
            
            # Risk
            beta=profile.get("beta", 1.0) or 1.0,
            volatility_252d=0,  # Would need price data to calculate
            
            # Ownership
            insider_ownership=0,  # Requires separate API call
            institutional_ownership=0,  # Requires separate API call
        )
    
    def _calculate_cagr(
        self,
        statements: List[Dict[str, Any]],
        field: str,
        years: int,
    ) -> float:
        """
        Calculate Compound Annual Growth Rate.
        
        Args:
            statements: List of financial statements
            field: Field name to calculate CAGR for
            years: Number of years for CAGR calculation
            
        Returns:
            CAGR as decimal (e.g., 0.15 for 15%)
        """
        if len(statements) < years + 1:
            return 0
        
        end_value = statements[0].get(field, 0)
        start_value = statements[years].get(field, 0)
        
        if start_value <= 0 or end_value <= 0:
            return 0
        
        return (end_value / start_value) ** (1 / years) - 1
    
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
        
        for year in anchor_years:
            year_data = {
                "income": self._find_year_data(raw_data["income_statements"], year),
                "balance": self._find_year_data(raw_data["balance_sheets"], year),
                "cash_flow": self._find_year_data(raw_data["cash_flow_statements"], year),
                "metrics": self._find_year_data(raw_data["key_metrics"], year),
                "ratios": self._find_year_data(raw_data["ratios"], year),
            }
            
            if any(v is not None for v in year_data.values()):
                result[year] = year_data
        
        return result
    
    def _find_year_data(
        self,
        statements: List[Dict[str, Any]],
        year: int,
    ) -> Optional[Dict[str, Any]]:
        """Find statement data for a specific year."""
        for statement in statements:
            date_str = statement.get("date", "") or statement.get("calendarYear", "")
            if str(year) in str(date_str):
                return statement
        return None
