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

No API key required!
"""

import yfinance as yf
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor


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
        """Synchronously fetch all company data."""
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
            
            return {
                "ticker": ticker,
                "info": info,
                "income_statements": income_stmt,
                "income_quarterly": income_quarterly,
                "balance_sheets": balance_sheet,
                "cash_flow_statements": cash_flow,
                "fetched_at": datetime.now().isoformat(),
            }
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
        
        # Safe getter with default
        def get_val(key, default=0):
            val = info.get(key)
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
