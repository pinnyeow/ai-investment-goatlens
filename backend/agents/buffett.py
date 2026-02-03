"""
Warren Buffett Investment Agent

Key Philosophy:
- "Buy wonderful companies at fair prices"
- Focus on durable competitive advantages (moats)
- Long-term ownership mentality
- Emphasis on quality management and owner earnings

Key Metrics:
- ROE (Return on Equity) > 15%
- Profit margins (consistency and improvement)
- Debt levels (conservative use of leverage)
- Owner earnings (FCF adjusted for maintenance capex)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class BuffettMetrics:
    """Key metrics for Buffett-style analysis."""
    roe: float
    profit_margin: float
    debt_to_equity: float
    owner_earnings: float
    earnings_consistency: float  # Variance in earnings over years
    moat_strength: str


class BuffettAgent:
    """
    Warren Buffett Investment Analysis Agent.
    
    Evaluates companies based on:
    1. Durable competitive advantage (moat)
    2. Consistent earnings power
    3. Return on equity
    4. Conservative debt levels
    5. Quality of management
    """
    
    name = "Warren Buffett"
    style = "Value Investing with Quality Focus"
    
    # Buffett's preferred thresholds
    MIN_ROE = 0.15  # 15%
    MAX_DEBT_TO_EQUITY = 0.5
    MIN_PROFIT_MARGIN = 0.10  # 10%
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize Buffett agent.
        
        Args:
            llm_client: Optional LLM client for narrative analysis
        """
        self.llm_client = llm_client
    
    async def analyze(
        self,
        ticker: str,
        financials: Dict[str, Any],
        anchor_years: List[int],
    ) -> Dict[str, Any]:
        """
        Perform Buffett-style analysis on a company.
        
        Args:
            ticker: Stock ticker symbol
            financials: Historical financial data
            anchor_years: Years to focus analysis on
            
        Returns:
            Analysis result with verdict, score, and insights
        """
        metrics = self._calculate_metrics(financials)
        moat_analysis = self._assess_moat(metrics)
        management_quality = self._assess_management(financials)
        
        # Calculate Buffett score
        score = self._calculate_score(metrics, moat_analysis, management_quality)
        
        # Generate LLM-powered insights if client available
        if self.llm_client:
            insights = await self._generate_llm_insights(ticker, metrics, score)
        else:
            insights = self._generate_insights(metrics, moat_analysis)
        
        concerns = self._identify_concerns(metrics)
        
        return {
            "agent": self.name,
            "style": self.style,
            "ticker": ticker,
            "score": score,
            "verdict": self._score_to_verdict(score),
            "metrics": metrics.__dict__,
            "moat_analysis": moat_analysis,
            "insights": insights,
            "concerns": concerns,
            "buffett_quote": self._get_relevant_quote(score),
        }
    
    def _calculate_metrics(self, financials: Dict[str, Any]) -> BuffettMetrics:
        """Extract and calculate Buffett-relevant metrics."""
        roe = financials.get("roe", 0)
        profit_margin = financials.get("profit_margin", 0)
        debt_to_equity = financials.get("debt_to_equity", 0)
        
        # Determine moat strength from margins and ROE
        if roe >= 0.20 and profit_margin >= 0.15:
            moat = "Strong"
        elif roe >= 0.15 and profit_margin >= 0.10:
            moat = "Moderate"
        else:
            moat = "Weak"
        
        return BuffettMetrics(
            roe=roe,
            profit_margin=profit_margin,
            debt_to_equity=debt_to_equity,
            owner_earnings=financials.get("free_cash_flow", 0),
            earnings_consistency=0.0,
            moat_strength=moat,
        )
    
    def _assess_moat(self, metrics: BuffettMetrics) -> Dict[str, Any]:
        """Assess the durability of competitive advantage."""
        if metrics.moat_strength == "Strong":
            durability = "High"
            analysis = "Strong pricing power and high returns suggest durable competitive advantage"
        elif metrics.moat_strength == "Moderate":
            durability = "Medium"
            analysis = "Decent returns but moat durability needs monitoring"
        else:
            durability = "Low"
            analysis = "Weak margins suggest limited pricing power"
        
        return {
            "moat_type": "Financial Moat",
            "durability": durability,
            "strength": metrics.moat_strength,
            "analysis": analysis,
        }
    
    def _assess_management(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """Assess management quality."""
        return {
            "capital_allocation": "Unknown",
            "shareholder_focus": "Unknown",
            "insider_ownership": financials.get("insider_ownership", 0),
        }
    
    def _calculate_score(
        self,
        metrics: BuffettMetrics,
        moat_analysis: Dict[str, Any],
        management_quality: Dict[str, Any],
    ) -> float:
        """Calculate overall Buffett score (-100 to +100)."""
        score = 0.0
        
        # ROE contribution (max 25 points)
        if metrics.roe >= self.MIN_ROE:
            score += min(25, (metrics.roe / self.MIN_ROE) * 15)
        else:
            score -= 15
        
        # Profit margin contribution (max 20 points)
        if metrics.profit_margin >= self.MIN_PROFIT_MARGIN:
            score += min(20, (metrics.profit_margin / self.MIN_PROFIT_MARGIN) * 10)
        else:
            score -= 10
        
        # Debt level contribution (max 20 points)
        if metrics.debt_to_equity <= self.MAX_DEBT_TO_EQUITY:
            score += 20
        elif metrics.debt_to_equity <= 1.0:
            score += 10
        else:
            score -= 20
        
        # Clamp to valid range
        return max(-100, min(100, score))
    
    def _score_to_verdict(self, score: float) -> str:
        """Convert numeric score to investment verdict."""
        if score >= 60:
            return "strong_buy"
        elif score >= 20:
            return "buy"
        elif score >= -20:
            return "hold"
        elif score >= -60:
            return "sell"
        else:
            return "strong_sell"
    
    async def _generate_llm_insights(
        self,
        ticker: str,
        metrics: BuffettMetrics,
        score: float,
    ) -> List[str]:
        """Generate LLM-powered insights using Buffett's voice."""
        prompt = f"""Analyze {ticker} with these metrics:
- ROE: {metrics.roe:.1%}
- Profit Margin: {metrics.profit_margin:.1%}
- Debt/Equity: {metrics.debt_to_equity:.2f}
- Moat: {metrics.moat_strength}
- Score: {score:.0f}/100

Provide 3 key insights in Warren Buffett's voice. Focus on moat quality, management efficiency, and whether this is a wonderful business at a fair price."""

        try:
            response = await self.llm_client.analyze(prompt, persona="Warren Buffett")
            # Split into list of insights
            insights = [line.strip() for line in response.split("\n") if line.strip() and not line.strip().startswith("#")]
            return insights[:3] if insights else self._generate_insights(metrics, {"strength": metrics.moat_strength})
        except Exception:
            return self._generate_insights(metrics, {"strength": metrics.moat_strength})
    
    def _generate_insights(
        self,
        metrics: BuffettMetrics,
        moat_analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate key insights from analysis (fallback)."""
        insights = []
        
        if metrics.roe >= self.MIN_ROE:
            insights.append(f"Strong ROE of {metrics.roe:.1%} indicates efficient capital deployment")
        
        if metrics.profit_margin >= self.MIN_PROFIT_MARGIN:
            insights.append(f"Healthy profit margin of {metrics.profit_margin:.1%} suggests pricing power")
        
        if metrics.debt_to_equity <= self.MAX_DEBT_TO_EQUITY:
            insights.append("Conservative debt levels provide margin of safety")
        
        return insights
    
    def _identify_concerns(self, metrics: BuffettMetrics) -> List[str]:
        """Identify potential concerns from Buffett's perspective."""
        concerns = []
        
        if metrics.roe < self.MIN_ROE:
            concerns.append(f"ROE of {metrics.roe:.1%} below Buffett's 15% threshold")
        
        if metrics.debt_to_equity > 1.0:
            concerns.append(f"High debt-to-equity ratio of {metrics.debt_to_equity:.2f}")
        
        if metrics.profit_margin < self.MIN_PROFIT_MARGIN:
            concerns.append("Thin profit margins may indicate weak competitive position")
        
        return concerns
    
    def _get_relevant_quote(self, score: float) -> str:
        """Get a relevant Buffett quote based on the analysis."""
        if score >= 60:
            return "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price."
        elif score >= 20:
            return "Price is what you pay. Value is what you get."
        elif score >= -20:
            return "The stock market is designed to transfer money from the active to the patient."
        else:
            return "Rule No. 1: Never lose money. Rule No. 2: Never forget Rule No. 1."
