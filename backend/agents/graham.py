"""
Benjamin Graham Investment Agent

Key Philosophy:
- "Father of Value Investing"
- Margin of Safety - buy below intrinsic value
- Mr. Market metaphor - market is emotional, exploit it
- Net-Net investing - buy below liquidation value

Key Metrics:
- P/E ratio < 15
- P/B ratio < 1.5
- Current ratio > 2
- Dividend history (consistent payer)
- Graham Number = sqrt(22.5 × EPS × BVPS)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import math


@dataclass
class GrahamMetrics:
    """Key metrics for Graham-style analysis."""
    pe_ratio: float
    pb_ratio: float
    current_ratio: float
    dividend_yield: float
    dividend_years: int
    graham_number: float
    current_price: float
    net_net_value: float  # NCAV per share


class GrahamAgent:
    """
    Benjamin Graham Investment Analysis Agent.
    
    Evaluates companies based on:
    1. Margin of Safety (intrinsic value vs price)
    2. Balance sheet strength
    3. Earnings stability
    4. Dividend record
    5. Valuation ratios
    """
    
    name = "Benjamin Graham"
    style = "Deep Value / Margin of Safety"
    
    # Graham's defensive investor criteria
    MAX_PE = 15.0
    MAX_PB = 1.5
    MAX_PE_X_PB = 22.5  # P/E × P/B < 22.5
    MIN_CURRENT_RATIO = 2.0
    MIN_DIVIDEND_YEARS = 20
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize Graham agent.
        
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
        """Perform Graham-style analysis on a company."""
        metrics = self._calculate_metrics(financials)
        margin_of_safety = self._calculate_margin_of_safety(metrics)
        defensive_criteria = self._check_defensive_criteria(metrics)
        
        # Calculate Graham score
        score = self._calculate_score(metrics, margin_of_safety, defensive_criteria)
        verdict = self._score_to_verdict(score)
        
        # Generate LLM-powered insights if client available
        if self.llm_client:
            insights = await self._generate_llm_insights(ticker, metrics, margin_of_safety, score, verdict)
        else:
            insights = self._generate_insights(metrics, margin_of_safety)
        
        concerns = self._identify_concerns(metrics)
        
        return {
            "agent": self.name,
            "style": self.style,
            "ticker": ticker,
            "score": score,
            "verdict": self._score_to_verdict(score),
            "metrics": metrics.__dict__,
            "margin_of_safety": margin_of_safety,
            "defensive_criteria": defensive_criteria,
            "insights": insights,
            "concerns": concerns,
            "graham_quote": self._get_relevant_quote(margin_of_safety),
        }
    
    def _calculate_metrics(self, financials: Dict[str, Any]) -> GrahamMetrics:
        """Extract and calculate Graham-relevant metrics."""
        eps = financials.get("eps", 0)
        bvps = financials.get("book_value_per_share", 0)
        
        # Graham Number = sqrt(22.5 × EPS × BVPS)
        if eps > 0 and bvps > 0:
            graham_number = math.sqrt(22.5 * eps * bvps)
        else:
            graham_number = 0
        
        # Net-Net Value (NCAV) = Current Assets - Total Liabilities
        current_assets = financials.get("current_assets", 0)
        total_liabilities = financials.get("total_liabilities", 0)
        shares_outstanding = financials.get("shares_outstanding", 1)
        net_net = (current_assets - total_liabilities) / shares_outstanding
        
        return GrahamMetrics(
            pe_ratio=financials.get("pe_ratio", 0),
            pb_ratio=financials.get("pb_ratio", 0),
            current_ratio=financials.get("current_ratio", 0),
            dividend_yield=financials.get("dividend_yield", 0),
            dividend_years=financials.get("consecutive_dividend_years", 0),
            graham_number=graham_number,
            current_price=financials.get("current_price", 0),
            net_net_value=net_net,
        )
    
    def _calculate_margin_of_safety(self, metrics: GrahamMetrics) -> Dict[str, Any]:
        """Calculate margin of safety."""
        if metrics.current_price <= 0:
            return {
                "percentage": 0,
                "assessment": "Unable to calculate",
                "graham_number_margin": 0,
                "net_net_margin": 0,
            }
        
        # Margin vs Graham Number
        graham_margin = 0
        if metrics.graham_number > 0:
            graham_margin = (metrics.graham_number - metrics.current_price) / metrics.graham_number
        
        # Margin vs Net-Net Value
        net_net_margin = 0
        if metrics.net_net_value > 0:
            net_net_margin = (metrics.net_net_value - metrics.current_price) / metrics.net_net_value
        
        # Use the better margin
        best_margin = max(graham_margin, net_net_margin)
        
        if best_margin >= 0.33:
            assessment = "Excellent - meets Graham's 1/3 discount requirement"
        elif best_margin >= 0.20:
            assessment = "Good - reasonable margin of safety"
        elif best_margin >= 0:
            assessment = "Fair - some margin of safety"
        else:
            assessment = "Poor - trading above intrinsic value"
        
        return {
            "percentage": best_margin,
            "assessment": assessment,
            "graham_number_margin": graham_margin,
            "net_net_margin": net_net_margin,
        }
    
    def _check_defensive_criteria(self, metrics: GrahamMetrics) -> Dict[str, bool]:
        """Check Graham's defensive investor criteria."""
        pe_x_pb = metrics.pe_ratio * metrics.pb_ratio
        
        return {
            "current_ratio_pass": metrics.current_ratio >= self.MIN_CURRENT_RATIO,
            "pe_ratio_pass": metrics.pe_ratio <= self.MAX_PE and metrics.pe_ratio > 0,
            "pb_ratio_pass": metrics.pb_ratio <= self.MAX_PB and metrics.pb_ratio > 0,
            "pe_x_pb_pass": pe_x_pb <= self.MAX_PE_X_PB and pe_x_pb > 0,
            "dividend_pass": metrics.dividend_years >= self.MIN_DIVIDEND_YEARS,
            "criteria_met": sum([
                metrics.current_ratio >= self.MIN_CURRENT_RATIO,
                metrics.pe_ratio <= self.MAX_PE and metrics.pe_ratio > 0,
                metrics.pb_ratio <= self.MAX_PB or pe_x_pb <= self.MAX_PE_X_PB,
                metrics.dividend_years >= 10,  # Relaxed from 20
            ]),
        }
    
    def _calculate_score(
        self,
        metrics: GrahamMetrics,
        margin_of_safety: Dict[str, Any],
        defensive_criteria: Dict[str, bool],
    ) -> float:
        """Calculate overall Graham score (-100 to +100)."""
        score = 0.0
        
        # Margin of safety contribution (max 40 points)
        margin_pct = margin_of_safety["percentage"]
        if margin_pct >= 0.33:
            score += 40
        elif margin_pct >= 0.20:
            score += 25
        elif margin_pct >= 0.10:
            score += 10
        elif margin_pct < 0:
            score -= 30
        
        # P/E contribution (max 20 points)
        if metrics.pe_ratio > 0 and metrics.pe_ratio <= 10:
            score += 20
        elif metrics.pe_ratio <= 15:
            score += 15
        elif metrics.pe_ratio <= 20:
            score += 5
        else:
            score -= 15
        
        # Current ratio contribution (max 15 points)
        if metrics.current_ratio >= 2.0:
            score += 15
        elif metrics.current_ratio >= 1.5:
            score += 5
        else:
            score -= 15
        
        # Dividend history contribution (max 15 points)
        if metrics.dividend_years >= 20:
            score += 15
        elif metrics.dividend_years >= 10:
            score += 10
        elif metrics.dividend_years >= 5:
            score += 5
        
        # Net-Net bonus (rare but valuable)
        if metrics.net_net_value > metrics.current_price:
            score += 20
        
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
        metrics: GrahamMetrics,
        margin_of_safety: Dict[str, Any],
        score: float,
        verdict: str,
    ) -> List[str]:
        """Generate LLM-powered insights using Graham's voice."""
        prompt = f"""Analyze {ticker}: P/E {metrics.pe_ratio:.1f}, P/B {metrics.pb_ratio:.1f}, Current Ratio {metrics.current_ratio:.1f}, Margin of Safety {margin_of_safety['percentage']:.1%}"""
        try:
            response = await self.llm_client.analyze(prompt, persona="Benjamin Graham", verdict=verdict)
            return [response] if response else self._generate_insights(metrics, margin_of_safety)
        except Exception:
            return self._generate_insights(metrics, margin_of_safety)
    
    def _generate_insights(
        self,
        metrics: GrahamMetrics,
        margin_of_safety: Dict[str, Any],
    ) -> List[str]:
        """Generate key insights from analysis (fallback)."""
        insights = []
        
        if margin_of_safety["percentage"] >= 0.20:
            pct = margin_of_safety["percentage"] * 100
            insights.append(f"Strong margin of safety at {pct:.0f}% discount to intrinsic value")
        
        if metrics.pe_ratio > 0 and metrics.pe_ratio <= 15:
            insights.append(f"Attractive P/E ratio of {metrics.pe_ratio:.1f}")
        
        if metrics.current_ratio >= 2.0:
            insights.append(f"Strong balance sheet with current ratio of {metrics.current_ratio:.1f}")
        
        if metrics.dividend_years >= 10:
            insights.append(f"Consistent dividend payer for {metrics.dividend_years} years")
        
        if metrics.net_net_value > metrics.current_price:
            insights.append("Net-Net opportunity - trading below liquidation value!")
        
        return insights
    
    def _identify_concerns(self, metrics: GrahamMetrics) -> List[str]:
        """Identify potential concerns from Graham's perspective."""
        concerns = []
        
        if metrics.pe_ratio > 20:
            concerns.append(f"High P/E of {metrics.pe_ratio:.1f} exceeds Graham's maximum of 15")
        
        if metrics.current_ratio < 1.5:
            concerns.append(f"Weak current ratio of {metrics.current_ratio:.1f} indicates liquidity risk")
        
        if metrics.pb_ratio > 2.0:
            concerns.append(f"High P/B ratio of {metrics.pb_ratio:.1f}")
        
        if metrics.dividend_years < 5:
            concerns.append("Limited dividend history")
        
        return concerns
    
    def _get_relevant_quote(self, margin_of_safety: Dict[str, Any]) -> str:
        """Get a relevant Graham quote based on the analysis."""
        if margin_of_safety["percentage"] >= 0.33:
            return "The margin of safety is always dependent on the price paid."
        elif margin_of_safety["percentage"] >= 0:
            return "The intelligent investor is a realist who sells to optimists and buys from pessimists."
        else:
            return "In the short run, the market is a voting machine but in the long run, it is a weighing machine."
