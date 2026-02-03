"""
Peter Lynch Investment Agent

Key Philosophy:
- "Invest in what you know"
- Growth at a Reasonable Price (GARP)
- Find "ten-baggers" - stocks that can grow 10x
- Look for companies with simple, understandable businesses

Key Metrics:
- PEG ratio (P/E divided by growth rate) < 1.0
- Earnings growth rate
- Institutional ownership (prefer lower)
- Balance sheet strength
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class LynchMetrics:
    """Key metrics for Lynch-style analysis."""
    peg_ratio: float
    earnings_growth: float
    pe_ratio: float
    institutional_ownership: float
    debt_to_equity: float
    cash_position: float


class LynchAgent:
    """
    Peter Lynch Investment Analysis Agent.
    
    Evaluates companies based on:
    1. PEG ratio (Growth at Reasonable Price)
    2. Earnings growth momentum
    3. Business simplicity and understandability
    4. Hidden assets and overlooked opportunities
    5. Institutional ownership (contrarian indicator)
    """
    
    name = "Peter Lynch"
    style = "Growth at Reasonable Price (GARP)"
    
    # Lynch's preferred thresholds
    MAX_PEG = 1.0
    MIN_EARNINGS_GROWTH = 0.15  # 15%
    IDEAL_INSTITUTIONAL = 0.50  # Prefer < 50%
    
    # Lynch's stock categories
    CATEGORIES = [
        "slow_grower",      # 2-4% growth
        "stalwart",         # 10-12% growth
        "fast_grower",      # 20-25% growth
        "cyclical",
        "turnaround",
        "asset_play",
    ]
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize Lynch agent.
        
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
        Perform Lynch-style analysis on a company.
        """
        metrics = self._calculate_metrics(financials)
        category = self._categorize_stock(metrics)
        ten_bagger_potential = self._assess_ten_bagger_potential(metrics, financials)
        
        # Calculate Lynch score
        score = self._calculate_score(metrics, category, ten_bagger_potential)
        
        # Generate LLM-powered insights if client available
        if self.llm_client:
            insights = await self._generate_llm_insights(ticker, metrics, category, score)
        else:
            insights = self._generate_insights(metrics, category)
        
        concerns = self._identify_concerns(metrics)
        
        return {
            "agent": self.name,
            "style": self.style,
            "ticker": ticker,
            "score": score,
            "verdict": self._score_to_verdict(score),
            "metrics": metrics.__dict__,
            "stock_category": category,
            "ten_bagger_potential": ten_bagger_potential,
            "insights": insights,
            "concerns": concerns,
            "lynch_tip": self._get_relevant_tip(category),
        }
    
    def _calculate_metrics(self, financials: Dict[str, Any]) -> LynchMetrics:
        """Extract and calculate Lynch-relevant metrics."""
        pe = financials.get("pe_ratio", 0)
        growth = financials.get("earnings_growth", 0)
        
        # PEG = P/E / Earnings Growth Rate
        peg = pe / (growth * 100) if growth > 0 else float('inf')
        
        return LynchMetrics(
            peg_ratio=peg,
            earnings_growth=growth,
            pe_ratio=pe,
            institutional_ownership=financials.get("institutional_ownership", 0),
            debt_to_equity=financials.get("debt_to_equity", 0),
            cash_position=financials.get("cash_and_equivalents", 0),
        )
    
    def _categorize_stock(self, metrics: LynchMetrics) -> str:
        """Categorize stock according to Lynch's framework."""
        growth = metrics.earnings_growth
        
        if growth < 0.05:
            return "slow_grower"
        elif growth < 0.15:
            return "stalwart"
        elif growth >= 0.20:
            return "fast_grower"
        else:
            return "stalwart"
    
    def _assess_ten_bagger_potential(
        self,
        metrics: LynchMetrics,
        financials: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess potential for 10x returns."""
        score = 0
        factors = []
        
        if metrics.peg_ratio < 1.0:
            score += 30
            factors.append("Attractive PEG ratio")
        
        if metrics.earnings_growth >= 0.20:
            score += 25
            factors.append("High growth rate")
        
        if metrics.institutional_ownership < 0.50:
            score += 15
            factors.append("Under-discovered by institutions")
        
        if metrics.debt_to_equity < 0.5:
            score += 15
            factors.append("Strong balance sheet")
        
        return {
            "potential_score": min(100, score),
            "favorable_factors": factors,
            "assessment": "High" if score >= 60 else "Medium" if score >= 30 else "Low",
        }
    
    def _calculate_score(
        self,
        metrics: LynchMetrics,
        category: str,
        ten_bagger: Dict[str, Any],
    ) -> float:
        """Calculate overall Lynch score (-100 to +100)."""
        score = 0.0
        
        # PEG ratio contribution (max 35 points)
        if metrics.peg_ratio < 0.5:
            score += 35
        elif metrics.peg_ratio < 1.0:
            score += 25
        elif metrics.peg_ratio < 1.5:
            score += 10
        elif metrics.peg_ratio < 2.0:
            score -= 10
        else:
            score -= 25
        
        # Growth rate contribution (max 25 points)
        if category == "fast_grower":
            score += 25
        elif category == "stalwart":
            score += 15
        elif category == "slow_grower":
            score -= 15
        
        # Institutional ownership (contrarian - max 15 points)
        if metrics.institutional_ownership < 0.30:
            score += 15
        elif metrics.institutional_ownership < 0.50:
            score += 10
        elif metrics.institutional_ownership > 0.80:
            score -= 10
        
        # Ten-bagger potential bonus
        score += ten_bagger["potential_score"] * 0.25
        
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
        metrics: LynchMetrics,
        category: str,
        score: float,
    ) -> List[str]:
        """Generate LLM-powered insights using Lynch's voice."""
        prompt = f"""Analyze {ticker} with these metrics:
- PEG Ratio: {metrics.peg_ratio:.2f}
- Earnings Growth: {metrics.earnings_growth:.1%}
- P/E: {metrics.pe_ratio:.1f}
- Category: {category}
- Score: {score:.0f}/100

Provide 3 key insights in Peter Lynch's voice. Focus on whether this is a ten-bagger opportunity, if it's a GARP play, and whether institutions have discovered it yet."""

        try:
            response = await self.llm_client.analyze(prompt, persona="Peter Lynch")
            insights = [line.strip() for line in response.split("\n") if line.strip() and not line.strip().startswith("#")]
            return insights[:3] if insights else self._generate_insights(metrics, category)
        except Exception:
            return self._generate_insights(metrics, category)
    
    def _generate_insights(
        self,
        metrics: LynchMetrics,
        category: str,
    ) -> List[str]:
        """Generate key insights from analysis (fallback)."""
        insights = []
        
        if metrics.peg_ratio < 1.0:
            insights.append(f"Attractive PEG ratio of {metrics.peg_ratio:.2f} - growth at reasonable price")
        
        if category == "fast_grower":
            insights.append(f"Fast grower with {metrics.earnings_growth:.1%} earnings growth")
        
        if metrics.institutional_ownership < 0.50:
            insights.append("Low institutional ownership suggests discovery potential")
        
        return insights
    
    def _identify_concerns(self, metrics: LynchMetrics) -> List[str]:
        """Identify potential concerns from Lynch's perspective."""
        concerns = []
        
        if metrics.peg_ratio > 2.0:
            concerns.append(f"High PEG ratio of {metrics.peg_ratio:.2f} - paying too much for growth")
        
        if metrics.institutional_ownership > 0.80:
            concerns.append("Heavy institutional ownership - may be fully discovered")
        
        if metrics.earnings_growth < 0:
            concerns.append("Negative earnings growth")
        
        return concerns
    
    def _get_relevant_tip(self, category: str) -> str:
        """Get a relevant Lynch tip based on stock category."""
        tips = {
            "slow_grower": "Slow growers are good for dividends but rarely produce big gains.",
            "stalwart": "Stalwarts won't make you rich, but they can protect your portfolio in downturns.",
            "fast_grower": "Fast growers are where the big wins come from - if you pick the right ones.",
            "cyclical": "With cyclicals, timing is everything. Buy when things look terrible.",
            "turnaround": "Turnarounds can produce the biggest gains, but also the biggest losses.",
            "asset_play": "Asset plays require patience and often a catalyst to unlock value.",
        }
        return tips.get(category, "Know what you own and why you own it.")
