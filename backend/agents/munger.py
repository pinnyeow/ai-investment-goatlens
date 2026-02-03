"""
Charlie Munger Investment Agent

Key Philosophy:
- "Invert, always invert" - think about what to avoid
- Mental models from multiple disciplines
- Focus on quality businesses, not statistical cheapness
- "Sit on your ass investing" - patience and conviction

Key Focus:
- Quality of business (not just cheap price)
- Management integrity and competence
- Pricing power and competitive position
- Avoiding mistakes (what NOT to buy)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class MungerMetrics:
    """Key metrics for Munger-style analysis."""
    gross_margin: float
    operating_margin: float
    roic: float  # Return on Invested Capital
    revenue_growth_5y: float
    management_ownership: float
    capex_to_revenue: float


class MungerAgent:
    """
    Charlie Munger Investment Analysis Agent.
    
    Evaluates companies based on:
    1. Business quality (not just valuation)
    2. Mental models and multidisciplinary thinking
    3. Management quality and incentives
    4. Avoiding mistakes (inversion)
    5. Long-term structural advantages
    """
    
    name = "Charlie Munger"
    style = "Quality-Focused Mental Models"
    
    # Munger's quality thresholds
    MIN_GROSS_MARGIN = 0.40  # 40%
    MIN_ROIC = 0.15  # 15%
    MIN_OPERATING_MARGIN = 0.15  # 15%
    
    # Red flags (inversion - what to avoid)
    RED_FLAGS = [
        "high_debt",
        "declining_margins",
        "promotional_management",
        "complex_accounting",
        "high_capex_requirements",
        "commodity_business",
        "regulatory_risk",
    ]
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize Munger agent.
        
        Args:
            llm_client: Optional LLM client for narrative analysis
        """
        self.llm_client = llm_client
    
    async def analyze(
        self,
        ticker: str,
        financials: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform Munger-style analysis on a company.
        
        Args:
            ticker: Stock ticker symbol
            financials: Historical financial data
            
        Returns:
            Analysis result with verdict, score, and insights
        """
        metrics = self._calculate_metrics(financials)
        quality_assessment = self._assess_business_quality(metrics, financials)
        red_flags = self._identify_red_flags(metrics, financials)
        mental_models = self._apply_mental_models(financials)
        
        # Calculate Munger score
        score = self._calculate_score(metrics, quality_assessment, red_flags)
        
        insights = self._generate_insights(metrics, quality_assessment)
        concerns = red_flags  # Munger's concerns ARE the red flags
        
        return {
            "agent": self.name,
            "style": self.style,
            "ticker": ticker,
            "score": score,
            "verdict": self._score_to_verdict(score),
            "metrics": metrics.__dict__,
            "quality_assessment": quality_assessment,
            "red_flags": red_flags,
            "mental_models_applied": mental_models,
            "insights": insights,
            "concerns": concerns,
            "munger_wisdom": self._get_relevant_wisdom(score),
        }
    
    def _calculate_metrics(self, financials: Dict[str, Any]) -> MungerMetrics:
        """Extract and calculate Munger-relevant metrics."""
        return MungerMetrics(
            gross_margin=financials.get("gross_margin", 0),
            operating_margin=financials.get("operating_margin", 0),
            roic=financials.get("roic", 0),
            revenue_growth_5y=financials.get("revenue_growth_5y", 0),
            management_ownership=financials.get("insider_ownership", 0),
            capex_to_revenue=financials.get("capex_to_revenue", 0),
        )
    
    def _assess_business_quality(
        self,
        metrics: MungerMetrics,
        financials: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assess overall business quality.
        
        Munger looks for:
        - Structural competitive advantages
        - Pricing power
        - Low capital requirements
        - Simple, understandable business
        """
        quality_score = 0
        factors = []
        
        # High margins indicate pricing power
        if metrics.gross_margin >= self.MIN_GROSS_MARGIN:
            quality_score += 25
            factors.append("Strong pricing power (high gross margins)")
        
        if metrics.operating_margin >= self.MIN_OPERATING_MARGIN:
            quality_score += 20
            factors.append("Efficient operations (high operating margins)")
        
        # High ROIC indicates capital efficiency
        if metrics.roic >= self.MIN_ROIC:
            quality_score += 25
            factors.append("Excellent capital allocation (high ROIC)")
        
        # Low capex = capital-light business
        if metrics.capex_to_revenue < 0.05:
            quality_score += 15
            factors.append("Capital-light business model")
        
        # Consistent growth
        if metrics.revenue_growth_5y > 0.05:
            quality_score += 15
            factors.append("Sustainable revenue growth")
        
        # Classify quality level
        if quality_score >= 80:
            quality_level = "Exceptional"
        elif quality_score >= 60:
            quality_level = "High"
        elif quality_score >= 40:
            quality_level = "Moderate"
        else:
            quality_level = "Low"
        
        return {
            "score": quality_score,
            "level": quality_level,
            "favorable_factors": factors,
        }
    
    def _identify_red_flags(
        self,
        metrics: MungerMetrics,
        financials: Dict[str, Any],
    ) -> List[str]:
        """
        Apply inversion - identify reasons NOT to invest.
        
        Munger: "Tell me where I'm going to die, so I won't go there."
        """
        red_flags = []
        
        # High debt
        debt_to_equity = financials.get("debt_to_equity", 0)
        if debt_to_equity > 1.0:
            red_flags.append(f"High debt burden (D/E: {debt_to_equity:.1f})")
        
        # Declining margins (moat erosion)
        margin_trend = financials.get("margin_trend", 0)
        if margin_trend < -0.02:
            red_flags.append("Declining profit margins - possible moat erosion")
        
        # Low margins (commodity-like)
        if metrics.gross_margin < 0.20:
            red_flags.append("Low gross margins - commodity business characteristics")
        
        # High capex requirements
        if metrics.capex_to_revenue > 0.15:
            red_flags.append("High capital requirements to maintain business")
        
        # Poor capital allocation
        if metrics.roic < 0.08:
            red_flags.append("Poor return on invested capital")
        
        # Low insider ownership
        if metrics.management_ownership < 0.01:
            red_flags.append("Low management ownership - misaligned incentives")
        
        return red_flags
    
    def _apply_mental_models(self, financials: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Apply Munger's multidisciplinary mental models.
        
        Munger uses ~100 mental models from various fields.
        """
        models_applied = []
        
        # Moat Analysis (from Buffett/Competitive Strategy)
        models_applied.append({
            "model": "Economic Moat",
            "discipline": "Business Strategy",
            "application": "Assessing durability of competitive advantages",
        })
        
        # Incentive Analysis (Psychology)
        models_applied.append({
            "model": "Incentive-Caused Bias",
            "discipline": "Psychology",
            "application": "Evaluating management incentives and alignment",
        })
        
        # Margin of Safety (Engineering)
        models_applied.append({
            "model": "Margin of Safety",
            "discipline": "Engineering",
            "application": "Building in buffer for uncertainty",
        })
        
        # Circle of Competence
        models_applied.append({
            "model": "Circle of Competence",
            "discipline": "Epistemology",
            "application": "Staying within areas of understanding",
        })
        
        return models_applied
    
    def _calculate_score(
        self,
        metrics: MungerMetrics,
        quality_assessment: Dict[str, Any],
        red_flags: List[str],
    ) -> float:
        """Calculate overall Munger score (-100 to +100)."""
        # Start with quality score (max 100)
        score = quality_assessment["score"]
        
        # Subtract for each red flag (big penalties - Munger avoids mistakes)
        score -= len(red_flags) * 20
        
        # Management ownership bonus
        if metrics.management_ownership >= 0.10:
            score += 15
        elif metrics.management_ownership >= 0.05:
            score += 10
        
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
    
    def _generate_insights(
        self,
        metrics: MungerMetrics,
        quality_assessment: Dict[str, Any],
    ) -> List[str]:
        """Generate key insights from analysis."""
        insights = []
        
        if quality_assessment["level"] in ["Exceptional", "High"]:
            insights.append(f"{quality_assessment['level']} quality business")
        
        if metrics.roic >= 0.20:
            insights.append(f"Outstanding capital allocation with {metrics.roic:.1%} ROIC")
        
        if metrics.gross_margin >= 0.50:
            insights.append(f"Strong pricing power evident in {metrics.gross_margin:.1%} gross margin")
        
        if metrics.management_ownership >= 0.10:
            insights.append("Significant management ownership aligns incentives")
        
        return insights
    
    def _get_relevant_wisdom(self, score: float) -> str:
        """Get a relevant Munger quote based on the analysis."""
        if score >= 60:
            return "A great business at a fair price is superior to a fair business at a great price."
        elif score >= 20:
            return "The big money is not in the buying and selling, but in the waiting."
        elif score >= -20:
            return "The first rule is that you can't really know anything if you just remember isolated facts."
        else:
            return "All I want to know is where I'm going to die, so I'll never go there."
