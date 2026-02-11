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
import traceback


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
    
    # Model routing: Buffett needs nuanced reasoning for moat analysis
    # Use gpt-4o for better quality, gpt-4o-mini for cost efficiency
    model_preference = "gpt-4o"  # Can be overridden via env var
    
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
        *,
        earnings_data: Optional[List[Dict]] = None,
        earnings_streak: Optional[Dict] = None,
        config: dict = None,
    ) -> Dict[str, Any]:
        """
        Perform Buffett-style analysis on a company.
        
        Args:
            ticker: Stock ticker symbol
            financials: Historical financial data
            earnings_data: List of quarterly earnings (actual vs estimate)
            earnings_streak: Streak summary dict
            config: LangChain RunnableConfig for trace propagation
            
        Returns:
            Analysis result with verdict, score, and insights
        """
        metrics = self._calculate_metrics(financials)
        moat_analysis = self._assess_moat(metrics)
        management_quality = self._assess_management(financials)
        
        # Calculate Buffett score (now includes earnings consistency bonus)
        earnings_bonus = self._earnings_consistency_bonus(earnings_data or [], earnings_streak or {})
        score = self._calculate_score(metrics, moat_analysis, management_quality) + earnings_bonus
        score = round(max(-100, min(100, score)), 2)
        verdict = self._score_to_verdict(score)
        
        # Generate LLM-powered insights if client available
        if self.llm_client:
            insights = await self._generate_llm_insights(ticker, metrics, score, verdict, config=config)
        else:
            insights = self._generate_insights(metrics, moat_analysis)
        
        # Add earnings-specific insights
        insights.extend(self._earnings_insights(earnings_data or [], earnings_streak or {}))
        
        concerns = self._identify_concerns(metrics)
        concerns.extend(self._earnings_concerns(earnings_data or [], earnings_streak or {}))
        
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
    
    def _get_relevant_context(self, metrics: BuffettMetrics) -> Dict[str, Any]:
        """
        Context engineering: Return only metrics relevant to Buffett's analysis.
        
        This reduces token usage in LLM calls by filtering out irrelevant data.
        Buffett focuses on: ROE, margins, debt, moat strength, owner earnings.
        He doesn't need: beta, institutional ownership, PEG, etc.
        
        Token savings: ~30-40 tokens per LLM call (from ~120 to ~80 tokens).
        With 5 agents × multiple calls, this adds up to significant savings.
        
        Args:
            metrics: Full metrics object
            
        Returns:
            Filtered dict with only relevant metrics
        """
        return {
            "roe": metrics.roe,
            "profit_margin": metrics.profit_margin,
            "debt_to_equity": metrics.debt_to_equity,
            "moat_strength": metrics.moat_strength,
            "owner_earnings": metrics.owner_earnings,
        }
    
    async def _generate_llm_insights(
        self,
        ticker: str,
        metrics: BuffettMetrics,
        score: float,
        verdict: str,
        config: dict = None,
    ) -> List[str]:
        """Generate LLM-powered insights using Buffett's voice."""
        # Context optimization: only pass relevant metrics to reduce token usage
        relevant = self._get_relevant_context(metrics)
        prompt = f"""Analyze {ticker}: ROE {relevant['roe']:.1%}, Profit Margin {relevant['profit_margin']:.1%}, Debt/Equity {relevant['debt_to_equity']:.2f}, Moat {relevant['moat_strength']}"""
        try:
            response = await self.llm_client.analyze(prompt, persona="Warren Buffett", verdict=verdict, config=config)
            return [response] if response else self._generate_insights(metrics, {"strength": metrics.moat_strength})
        except Exception as e:
            print(f"[{self.name}] LLM insight generation failed: {e}")
            traceback.print_exc()
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
    
    # ------------------------------------------------------------------
    # Earnings-aware methods (Buffett: values CONSISTENCY of earnings)
    # ------------------------------------------------------------------

    def _earnings_consistency_bonus(
        self,
        earnings_data: List[Dict],
        earnings_streak: Dict,
    ) -> float:
        """
        Buffett rewards predictable earnings power.
        Consistent beats = management under-promises and over-delivers.
        """
        if not earnings_data:
            return 0.0

        bonus = 0.0
        total = earnings_streak.get("total", 0)
        beats = earnings_streak.get("beats", 0)
        streak_count = earnings_streak.get("streak_count", 0)
        streak_type = earnings_streak.get("streak_type", "")

        # Beat rate — Buffett loves consistency (max 10 pts)
        if total > 0:
            beat_rate = beats / total
            if beat_rate >= 0.80:
                bonus += 10
            elif beat_rate >= 0.60:
                bonus += 5

        # Consecutive beat streak bonus (max 5 pts)
        if streak_type == "beat" and streak_count >= 4:
            bonus += 5
        elif streak_type == "miss" and streak_count >= 2:
            bonus -= 10  # Buffett worries about deteriorating earnings

        return bonus

    def _earnings_insights(
        self,
        earnings_data: List[Dict],
        earnings_streak: Dict,
    ) -> List[str]:
        """Generate Buffett-style earnings insights."""
        insights = []
        beats = earnings_streak.get("beats", 0)
        total = earnings_streak.get("total", 0)
        streak_type = earnings_streak.get("streak_type", "")
        streak_count = earnings_streak.get("streak_count", 0)

        if total >= 4 and beats / total >= 0.80:
            insights.append(f"Consistent earnings beats ({beats}/{total} quarters) — predictable earnings power")
        if streak_type == "beat" and streak_count >= 4:
            insights.append(f"Beat consensus {streak_count} consecutive quarters — management credibility")
        return insights

    def _earnings_concerns(
        self,
        earnings_data: List[Dict],
        earnings_streak: Dict,
    ) -> List[str]:
        """Generate Buffett-style earnings concerns."""
        concerns = []
        streak_type = earnings_streak.get("streak_type", "")
        streak_count = earnings_streak.get("streak_count", 0)
        misses = earnings_streak.get("misses", 0)
        total = earnings_streak.get("total", 0)

        if streak_type == "miss" and streak_count >= 2:
            concerns.append(f"Missed earnings {streak_count} quarters in a row — deteriorating earnings power")
        elif total > 0 and misses / total >= 0.50:
            concerns.append(f"Missed earnings {misses}/{total} quarters — unreliable earnings")
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
