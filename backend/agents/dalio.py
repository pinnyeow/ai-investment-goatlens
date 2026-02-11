"""
Ray Dalio Investment Agent

Key Philosophy:
- "Principles-based" decision making
- Macro/economic cycle awareness
- Risk parity and diversification
- "Radical transparency" in analysis

Key Focus:
- Debt cycles (short-term and long-term)
- Economic machine understanding
- Portfolio construction and risk management
- Correlation and diversification
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import traceback


@dataclass
class DalioMetrics:
    """Key metrics for Dalio-style analysis."""
    beta: float
    volatility: float
    debt_to_equity: float
    interest_coverage: float
    revenue_cyclicality: float
    correlation_to_market: float


class DalioAgent:
    """
    Ray Dalio Investment Analysis Agent.
    
    Evaluates companies based on:
    1. Macro positioning within economic cycles
    2. Risk-adjusted returns
    3. Debt cycle analysis
    4. Diversification contribution
    5. Stress-testing under various scenarios
    """
    
    name = "Ray Dalio"
    style = "Macro & Risk Parity"
    
    # Model routing: Dalio analyzes macro cycles (complex), but gpt-4o-mini works for now
    model_preference = "gpt-4o-mini"
    
    # Dalio's risk thresholds
    MAX_BETA = 1.5
    MIN_INTEREST_COVERAGE = 3.0
    MAX_DEBT_TO_EQUITY = 0.8
    
    # Economic seasons (Dalio's All-Weather framework)
    SEASONS = [
        "rising_growth",      # Stocks, corporate bonds
        "falling_growth",     # Bonds, gold
        "rising_inflation",   # Commodities, TIPS
        "falling_inflation",  # Stocks, bonds
    ]
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize Dalio agent.
        
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
        Perform Dalio-style analysis on a company.
        
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
        cycle_analysis = self._analyze_debt_cycle(financials)
        risk_assessment = self._assess_risk_adjusted_returns(metrics, financials)
        macro_positioning = self._analyze_macro_positioning(financials)
        
        # Calculate Dalio score with earnings volatility adjustment
        earnings_adj = self._earnings_predictability_bonus(earnings_data or [], earnings_streak or {})
        score = self._calculate_score(metrics, risk_assessment, cycle_analysis) + earnings_adj
        score = round(max(-100, min(100, score)), 2)
        verdict = self._score_to_verdict(score)
        
        # Generate LLM-powered insights if client available
        if self.llm_client:
            insights = await self._generate_llm_insights(ticker, metrics, cycle_analysis, risk_assessment, score, verdict, config=config)
        else:
            insights = self._generate_insights(metrics, cycle_analysis, macro_positioning)
        insights.extend(self._earnings_insights(earnings_data or [], earnings_streak or {}))
        
        concerns = self._identify_concerns(metrics, cycle_analysis)
        concerns.extend(self._earnings_concerns(earnings_data or [], earnings_streak or {}))
        
        return {
            "agent": self.name,
            "style": self.style,
            "ticker": ticker,
            "score": score,
            "verdict": self._score_to_verdict(score),
            "metrics": metrics.__dict__,
            "cycle_analysis": cycle_analysis,
            "risk_assessment": risk_assessment,
            "macro_positioning": macro_positioning,
            "insights": insights,
            "concerns": concerns,
            "dalio_principle": self._get_relevant_principle(score),
        }
    
    def _calculate_metrics(self, financials: Dict[str, Any]) -> DalioMetrics:
        """Extract and calculate Dalio-relevant metrics."""
        return DalioMetrics(
            beta=financials.get("beta", 1.0),
            volatility=financials.get("volatility_252d", 0),
            debt_to_equity=financials.get("debt_to_equity", 0),
            interest_coverage=financials.get("interest_coverage", 0),
            revenue_cyclicality=financials.get("revenue_cyclicality", 0),
            correlation_to_market=financials.get("correlation_sp500", 0),
        )
    
    def _analyze_debt_cycle(
        self,
        financials: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze company's position in debt cycle.
        
        Dalio's framework:
        - Short-term debt cycle (~5-8 years)
        - Long-term debt cycle (~75-100 years)
        """
        debt_to_equity = financials.get("debt_to_equity", 0)
        debt_growth = financials.get("debt_growth_3y", 0)
        
        # Assess debt cycle phase
        if debt_growth > 0.15:
            cycle_phase = "expansion"
            risk_level = "elevated"
        elif debt_growth > 0:
            cycle_phase = "moderate_growth"
            risk_level = "normal"
        elif debt_growth > -0.10:
            cycle_phase = "deleveraging"
            risk_level = "caution"
        else:
            cycle_phase = "aggressive_deleveraging"
            risk_level = "distressed"
        
        return {
            "cycle_phase": cycle_phase,
            "debt_level": debt_to_equity,
            "debt_trajectory": "increasing" if debt_growth > 0 else "decreasing",
            "risk_level": risk_level,
            "leverage_sustainability": debt_to_equity < 1.0,
        }
    
    def _assess_risk_adjusted_returns(
        self,
        metrics: DalioMetrics,
        financials: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assess risk-adjusted return potential.
        
        Dalio focuses on return per unit of risk,
        not absolute returns.
        """
        # Sharpe-like analysis
        expected_return = financials.get("expected_return", 0.10)
        risk_free_rate = 0.04  # Assumed
        
        if metrics.volatility > 0:
            risk_adjusted_ratio = (expected_return - risk_free_rate) / metrics.volatility
        else:
            risk_adjusted_ratio = 0
        
        # Diversification benefit (low correlation = good)
        diversification_score = 100 * (1 - abs(metrics.correlation_to_market))
        
        return {
            "risk_adjusted_ratio": risk_adjusted_ratio,
            "volatility": metrics.volatility,
            "beta": metrics.beta,
            "diversification_score": diversification_score,
            "correlation_to_market": metrics.correlation_to_market,
            "assessment": "Favorable" if risk_adjusted_ratio > 0.5 else "Moderate" if risk_adjusted_ratio > 0 else "Unfavorable",
        }
    
    def _analyze_macro_positioning(
        self,
        financials: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze how company performs across economic seasons.
        
        Based on Dalio's All-Weather portfolio concept.
        """
        sector = financials.get("sector", "Unknown")
        
        # Sector sensitivity to economic conditions
        sector_profiles = {
            "Technology": {
                "rising_growth": "positive",
                "falling_growth": "negative",
                "rising_inflation": "mixed",
                "falling_inflation": "positive",
            },
            "Consumer Staples": {
                "rising_growth": "neutral",
                "falling_growth": "positive",
                "rising_inflation": "mixed",
                "falling_inflation": "neutral",
            },
            "Financials": {
                "rising_growth": "positive",
                "falling_growth": "negative",
                "rising_inflation": "positive",
                "falling_inflation": "negative",
            },
            "Utilities": {
                "rising_growth": "neutral",
                "falling_growth": "positive",
                "rising_inflation": "negative",
                "falling_inflation": "positive",
            },
        }
        
        profile = sector_profiles.get(sector, {
            "rising_growth": "unknown",
            "falling_growth": "unknown",
            "rising_inflation": "unknown",
            "falling_inflation": "unknown",
        })
        
        return {
            "sector": sector,
            "season_sensitivity": profile,
            "all_weather_fit": self._calculate_all_weather_fit(profile),
        }
    
    def _calculate_all_weather_fit(self, profile: Dict[str, str]) -> str:
        """Calculate how well company fits All-Weather portfolio."""
        positive_count = sum(1 for v in profile.values() if v == "positive")
        neutral_count = sum(1 for v in profile.values() if v == "neutral")
        
        if positive_count >= 3:
            return "Excellent - performs well in most conditions"
        elif positive_count + neutral_count >= 3:
            return "Good - stable across conditions"
        else:
            return "Limited - sensitive to economic cycles"
    
    def _calculate_score(
        self,
        metrics: DalioMetrics,
        risk_assessment: Dict[str, Any],
        cycle_analysis: Dict[str, Any],
    ) -> float:
        """Calculate overall Dalio score (-100 to +100)."""
        score = 0.0
        
        # Risk-adjusted returns contribution (max 30 points)
        ratio = risk_assessment["risk_adjusted_ratio"]
        if ratio >= 1.0:
            score += 30
        elif ratio >= 0.5:
            score += 20
        elif ratio >= 0.25:
            score += 10
        elif ratio < 0:
            score -= 20
        
        # Diversification contribution (max 20 points)
        div_score = risk_assessment["diversification_score"]
        score += div_score * 0.2
        
        # Debt cycle positioning (max 25 points)
        if cycle_analysis["leverage_sustainability"]:
            score += 15
        else:
            score -= 20
        
        if cycle_analysis["cycle_phase"] == "moderate_growth":
            score += 10
        elif cycle_analysis["cycle_phase"] == "expansion":
            score += 5
        elif cycle_analysis["cycle_phase"] == "deleveraging":
            score -= 5
        elif cycle_analysis["cycle_phase"] == "aggressive_deleveraging":
            score -= 15
        
        # Beta adjustment (max 15 points)
        if 0.8 <= metrics.beta <= 1.2:
            score += 15
        elif 0.5 <= metrics.beta <= 1.5:
            score += 10
        else:
            score -= 10
        
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
    
    def _get_relevant_context(self, metrics: DalioMetrics, cycle_analysis: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Context engineering: only pass Dalio-relevant metrics to reduce tokens."""
        return {
            "beta": metrics.beta,
            "debt_to_equity": metrics.debt_to_equity,
            "interest_coverage": metrics.interest_coverage,
            "cycle_phase": cycle_analysis.get("cycle_phase", "unknown"),
            "risk_level": cycle_analysis.get("risk_level", "unknown"),
            "diversification_score": risk_assessment.get("diversification_score", 0),
        }

    async def _generate_llm_insights(
        self,
        ticker: str,
        metrics: DalioMetrics,
        cycle_analysis: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        score: float,
        verdict: str,
        config: dict = None,
    ) -> List[str]:
        """Generate LLM-powered insights using Dalio's voice."""
        relevant = self._get_relevant_context(metrics, cycle_analysis, risk_assessment)
        prompt = (
            f"Analyze {ticker}: Beta {relevant['beta']:.2f}, "
            f"D/E {relevant['debt_to_equity']:.2f}, "
            f"Interest Coverage {relevant['interest_coverage']:.1f}x, "
            f"Cycle {relevant['cycle_phase']}, Risk {relevant['risk_level']}"
        )
        try:
            response = await self.llm_client.analyze(prompt, persona="Ray Dalio", verdict=verdict, config=config)
            return [response] if response else self._generate_insights(metrics, cycle_analysis, {})
        except Exception as e:
            print(f"[{self.name}] LLM insight generation failed: {e}")
            traceback.print_exc()
            return self._generate_insights(metrics, cycle_analysis, {})

    def _generate_insights(
        self,
        metrics: DalioMetrics,
        cycle_analysis: Dict[str, Any],
        macro_positioning: Dict[str, Any],
    ) -> List[str]:
        """Generate key insights from analysis."""
        insights = []
        
        if cycle_analysis["leverage_sustainability"]:
            insights.append("Sustainable debt levels provide flexibility")
        
        if metrics.correlation_to_market < 0.5:
            insights.append("Low market correlation offers diversification benefit")
        
        if macro_positioning["all_weather_fit"].startswith("Excellent"):
            insights.append("Performs well across different economic conditions")
        
        if 0.8 <= metrics.beta <= 1.2:
            insights.append("Beta near 1.0 indicates market-like risk profile")
        
        return insights
    
    def _identify_concerns(
        self,
        metrics: DalioMetrics,
        cycle_analysis: Dict[str, Any],
    ) -> List[str]:
        """Identify potential concerns from Dalio's perspective."""
        concerns = []
        
        if not cycle_analysis["leverage_sustainability"]:
            concerns.append("High debt levels create vulnerability in downturns")
        
        if metrics.beta > 1.5:
            concerns.append(f"High beta ({metrics.beta:.1f}) amplifies market risk")
        
        if cycle_analysis["risk_level"] in ["elevated", "distressed"]:
            concerns.append(f"Debt cycle risk level: {cycle_analysis['risk_level']}")
        
        if metrics.interest_coverage < 3:
            concerns.append(f"Low interest coverage ({metrics.interest_coverage:.1f}x)")
        
        return concerns
    
    # ------------------------------------------------------------------
    # Earnings-aware methods (Dalio: PREDICTABILITY reduces portfolio risk)
    # ------------------------------------------------------------------

    def _earnings_predictability_bonus(
        self,
        earnings_data: List[Dict],
        earnings_streak: Dict,
    ) -> float:
        """
        Dalio cares about predictability for risk-parity construction.
        Low earnings surprise variance → lower portfolio risk.
        """
        if not earnings_data:
            return 0.0

        bonus = 0.0

        # Low surprise variance = predictable → lower risk (max 10 pts)
        surprises = [abs(e.get("surprise_pct", 0)) for e in earnings_data]
        if len(surprises) >= 3:
            avg_abs_surprise = sum(surprises) / len(surprises)
            if avg_abs_surprise < 3.0:
                bonus += 10  # Very predictable
            elif avg_abs_surprise < 5.0:
                bonus += 5
            elif avg_abs_surprise > 15.0:
                bonus -= 10  # High volatility = risk

        # Consistent beats reduce downside risk (max 5 pts)
        beats = earnings_streak.get("beats", 0)
        total = earnings_streak.get("total", 0)
        if total >= 4 and beats / total >= 0.75:
            bonus += 5

        return bonus

    def _earnings_insights(
        self,
        earnings_data: List[Dict],
        earnings_streak: Dict,
    ) -> List[str]:
        """Generate Dalio-style earnings insights."""
        insights = []
        if len(earnings_data) >= 3:
            surprises = [abs(e.get("surprise_pct", 0)) for e in earnings_data]
            avg = sum(surprises) / len(surprises)
            if avg < 3.0:
                insights.append(f"Low earnings surprise volatility ({avg:.1f}% avg) — predictable cash flows reduce portfolio risk")
        
        beats = earnings_streak.get("beats", 0)
        total = earnings_streak.get("total", 0)
        if total >= 4 and beats / total >= 0.75:
            insights.append(f"Beat {beats}/{total} quarters — consistent outperformance reduces downside risk")
        return insights

    def _earnings_concerns(
        self,
        earnings_data: List[Dict],
        earnings_streak: Dict,
    ) -> List[str]:
        """Generate Dalio-style earnings concerns."""
        concerns = []
        if len(earnings_data) >= 3:
            surprises = [abs(e.get("surprise_pct", 0)) for e in earnings_data]
            avg = sum(surprises) / len(surprises)
            if avg > 15.0:
                concerns.append(f"High earnings surprise volatility ({avg:.1f}% avg) — adds portfolio risk in all-weather context")

        streak_type = earnings_streak.get("streak_type", "")
        streak_count = earnings_streak.get("streak_count", 0)
        if streak_type == "miss" and streak_count >= 3:
            concerns.append(f"Missed {streak_count} consecutive quarters — systematic underperformance increases drawdown risk")
        return concerns

    def _get_relevant_principle(self, score: float) -> str:
        """Get a relevant Dalio principle based on the analysis."""
        if score >= 60:
            return "Pain + Reflection = Progress. Great investments come from understanding cycles."
        elif score >= 20:
            return "Diversifying well is the most important thing you can do to improve risk-adjusted returns."
        elif score >= -20:
            return "Don't mistake possibilities for probabilities."
        else:
            return "The biggest mistake most investors make is to believe that what happened recently is likely to persist."
