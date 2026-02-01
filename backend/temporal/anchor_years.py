"""
Temporal Analysis - Anchor Year Comparison

Implements the "Time-Travel" analysis feature:
- Compare company fundamentals across key years
- Detect moat decay or strengthening
- Track how the company story has evolved
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class MoatTrend(Enum):
    """Direction of moat strength change."""
    STRENGTHENING = "strengthening"
    STABLE = "stable"
    WEAKENING = "weakening"
    COLLAPSED = "collapsed"


@dataclass
class AnchorYearSnapshot:
    """
    Financial snapshot for a specific anchor year.
    
    Contains key metrics and narrative for that point in time.
    """
    year: int
    
    # Key Financial Metrics
    revenue: float
    net_income: float
    eps: float
    roe: float
    profit_margin: float
    debt_to_equity: float
    free_cash_flow: float
    
    # Competitive Position
    market_share: Optional[float]
    gross_margin: float
    operating_margin: float
    
    # Narrative Elements
    narrative_summary: Optional[str]
    key_events: List[str]


@dataclass
class MoatDecayResult:
    """
    Result of moat decay analysis.
    
    Compares competitive advantages across anchor years
    to identify strengthening or erosion.
    """
    overall_trend: MoatTrend
    moat_score_history: Dict[int, float]  # year -> score
    margin_trend: Dict[str, float]  # metric -> change
    key_observations: List[str]
    risk_factors: List[str]


@dataclass
class StoryEvolution:
    """
    How the company narrative has evolved over time.
    
    Tracks major strategic shifts, pivots, and
    changes in competitive positioning.
    """
    original_story: str
    current_story: str
    pivot_points: List[Dict[str, Any]]  # {year, description, impact}
    consistency_score: float  # 0-100
    narrative_risks: List[str]


class TemporalAnalyzer:
    """
    Performs time-travel analysis across anchor years.
    
    Core functionality:
    1. Create snapshots of company at different points
    2. Detect moat decay/strengthening patterns
    3. Track story evolution and strategic pivots
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize temporal analyzer.
        
        Args:
            llm_client: Optional LLM for narrative analysis
        """
        self.llm_client = llm_client
    
    def create_snapshot(
        self,
        year: int,
        year_data: Dict[str, Any],
    ) -> AnchorYearSnapshot:
        """
        Create a financial snapshot for a specific year.
        
        Args:
            year: The anchor year
            year_data: Financial data for that year (from FMP)
            
        Returns:
            AnchorYearSnapshot with key metrics
        """
        income = year_data.get("income", {}) or {}
        balance = year_data.get("balance", {}) or {}
        cash_flow = year_data.get("cash_flow", {}) or {}
        ratios = year_data.get("ratios", {}) or {}
        
        revenue = income.get("revenue", 0)
        net_income = income.get("netIncome", 0)
        
        return AnchorYearSnapshot(
            year=year,
            revenue=revenue,
            net_income=net_income,
            eps=income.get("eps", 0),
            roe=ratios.get("returnOnEquity", 0),
            profit_margin=net_income / revenue if revenue else 0,
            debt_to_equity=ratios.get("debtEquityRatio", 0),
            free_cash_flow=cash_flow.get("freeCashFlow", 0),
            market_share=None,  # Would need industry data
            gross_margin=ratios.get("grossProfitMargin", 0),
            operating_margin=ratios.get("operatingProfitMargin", 0),
            narrative_summary=None,  # Populated by LLM
            key_events=[],  # Populated by LLM or manual input
        )
    
    def analyze_moat_decay(
        self,
        snapshots: List[AnchorYearSnapshot],
    ) -> MoatDecayResult:
        """
        Analyze whether competitive advantages are strengthening or eroding.
        
        Key indicators:
        - Gross margin trends (pricing power)
        - Operating margin trends (efficiency)
        - ROE trends (capital allocation)
        - Revenue growth consistency
        
        Args:
            snapshots: List of anchor year snapshots (chronological)
            
        Returns:
            MoatDecayResult with trend analysis
        """
        if len(snapshots) < 2:
            return MoatDecayResult(
                overall_trend=MoatTrend.STABLE,
                moat_score_history={},
                margin_trend={},
                key_observations=["Insufficient data for trend analysis"],
                risk_factors=[],
            )
        
        # Sort by year
        snapshots = sorted(snapshots, key=lambda s: s.year)
        
        # Calculate moat score for each year
        moat_scores = {}
        for snapshot in snapshots:
            score = self._calculate_moat_score(snapshot)
            moat_scores[snapshot.year] = score
        
        # Calculate margin trends
        first, last = snapshots[0], snapshots[-1]
        margin_trend = {
            "gross_margin": last.gross_margin - first.gross_margin,
            "operating_margin": last.operating_margin - first.operating_margin,
            "roe": last.roe - first.roe,
            "profit_margin": last.profit_margin - first.profit_margin,
        }
        
        # Determine overall trend
        overall_trend = self._determine_moat_trend(moat_scores, margin_trend)
        
        # Generate observations
        observations = self._generate_moat_observations(snapshots, margin_trend)
        
        # Identify risk factors
        risk_factors = self._identify_moat_risks(snapshots, margin_trend)
        
        return MoatDecayResult(
            overall_trend=overall_trend,
            moat_score_history=moat_scores,
            margin_trend=margin_trend,
            key_observations=observations,
            risk_factors=risk_factors,
        )
    
    def analyze_story_evolution(
        self,
        snapshots: List[AnchorYearSnapshot],
        company_name: str,
    ) -> StoryEvolution:
        """
        Analyze how the company narrative has evolved.
        
        Identifies:
        - Strategic pivots
        - Business model changes
        - Narrative consistency
        
        Args:
            snapshots: Anchor year snapshots
            company_name: Company name for context
            
        Returns:
            StoryEvolution analysis
        """
        if len(snapshots) < 2:
            return StoryEvolution(
                original_story=f"{company_name} - insufficient historical data",
                current_story=f"{company_name} - current state unknown",
                pivot_points=[],
                consistency_score=50.0,
                narrative_risks=["Limited historical data"],
            )
        
        snapshots = sorted(snapshots, key=lambda s: s.year)
        first, last = snapshots[0], snapshots[-1]
        
        # Identify pivot points (significant changes)
        pivot_points = self._identify_pivot_points(snapshots)
        
        # Calculate narrative consistency
        consistency_score = self._calculate_consistency(snapshots)
        
        # Identify narrative risks
        narrative_risks = []
        if len(pivot_points) > 3:
            narrative_risks.append("Frequent strategic changes may indicate lack of focus")
        
        if consistency_score < 50:
            narrative_risks.append("Inconsistent narrative raises execution concerns")
        
        # Generate story summaries
        original_story = self._generate_period_story(first, company_name)
        current_story = self._generate_period_story(last, company_name)
        
        return StoryEvolution(
            original_story=original_story,
            current_story=current_story,
            pivot_points=pivot_points,
            consistency_score=consistency_score,
            narrative_risks=narrative_risks,
        )
    
    def compare_anchor_years(
        self,
        anchor_years: List[int],
        year_data: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare key metrics across anchor years.
        
        Creates a comprehensive comparison table.
        
        Args:
            anchor_years: Years to compare
            year_data: Financial data by year
            
        Returns:
            Comparison results with metrics and insights
        """
        snapshots = []
        for year in anchor_years:
            if year in year_data:
                snapshot = self.create_snapshot(year, year_data[year])
                snapshots.append(snapshot)
        
        moat_analysis = self.analyze_moat_decay(snapshots)
        
        # Build comparison table
        comparison = {
            "years": anchor_years,
            "metrics": {},
            "moat_analysis": moat_analysis,
        }
        
        metrics_to_compare = [
            "revenue", "net_income", "eps", "roe",
            "profit_margin", "gross_margin", "operating_margin",
            "debt_to_equity", "free_cash_flow",
        ]
        
        for metric in metrics_to_compare:
            comparison["metrics"][metric] = {
                s.year: getattr(s, metric) for s in snapshots
            }
        
        # Calculate CAGR for key metrics
        if len(snapshots) >= 2:
            first, last = snapshots[0], snapshots[-1]
            years_diff = last.year - first.year
            
            if years_diff > 0 and first.revenue > 0 and last.revenue > 0:
                revenue_cagr = (last.revenue / first.revenue) ** (1 / years_diff) - 1
                comparison["revenue_cagr"] = revenue_cagr
            
            if years_diff > 0 and first.eps > 0 and last.eps > 0:
                eps_cagr = (last.eps / first.eps) ** (1 / years_diff) - 1
                comparison["eps_cagr"] = eps_cagr
        
        return comparison
    
    def _calculate_moat_score(self, snapshot: AnchorYearSnapshot) -> float:
        """
        Calculate a moat strength score (0-100).
        
        Higher margins and ROE indicate stronger moat.
        """
        score = 0.0
        
        # Gross margin contribution (max 30)
        if snapshot.gross_margin >= 0.50:
            score += 30
        elif snapshot.gross_margin >= 0.40:
            score += 25
        elif snapshot.gross_margin >= 0.30:
            score += 20
        elif snapshot.gross_margin >= 0.20:
            score += 10
        
        # Operating margin contribution (max 30)
        if snapshot.operating_margin >= 0.25:
            score += 30
        elif snapshot.operating_margin >= 0.15:
            score += 20
        elif snapshot.operating_margin >= 0.10:
            score += 10
        
        # ROE contribution (max 25)
        if snapshot.roe >= 0.25:
            score += 25
        elif snapshot.roe >= 0.15:
            score += 20
        elif snapshot.roe >= 0.10:
            score += 10
        
        # Low debt contribution (max 15)
        if snapshot.debt_to_equity < 0.3:
            score += 15
        elif snapshot.debt_to_equity < 0.5:
            score += 10
        elif snapshot.debt_to_equity < 1.0:
            score += 5
        
        return score
    
    def _determine_moat_trend(
        self,
        moat_scores: Dict[int, float],
        margin_trend: Dict[str, float],
    ) -> MoatTrend:
        """Determine overall moat trend from score history."""
        if len(moat_scores) < 2:
            return MoatTrend.STABLE
        
        years = sorted(moat_scores.keys())
        first_score = moat_scores[years[0]]
        last_score = moat_scores[years[-1]]
        
        score_change = last_score - first_score
        
        # Check margin trends
        avg_margin_change = sum(margin_trend.values()) / len(margin_trend)
        
        if score_change > 15 and avg_margin_change > 0.02:
            return MoatTrend.STRENGTHENING
        elif score_change < -15 and avg_margin_change < -0.02:
            return MoatTrend.WEAKENING
        elif score_change < -30:
            return MoatTrend.COLLAPSED
        else:
            return MoatTrend.STABLE
    
    def _generate_moat_observations(
        self,
        snapshots: List[AnchorYearSnapshot],
        margin_trend: Dict[str, float],
    ) -> List[str]:
        """Generate key observations about moat evolution."""
        observations = []
        
        if margin_trend["gross_margin"] > 0.05:
            observations.append("Gross margins expanded, indicating strengthening pricing power")
        elif margin_trend["gross_margin"] < -0.05:
            observations.append("Gross margins contracted, suggesting competitive pressure")
        
        if margin_trend["roe"] > 0.05:
            observations.append("ROE improvement shows better capital deployment")
        elif margin_trend["roe"] < -0.05:
            observations.append("Declining ROE may indicate diminishing competitive advantages")
        
        first, last = snapshots[0], snapshots[-1]
        revenue_growth = (last.revenue / first.revenue - 1) if first.revenue else 0
        
        if revenue_growth > 1.0:  # 2x+ growth
            observations.append(f"Revenue more than doubled from {first.year} to {last.year}")
        elif revenue_growth > 0.5:
            observations.append(f"Solid revenue growth of {revenue_growth:.0%}")
        elif revenue_growth < 0:
            observations.append("Revenue has declined - concerning trend")
        
        return observations
    
    def _identify_moat_risks(
        self,
        snapshots: List[AnchorYearSnapshot],
        margin_trend: Dict[str, float],
    ) -> List[str]:
        """Identify risk factors for moat sustainability."""
        risks = []
        
        if margin_trend["gross_margin"] < -0.03:
            risks.append("Eroding gross margins signal pricing power loss")
        
        if margin_trend["operating_margin"] < -0.03:
            risks.append("Declining operating margins suggest cost structure issues")
        
        last = snapshots[-1]
        if last.debt_to_equity > 1.5:
            risks.append("High leverage reduces financial flexibility")
        
        if last.roe < 0.10:
            risks.append("Low ROE questions capital allocation effectiveness")
        
        return risks
    
    def _identify_pivot_points(
        self,
        snapshots: List[AnchorYearSnapshot],
    ) -> List[Dict[str, Any]]:
        """Identify years with significant strategic or financial changes."""
        pivot_points = []
        
        for i in range(1, len(snapshots)):
            prev, curr = snapshots[i-1], snapshots[i]
            
            # Check for significant revenue change
            if prev.revenue > 0:
                revenue_change = (curr.revenue - prev.revenue) / prev.revenue
                if abs(revenue_change) > 0.30:
                    pivot_points.append({
                        "year": curr.year,
                        "type": "revenue_shift",
                        "description": f"Revenue {'jumped' if revenue_change > 0 else 'dropped'} {abs(revenue_change):.0%}",
                        "magnitude": revenue_change,
                    })
            
            # Check for margin shifts
            margin_change = curr.profit_margin - prev.profit_margin
            if abs(margin_change) > 0.05:
                pivot_points.append({
                    "year": curr.year,
                    "type": "margin_shift",
                    "description": f"Profit margin {'improved' if margin_change > 0 else 'declined'} by {abs(margin_change):.1%}",
                    "magnitude": margin_change,
                })
        
        return pivot_points
    
    def _calculate_consistency(self, snapshots: List[AnchorYearSnapshot]) -> float:
        """
        Calculate narrative/financial consistency score.
        
        Higher score = more consistent performance.
        """
        if len(snapshots) < 2:
            return 50.0
        
        # Check margin consistency
        margins = [s.profit_margin for s in snapshots]
        margin_variance = self._calculate_variance(margins)
        
        # Check growth consistency
        revenues = [s.revenue for s in snapshots]
        growth_consistency = all(
            revenues[i+1] >= revenues[i] * 0.9 
            for i in range(len(revenues)-1)
        )
        
        score = 100.0
        
        # Penalize high variance
        if margin_variance > 0.01:
            score -= min(30, margin_variance * 1000)
        
        # Penalize inconsistent growth
        if not growth_consistency:
            score -= 20
        
        return max(0, score)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
    
    def _generate_period_story(
        self,
        snapshot: AnchorYearSnapshot,
        company_name: str,
    ) -> str:
        """Generate a brief narrative for a time period."""
        if snapshot.profit_margin >= 0.20:
            profitability = "highly profitable"
        elif snapshot.profit_margin >= 0.10:
            profitability = "solidly profitable"
        elif snapshot.profit_margin >= 0:
            profitability = "marginally profitable"
        else:
            profitability = "unprofitable"
        
        if snapshot.roe >= 0.20:
            capital_efficiency = "exceptional capital returns"
        elif snapshot.roe >= 0.15:
            capital_efficiency = "strong capital returns"
        else:
            capital_efficiency = "modest capital returns"
        
        return (
            f"In {snapshot.year}, {company_name} was {profitability} with "
            f"{capital_efficiency}. Revenue stood at ${snapshot.revenue/1e9:.1f}B "
            f"with ROE of {snapshot.roe:.1%}."
        )
