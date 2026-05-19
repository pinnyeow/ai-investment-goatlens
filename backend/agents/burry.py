"""
Michael Burry Investment Agent

Key Philosophy:
- Find what everyone is missing — the hidden risk buried in the footnotes
- Only push back when quantitative red flags are present, not on gut feel
- "The market can stay irrational longer than you can stay solvent" — but the numbers don't lie
- Contrarian by evidence, not by temperament

Key Metrics:
- Debt/equity (leverage amplifies downside in stress scenarios)
- Gross margin trends (structural vs. cyclical deterioration)
- P/E vs. sector average (euphoric valuation vs. reality)
- Free cash flow (accounting earnings can be manipulated; FCF cannot)
- Revenue growth deceleration (growth story breaking down while bulls stay bullish)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import traceback

@dataclass
class BurryMetrics:
    """Key metrics for Burry-style forensic analysis."""
    debt_to_equity: float
    gross_margin: float
    pe_ratio: float
    fcf_per_share: float
    revenue_growth: float      # current YoY
    revenue_growth_5y: float   # 5-year CAGR
    sector: str
    sector_pe: float           # estimated from sector name


# Sector PE lookup — rough historical averages used for relative valuation
_SECTOR_PE = {
    "Technology": 35.0,
    "Healthcare": 28.0,
    "Consumer Cyclical": 25.0,
    "Consumer Defensive": 22.0,
    "Consumer Staples": 22.0,
    "Financials": 15.0,
    "Financial Services": 15.0,
    "Energy": 18.0,
    "Utilities": 20.0,
    "Real Estate": 30.0,
    "Materials": 22.0,
    "Industrials": 22.0,
    "Communication Services": 28.0,
    "Basic Materials": 20.0,
}


class BurryAgent:
    """
    Michael Burry Contrarian Investment Agent.

    Receives the outputs of all 5 existing agents plus raw financial data.
    Only fires contrarian signals when specific quantitative thresholds are met.
    Defaults to agreeing with consensus when no red flags are found.
    """

    name = "Michael Burry"
    style = "Contrarian Deep Value"
    model_preference = "gpt-4o"

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    async def analyze(
        self,
        ticker: str,
        financials: Dict[str, Any],
        *,
        agent_results: List[Dict[str, Any]],
        earnings_data: Optional[List[Dict]] = None,
        earnings_streak: Optional[Dict] = None,
        recent_news: Optional[List[Dict]] = None,
        config: dict = None,
    ) -> Dict[str, Any]:
        """
        Perform Burry-style forensic analysis.

        Args:
            ticker: Stock ticker symbol
            financials: Normalized financial data dict
            agent_results: Outputs from all 5 existing agents (verdicts, scores, insights, concerns)
            earnings_data: List of quarterly earnings (actual vs estimate)
            earnings_streak: Streak summary dict
            recent_news: Recent news headlines for LLM context
            config: LangChain RunnableConfig for trace propagation

        Returns:
            Analysis result with verdict, score, insights, concerns, and triggers_fired
        """
        metrics = self._calculate_metrics(financials)
        consensus_score = self._consensus_score(agent_results)
        triggers = self._check_contrarian_triggers(metrics, consensus_score)
        score = self._calculate_score(triggers, consensus_score)
        score = round(max(-100, min(100, score)), 2)
        verdict = self._score_to_verdict(score)

        if self.llm_client:
            insights = await self._generate_llm_insights(
                ticker, metrics, triggers, agent_results, score, verdict,
                recent_news=recent_news, config=config,
            )
            concerns = await self._generate_llm_concerns(
                ticker, metrics, triggers, agent_results, score, verdict,
                recent_news=recent_news, config=config,
            )
        else:
            insights = self._generate_insights(metrics, triggers, agent_results)
            concerns = self._identify_concerns(metrics, triggers)

        burry_note = (
            "Contrarian signals detected — see triggers_fired for details."
            if triggers
            else "No major red flags found. Consensus appears sound on the numbers."
        )

        return {
            "agent": self.name,
            "style": self.style,
            "ticker": ticker,
            "score": score,
            "verdict": verdict,
            "metrics": {
                "debt_to_equity": metrics.debt_to_equity,
                "gross_margin": metrics.gross_margin,
                "pe_ratio": metrics.pe_ratio,
                "fcf_per_share": metrics.fcf_per_share,
                "revenue_growth": metrics.revenue_growth,
                "revenue_growth_5y": metrics.revenue_growth_5y,
                "sector_pe": metrics.sector_pe,
            },
            "insights": insights,
            "concerns": concerns,
            "triggers_fired": triggers,
            "burry_note": burry_note,
        }

    # ------------------------------------------------------------------
    # Metrics & scoring
    # ------------------------------------------------------------------

    def _calculate_metrics(self, financials: Dict[str, Any]) -> BurryMetrics:
        sector = financials.get("sector", "")
        return BurryMetrics(
            debt_to_equity=financials.get("debt_to_equity", 0) or 0,
            gross_margin=financials.get("gross_margin", 0) or 0,
            pe_ratio=financials.get("pe_ratio", 0) or 0,
            fcf_per_share=financials.get("free_cash_flow_per_share", 0) or 0,
            revenue_growth=financials.get("revenue_growth", 0) or 0,
            revenue_growth_5y=financials.get("revenue_growth_5y", 0) or 0,
            sector=sector,
            sector_pe=self._estimate_sector_pe(sector),
        )

    def _estimate_sector_pe(self, sector: str) -> float:
        return _SECTOR_PE.get(sector, 25.0)

    def _consensus_score(self, agent_results: List[Dict[str, Any]]) -> float:
        scores = [r.get("score", 0) for r in agent_results if "score" in r]
        return sum(scores) / len(scores) if scores else 0.0

    def _check_contrarian_triggers(
        self,
        metrics: BurryMetrics,
        consensus_score: float,
    ) -> List[str]:
        """Return list of fired trigger names."""
        triggers = []
        bullish_consensus = consensus_score > 20

        # Trigger 1: Levered balance sheet while agents are bullish
        if metrics.debt_to_equity > 1.5 and bullish_consensus:
            triggers.append("debt_equity_bullish")

        # Trigger 2: Gross margin structural weakness
        # Proxy for "declining > 15% over 3 years" — only current period available
        if metrics.gross_margin < 0.20:
            triggers.append("margin_compression")

        # Trigger 3: Trading at > 2x sector average P/E
        if (
            metrics.pe_ratio > 0
            and metrics.sector_pe > 0
            and metrics.pe_ratio > 2 * metrics.sector_pe
        ):
            triggers.append("overvalued_vs_sector")

        # Trigger 4: Negative free cash flow
        # Proxy for "2+ consecutive years" — only current period available
        if metrics.fcf_per_share < 0:
            triggers.append("fcf_negative")

        # Trigger 5: Revenue growth deceleration while agents are bullish
        # Fires when current YoY growth lags 5Y CAGR by > 20 percentage points
        if (
            bullish_consensus
            and (metrics.revenue_growth_5y - metrics.revenue_growth) > 0.20
        ):
            triggers.append("growth_deceleration_bullish")

        return triggers

    def _calculate_score(self, triggers: List[str], consensus_score: float) -> float:
        if not triggers:
            # No red flags — agree with consensus, mild skepticism haircut
            return consensus_score * 0.85 - 5
        # Each trigger penalizes 18 points against the consensus base
        penalty = len(triggers) * 18
        return consensus_score - penalty

    def _score_to_verdict(self, score: float) -> str:
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

    # ------------------------------------------------------------------
    # LLM calls — 2 separate calls (insights, concerns)
    # ------------------------------------------------------------------

    def _build_agent_summary(self, agent_results: List[Dict[str, Any]]) -> str:
        """Summarize other agents' positions for LLM prompt context."""
        parts = []
        for r in agent_results:
            name = r.get("agent", "Unknown")
            score = r.get("score", 0)
            verdict = r.get("verdict", "hold").replace("_", " ").upper()
            insight = r.get("insights", [""])[0] if r.get("insights") else ""
            parts.append(f"- {name}: {verdict} ({score:+.0f}) — \"{insight}\"")
        return "\n".join(parts)

    def _build_trigger_summary(self, metrics: BurryMetrics, triggers: List[str]) -> str:
        descriptions = {
            "debt_equity_bullish": f"Debt/equity {metrics.debt_to_equity:.2f} (>1.5) while consensus is bullish",
            "margin_compression": f"Gross margin {metrics.gross_margin:.1%} below 20% structural floor",
            "overvalued_vs_sector": f"P/E {metrics.pe_ratio:.1f}x vs sector avg {metrics.sector_pe:.1f}x (>{2*metrics.sector_pe:.1f}x threshold)",
            "fcf_negative": f"Free cash flow per share {metrics.fcf_per_share:.2f} (negative)",
            "growth_deceleration_bullish": (
                f"Revenue growth {metrics.revenue_growth:.1%} YoY vs "
                f"{metrics.revenue_growth_5y:.1%} 5Y CAGR — decelerating >20pp while consensus is bullish"
            ),
        }
        return "\n".join(f"• {descriptions[t]}" for t in triggers if t in descriptions)

    async def _generate_llm_insights(
        self,
        ticker: str,
        metrics: BurryMetrics,
        triggers: List[str],
        agent_results: List[Dict[str, Any]],
        score: float,
        verdict: str,
        *,
        recent_news: Optional[List[Dict]] = None,
        config: dict = None,
    ) -> List[str]:
        agent_summary = self._build_agent_summary(agent_results)
        if triggers:
            trigger_summary = self._build_trigger_summary(metrics, triggers)
            prompt = (
                f"You are reviewing {ticker}. The other analysts concluded:\n{agent_summary}\n\n"
                f"You found quantitative red flags they missed:\n{trigger_summary}\n\n"
                f"In 2-3 sentences, explain what everyone is missing — be specific about the numbers "
                f"and name the analysts whose bullish thesis is most at risk."
            )
        else:
            prompt = (
                f"You are reviewing {ticker}. The other analysts concluded:\n{agent_summary}\n\n"
                f"The numbers check out — no major red flags. In 1-2 sentences, acknowledge the "
                f"consensus and note one understated risk worth watching, even if not a dealbreaker."
            )
        if recent_news:
            news_lines = ["Recent News (last 10 items):"]
            for item in recent_news:
                line = f"- {item['headline']} — {item['source']}, {item['published']}"
                if item.get("summary"):
                    line += f"\n  {item['summary']}"
                news_lines.append(line)
            prompt += "\n\n" + "\n".join(news_lines)
        try:
            response = await self.llm_client.analyze(
                prompt, persona="Michael Burry", verdict=verdict, config=config
            )
            return [response] if response else self._generate_insights(metrics, triggers, agent_results)
        except Exception as e:
            print(f"[{self.name}] LLM insight generation failed: {e}")
            traceback.print_exc()
            return self._generate_insights(metrics, triggers, agent_results)

    async def _generate_llm_concerns(
        self,
        ticker: str,
        metrics: BurryMetrics,
        triggers: List[str],
        agent_results: List[Dict[str, Any]],
        score: float,
        verdict: str,
        *,
        recent_news: Optional[List[Dict]] = None,
        config: dict = None,
    ) -> List[str]:
        if triggers:
            trigger_summary = self._build_trigger_summary(metrics, triggers)
            prompt = (
                f"For {ticker}, these quantitative signals are flashing red:\n{trigger_summary}\n\n"
                f"In 2-3 sentences, describe the real downside risk that nobody is talking about. "
                f"Be specific — reference the actual metrics and what they imply for the investment thesis."
            )
        else:
            prompt = (
                f"For {ticker}, the fundamentals look acceptable. In 1-2 sentences, name one "
                f"tail risk that could invalidate the bull case — something subtle, not obvious."
            )
        if recent_news:
            news_lines = ["Recent News (last 10 items):"]
            for item in recent_news:
                line = f"- {item['headline']} — {item['source']}, {item['published']}"
                if item.get("summary"):
                    line += f"\n  {item['summary']}"
                news_lines.append(line)
            prompt += "\n\n" + "\n".join(news_lines)
        try:
            response = await self.llm_client.analyze(
                prompt, persona="Michael Burry", verdict=verdict, config=config
            )
            return [response] if response else self._identify_concerns(metrics, triggers)
        except Exception as e:
            print(f"[{self.name}] LLM concern generation failed: {e}")
            traceback.print_exc()
            return self._identify_concerns(metrics, triggers)

    # ------------------------------------------------------------------
    # Rule-based fallbacks (no LLM)
    # ------------------------------------------------------------------

    def _generate_insights(
        self,
        metrics: BurryMetrics,
        triggers: List[str],
        agent_results: List[Dict[str, Any]],
    ) -> List[str]:
        if not triggers:
            return ["Consensus appears supported by the numbers — no forensic red flags detected."]

        insights = []
        bullish = [r["agent"] for r in agent_results if r.get("score", 0) > 20]
        bullish_str = " and ".join(bullish[:2]) if bullish else "the bullish analysts"

        if "debt_equity_bullish" in triggers:
            insights.append(
                f"Debt/equity of {metrics.debt_to_equity:.2f} is a hidden risk "
                f"{bullish_str} are overlooking — leverage amplifies downside in any stress scenario."
            )
        if "overvalued_vs_sector" in triggers:
            insights.append(
                f"At {metrics.pe_ratio:.1f}x P/E vs a sector average of {metrics.sector_pe:.1f}x, "
                f"the valuation leaves no margin of safety."
            )
        if "growth_deceleration_bullish" in triggers:
            insights.append(
                f"Revenue growth has decelerated from {metrics.revenue_growth_5y:.1%} (5Y) "
                f"to {metrics.revenue_growth:.1%} (current) — the growth story is breaking down."
            )
        return insights or ["Contrarian triggers fired — review triggers_fired for specifics."]

    def _identify_concerns(
        self,
        metrics: BurryMetrics,
        triggers: List[str],
    ) -> List[str]:
        if not triggers:
            return ["Monitor for any deterioration in FCF or margin trends."]

        concerns = []
        if "margin_compression" in triggers:
            concerns.append(
                f"Gross margin of {metrics.gross_margin:.1%} signals structural cost pressure "
                f"that could widen if revenue growth slows further."
            )
        if "fcf_negative" in triggers:
            concerns.append(
                f"Negative free cash flow ({metrics.fcf_per_share:.2f}/share) means the company "
                f"is burning cash — accounting earnings may be masking the true economic picture."
            )
        return concerns or ["Quantitative red flags present — see triggers_fired."]
