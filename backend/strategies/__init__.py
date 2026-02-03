"""
GOATlens - Strategy Evaluation Logic

Contains the core evaluation strategies and scoring mechanisms.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    """Investment verdict from an agent."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class StrategyResult:
    """Result from a single strategy evaluation."""
    agent_name: str
    verdict: Verdict
    confidence: float  # 0.0 - 1.0
    key_insights: List[str]
    concerns: List[str]
    moat_assessment: str
    score: float  # -100 to +100


@dataclass
class ConsensusResult:
    """Aggregated result from all agents."""
    consensus_verdict: Verdict
    agreement_score: float  # 0.0 - 1.0 (how much agents agree)
    consensus_points: List[str]
    divergence_points: List[str]
    individual_results: List[StrategyResult]


def calculate_consensus(results: List[StrategyResult]) -> ConsensusResult:
    """
    Calculate consensus from multiple agent evaluations.
    
    Identifies where GOATs agree vs. where they'd debate.
    """
    if not results:
        raise ValueError("No results to calculate consensus from")
    
    # Calculate average score
    avg_score = sum(r.score for r in results) / len(results)
    
    # Determine consensus verdict based on average score
    if avg_score >= 60:
        consensus_verdict = Verdict.STRONG_BUY
    elif avg_score >= 20:
        consensus_verdict = Verdict.BUY
    elif avg_score >= -20:
        consensus_verdict = Verdict.HOLD
    elif avg_score >= -60:
        consensus_verdict = Verdict.SELL
    else:
        consensus_verdict = Verdict.STRONG_SELL
    
    # Calculate agreement score (inverse of variance)
    variance = sum((r.score - avg_score) ** 2 for r in results) / len(results)
    max_variance = 10000  # (100 - (-100))^2 / 4
    agreement_score = max(0, 1 - (variance / max_variance))
    
    # Find consensus and divergence points
    all_insights = [insight for r in results for insight in r.key_insights]
    all_concerns = [concern for r in results for concern in r.concerns]
    
    # Simple frequency-based consensus (insights mentioned by multiple agents)
    insight_counts: Dict[str, int] = {}
    for insight in all_insights:
        insight_counts[insight] = insight_counts.get(insight, 0) + 1
    
    consensus_points = [i for i, c in insight_counts.items() if c >= len(results) // 2 + 1]
    
    # Divergence: where verdicts differ significantly
    verdicts = [r.verdict for r in results]
    divergence_points = []
    if len(set(verdicts)) > 2:
        divergence_points.append("Significant disagreement on investment thesis")
    
    return ConsensusResult(
        consensus_verdict=consensus_verdict,
        agreement_score=agreement_score,
        consensus_points=consensus_points,
        divergence_points=divergence_points,
        individual_results=results,
    )


async def calculate_consensus_with_llm(results: List[StrategyResult], llm_client, ticker: str) -> ConsensusResult:
    """LLM-powered consensus with rich insights."""
    base = calculate_consensus(results)
    
    summaries = "\n".join([
        f"{r.agent_name} ({r.verdict.value}): {r.key_insights[0][:80] if r.key_insights else 'N/A'}"
        for r in results
    ])
    
    prompt = f"""{ticker} analysis:
{summaries}

AGREE1: [one sentence, <30 words, where 2+ investors share a view]
AGREE2: [one sentence, <30 words, another shared view]
DEBATE1: [one sentence, <30 words, where investors disagree]
DEBATE2: [one sentence, <30 words, another disagreement]"""

    try:
        response = await llm_client.analyze(prompt, persona="analyst", verdict=base.consensus_verdict.value)
        consensus_points, divergence_points = [], []
        for line in response.split("\n"):
            clean = line.strip()
            if clean.startswith("AGREE"):
                consensus_points.append(clean.split(":", 1)[-1].strip())
            elif clean.startswith("DEBATE"):
                divergence_points.append(clean.split(":", 1)[-1].strip())
        consensus_points = consensus_points or base.consensus_points
        divergence_points = divergence_points or base.divergence_points
    except Exception:
        consensus_points, divergence_points = base.consensus_points, base.divergence_points
    
    return ConsensusResult(
        consensus_verdict=base.consensus_verdict,
        agreement_score=base.agreement_score,
        consensus_points=consensus_points[:2],
        divergence_points=divergence_points[:2],
        individual_results=results,
    )
