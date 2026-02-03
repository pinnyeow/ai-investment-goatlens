"""
GOATlens - Strategy Evaluation Logic

Contains the core evaluation strategies and scoring mechanisms.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from collections import Counter


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
    
    # Group verdicts by direction (bullish/neutral/bearish)
    def get_direction(v):
        if v in (Verdict.STRONG_BUY, Verdict.BUY):
            return "bullish"
        elif v == Verdict.HOLD:
            return "neutral"
        else:
            return "bearish"
    
    directions = [get_direction(r.verdict) for r in results]
    direction_counts = Counter(directions)
    most_common_count = direction_counts.most_common(1)[0][1]
    agreement_score = most_common_count / len(results)
    
    # Divergence: where directions differ
    divergence_points = []
    unique_directions = set(directions)
    if len(unique_directions) > 1:
        divergence_points.append(f"Agents split between {' and '.join(unique_directions)}")
    
    consensus_points = [] if len(unique_directions) > 1 else ["All agents agree on direction"]
    
    return ConsensusResult(
        consensus_verdict=consensus_verdict,
        agreement_score=agreement_score,
        consensus_points=consensus_points,
        divergence_points=divergence_points,
        individual_results=results,
    )
