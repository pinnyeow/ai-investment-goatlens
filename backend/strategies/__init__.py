"""
GOATlens - Strategy Evaluation Logic

Contains the core evaluation strategies and scoring mechanisms.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    """Investment verdict from an agent."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class ConsensusVerdict(Enum):
    """Aggregate verdict; mixed is valid only for consensus."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    MIXED = "mixed"


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
    consensus_verdict: ConsensusVerdict
    agreement_score: Optional[float]  # Share supporting the consensus direction
    consensus_points: List[str]
    divergence_points: List[str]
    individual_results: List[StrategyResult]


def calculate_ordinal_consensus(
    verdicts: List[Verdict],
) -> Tuple[Optional[int], ConsensusVerdict]:
    """Return the conservative ordinal median and aggregate verdict."""
    if not verdicts:
        raise ValueError("No verdicts to calculate consensus from")

    verdict_value = {
        Verdict.STRONG_SELL: -2,
        Verdict.SELL: -1,
        Verdict.HOLD: 0,
        Verdict.BUY: 1,
        Verdict.STRONG_BUY: 2,
    }
    value_verdict = {
        -2: ConsensusVerdict.STRONG_SELL,
        -1: ConsensusVerdict.SELL,
        0: ConsensusVerdict.HOLD,
        1: ConsensusVerdict.BUY,
        2: ConsensusVerdict.STRONG_BUY,
    }

    def direction(value: int) -> int:
        return 0 if value == 0 else 1 if value > 0 else -1

    ordered_votes = sorted(verdict_value[verdict] for verdict in verdicts)
    midpoint = len(ordered_votes) // 2
    if len(ordered_votes) % 2:
        consensus_value = ordered_votes[midpoint]
        return consensus_value, value_verdict[consensus_value]

    lower = ordered_votes[midpoint - 1]
    upper = ordered_votes[midpoint]
    if direction(lower) != direction(upper):
        return None, ConsensusVerdict.MIXED

    # A same-direction intensity split resolves conservatively toward HOLD.
    consensus_value = int((lower + upper) / 2)
    return consensus_value, value_verdict[consensus_value]


def calculate_consensus(results: List[StrategyResult]) -> ConsensusResult:
    """
    Calculate consensus from multiple agent evaluations.

    Identifies where GOATs agree vs. where they'd debate.
    """
    if not results:
        raise ValueError("No results to calculate consensus from")

    verdict_value = {
        Verdict.STRONG_SELL: -2,
        Verdict.SELL: -1,
        Verdict.HOLD: 0,
        Verdict.BUY: 1,
        Verdict.STRONG_BUY: 2,
    }

    def direction(value: int) -> int:
        return 0 if value == 0 else 1 if value > 0 else -1

    ordered_votes = sorted(verdict_value[result.verdict] for result in results)
    consensus_value, consensus_verdict = calculate_ordinal_consensus(
        [result.verdict for result in results]
    )

    if consensus_value is None:
        agreement_score = None
    else:
        consensus_direction = direction(consensus_value)
        agreeing_votes = sum(
            direction(value) == consensus_direction for value in ordered_votes
        )
        agreement_score = agreeing_votes / len(ordered_votes)
    
    # Find consensus and divergence points
    all_insights = [insight for r in results for insight in r.key_insights]

    # Simple frequency-based consensus (insights mentioned by multiple agents)
    insight_counts: Dict[str, int] = {}
    for insight in all_insights:
        insight_counts[insight] = insight_counts.get(insight, 0) + 1
    
    consensus_points = [i for i, c in insight_counts.items() if c >= len(results) // 2 + 1]
    
    # Divergence: where verdicts differ significantly
    verdicts = [r.verdict for r in results]
    divergence_points = []
    direction_counts = {
        candidate: sum(direction(value) == candidate for value in ordered_votes)
        for candidate in (-1, 0, 1)
    }
    if consensus_verdict is ConsensusVerdict.MIXED:
        divergence_points.append("Mixed investor verdicts; no consensus direction")
    elif max(direction_counts.values()) <= len(results) / 2:
        divergence_points.append("Mixed investor verdicts; no majority direction")
    elif len(set(verdicts)) > 2:
        divergence_points.append("Significant disagreement on investment thesis")
    
    return ConsensusResult(
        consensus_verdict=consensus_verdict,
        agreement_score=agreement_score,
        consensus_points=consensus_points,
        divergence_points=divergence_points,
        individual_results=results,
    )


async def calculate_consensus_with_llm(results: List[StrategyResult], llm_client, ticker: str, config: dict = None) -> ConsensusResult:
    """LLM-powered consensus with rich insights.
    
    Args:
        results: Individual agent strategy results
        llm_client: LLM client for generating consensus narrative
        ticker: Stock ticker symbol
        config: LangChain RunnableConfig for trace propagation.
                When provided, the consensus LLM span nests under the
                caller's trace via LangChain's parent_run_id mechanism.
    """
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
        response = await llm_client.analyze(prompt, persona="analyst", verdict=base.consensus_verdict.value, config=config)
        consensus_points, divergence_points = [], []
        for line in response.split("\n"):
            clean = line.strip()
            if clean.startswith("AGREE"):
                consensus_points.append(clean.split(":", 1)[-1].strip())
            elif clean.startswith("DEBATE"):
                divergence_points.append(clean.split(":", 1)[-1].strip())
        consensus_points = consensus_points or base.consensus_points
        divergence_points = list(
            dict.fromkeys(base.divergence_points + divergence_points)
        )
    except Exception:
        print("[Consensus] LLM consensus generation failed.")
        consensus_points, divergence_points = base.consensus_points, base.divergence_points
    
    return ConsensusResult(
        consensus_verdict=base.consensus_verdict,
        agreement_score=base.agreement_score,
        consensus_points=consensus_points[:2],
        divergence_points=divergence_points[:2],
        individual_results=results,
    )
