"""
Layer 3: Consensus Evals — Checks that the synthesis step is coherent.

The consensus combines 5 agents into one recommendation. If the synthesis
is broken, the user gets a contradictory "final answer."

Code-based checks:
  - Agreement score is valid (0-1 range)
  - Consensus verdict aligns with individual verdicts
  - No contradictions between agent majority and consensus

LLM-as-Judge:
  - Are consensus points coherent given the agent results?

WHY SEPARATE FROM LAYER 2:
Layer 2 checks each agent independently. Layer 3 checks the
*combination*. You can have 5 perfect agents and a broken consensus.
"""

import json
from typing import Dict, Any, List, Optional

from .data_pipeline_evals import EvalResult, LayerResults


# ─── Code-Based Consensus Checks ───


def eval_agreement_score_valid(consensus: Dict[str, Any]) -> List[EvalResult]:
    """
    Check that agreement_score is between 0.0 and 1.0.
    
    WHY: Agreement score represents how much agents agree.
    0 = total disagreement, 1 = unanimous. Outside this range = math bug.
    """
    score = consensus.get("agreement_score")
    if score is None:
        return [EvalResult(
            name="agreement_score_exists",
            passed=False,
            details="No agreement_score in consensus",
        )]

    ok = isinstance(score, (int, float)) and 0 <= score <= 1
    return [EvalResult(
        name="agreement_score_valid",
        passed=ok,
        details=f"agreement_score={score}, expected [0, 1]" if not ok else "",
    )]


def eval_consensus_matches_majority(
    consensus: Dict[str, Any],
    agent_outputs: List[Dict[str, Any]],
) -> List[EvalResult]:
    """
    Check that the consensus verdict aligns with the majority of agents.
    
    WHY: If 4 out of 5 agents say "sell" but consensus says "strong_buy",
    something is fundamentally wrong. The consensus should reflect the
    balance of opinions, not contradict them.
    
    We allow some flexibility: the consensus can differ by one tier
    (e.g., agents mostly say "buy" and consensus says "hold" is OK).
    But a 3-tier gap is a red flag.
    """
    verdict_order = ["strong_sell", "sell", "hold", "buy", "strong_buy"]

    consensus_verdict = consensus.get("verdict", "")
    if consensus_verdict not in verdict_order:
        return [EvalResult(
            name="consensus_verdict_known",
            passed=False,
            details=f"Unknown consensus verdict: '{consensus_verdict}'",
        )]

    # Get agent verdicts
    agent_verdicts = [o.get("verdict", "") for o in agent_outputs]
    valid_verdicts = [v for v in agent_verdicts if v in verdict_order]

    if not valid_verdicts:
        return [EvalResult(
            name="consensus_has_agent_verdicts",
            passed=False,
            details="No valid agent verdicts to compare",
        )]

    # Calculate average position
    avg_position = sum(verdict_order.index(v) for v in valid_verdicts) / len(valid_verdicts)
    consensus_position = verdict_order.index(consensus_verdict)

    # Allow ±1.5 tiers of difference
    gap = abs(consensus_position - avg_position)
    ok = gap <= 1.5

    return [EvalResult(
        name="consensus_matches_majority",
        passed=ok,
        details=(
            f"Consensus '{consensus_verdict}' (pos {consensus_position}) vs "
            f"avg agent position {avg_position:.1f}. Gap: {gap:.1f} tiers"
        ) if not ok else "",
    )]


def eval_no_extreme_contradiction(
    consensus: Dict[str, Any],
    agent_outputs: List[Dict[str, Any]],
) -> List[EvalResult]:
    """
    Check for extreme contradictions: all agents agree on direction
    but consensus goes opposite.
    
    Example: All agents say sell/strong_sell, consensus says buy/strong_buy.
    This is a critical bug in the synthesis logic.
    """
    consensus_verdict = consensus.get("verdict", "")
    agent_verdicts = [o.get("verdict", "") for o in agent_outputs]

    positive = {"buy", "strong_buy"}
    negative = {"sell", "strong_sell"}

    all_positive = all(v in positive for v in agent_verdicts if v)
    all_negative = all(v in negative for v in agent_verdicts if v)

    contradicts = False
    detail = ""

    if all_positive and consensus_verdict in negative:
        contradicts = True
        detail = f"All agents are positive but consensus is '{consensus_verdict}'"
    elif all_negative and consensus_verdict in positive:
        contradicts = True
        detail = f"All agents are negative but consensus is '{consensus_verdict}'"

    return [EvalResult(
        name="no_extreme_contradiction",
        passed=not contradicts,
        details=detail,
    )]


def eval_consensus_points_exist(consensus: Dict[str, Any]) -> List[EvalResult]:
    """
    Check that consensus has at least one consensus or divergence point.
    
    WHY: The consensus section with no points looks empty to the user.
    The LLM should always find something agents agree or disagree on.
    """
    consensus_points = consensus.get("consensus_points", [])
    divergence_points = consensus.get("divergence_points", [])

    total = len(consensus_points) + len(divergence_points)

    return [EvalResult(
        name="consensus_has_points",
        passed=total > 0,
        details="No consensus or divergence points" if total == 0 else "",
        severity="warning",
    )]


# ─── LLM-as-Judge for Consensus ───


CONSENSUS_COHERENCE_PROMPT = """You are evaluating the synthesis of a multi-agent investment analysis.

Five investment agents analyzed the same company. A consensus was generated.

[BEGIN DATA]
************
[Company]: {ticker}
[Agent Summaries]:
{agent_summaries}

[Consensus Verdict]: {consensus_verdict}
[Agreement Score]: {agreement_score}
[Consensus Points]: {consensus_points}
[Divergence Points]: {divergence_points}
[END DATA]

Evaluate whether the consensus is COHERENT given the agent results:
- Do the consensus points actually reflect what agents agreed on?
- Do the divergence points capture real disagreements?
- Is the consensus verdict a reasonable summary of the 5 views?

Your response must be EXACTLY one word on the first line:
coherent
incoherent

Then on the next line, provide a brief explanation (1-2 sentences)."""


async def eval_consensus_coherence(
    consensus: Dict[str, Any],
    agent_outputs: List[Dict[str, Any]],
    ticker: str,
    llm_client,
) -> List[EvalResult]:
    """
    LLM-as-Judge: Check if consensus synthesis is coherent.
    
    The judge reads all agent outputs and the consensus, then decides
    if the consensus actually makes sense given the agent results.
    """
    # Build agent summaries
    summaries = []
    for output in agent_outputs:
        name = output.get("agent", "Unknown")
        verdict = output.get("verdict", "unknown")
        score = output.get("score", 0)
        top_insight = output.get("insights", ["N/A"])[0] if output.get("insights") else "N/A"
        summaries.append(f"  {name} ({verdict}, score {score}): {top_insight[:100]}")

    prompt = CONSENSUS_COHERENCE_PROMPT.format(
        ticker=ticker,
        agent_summaries="\n".join(summaries),
        consensus_verdict=consensus.get("verdict", "unknown"),
        agreement_score=consensus.get("agreement_score", "N/A"),
        consensus_points="\n".join(f"  - {p}" for p in consensus.get("consensus_points", [])),
        divergence_points="\n".join(f"  - {p}" for p in consensus.get("divergence_points", [])),
    )

    try:
        response = await llm_client.analyze(
            prompt,
            persona="evaluator",
            verdict="hold",
        )
        first_line = response.strip().split("\n")[0].lower()
        passed = "coherent" in first_line and "incoherent" not in first_line
        return [EvalResult(
            name=f"consensus_coherence:{ticker}",
            passed=passed,
            details=response if not passed else "",
        )]
    except Exception as e:
        return [EvalResult(
            name=f"consensus_coherence:{ticker}",
            passed=False,
            details=f"LLM judge error: {e}",
            severity="warning",
        )]


# ─── Main Runner ───


async def run_consensus_evals(
    consensus: Dict[str, Any],
    agent_outputs: List[Dict[str, Any]],
    ticker: str,
    llm_client=None,
) -> LayerResults:
    """
    Run all Layer 3 consensus evals.
    
    Args:
        consensus: The consensus dict from the synthesis step
        agent_outputs: List of individual agent output dicts
        ticker: The ticker symbol
        llm_client: Optional LLM client for coherence check
    
    Returns:
        LayerResults with all check results
    """
    layer = LayerResults(layer="Layer 3: Consensus Quality")

    # Code-based checks (always run)
    layer.results.extend(eval_agreement_score_valid(consensus))
    layer.results.extend(eval_consensus_matches_majority(consensus, agent_outputs))
    layer.results.extend(eval_no_extreme_contradiction(consensus, agent_outputs))
    layer.results.extend(eval_consensus_points_exist(consensus))

    # LLM-as-judge (only if API key available)
    if llm_client:
        layer.results.extend(
            await eval_consensus_coherence(consensus, agent_outputs, ticker, llm_client)
        )

    return layer
