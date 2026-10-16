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

import math
from typing import Any, Dict, List, Optional

from .data_pipeline_evals import EvalResult, LayerResults
from strategies import Verdict, calculate_ordinal_consensus


# ─── Code-Based Consensus Checks ───


def _completed_vote_contract(consensus: Dict[str, Any]) -> Dict[str, Any]:
    """Return the final consensus or subordinate completed-vote indication."""
    if consensus.get("status") != "partial":
        return consensus
    indication = consensus.get("partial_indication")
    if not isinstance(indication, dict):
        return consensus
    return {
        **indication,
        "agreement_score": indication.get("directional_support"),
    }


def _completed_agent_outputs(
    agent_outputs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        output
        for output in agent_outputs
        if output.get("status", "complete") == "complete"
    ]


def eval_consensus_structural_integrity(
    consensus: Dict[str, Any],
    agent_outputs: List[Dict[str, Any]],
) -> List[EvalResult]:
    """Recompute quorum and partial-result structure from agent outputs."""
    completed_outputs = _completed_agent_outputs(agent_outputs)
    actual_requested = len(agent_outputs)
    actual_completed = len(completed_outputs)
    declared_requested = consensus.get("requested_analyses")
    declared_completed = consensus.get("completed_analyses")
    status = consensus.get("status")
    indication = consensus.get("partial_indication")

    error = ""
    agent_names = [output.get("agent") for output in agent_outputs]
    if (
        any(not isinstance(name, str) or not name for name in agent_names)
        or len(agent_names) != len(set(agent_names))
    ):
        error = "Agent results must contain unique non-empty identities"
    elif (
        declared_requested != actual_requested
        or declared_completed != actual_completed
    ):
        error = (
            "Declared analysis counts do not match agent results: "
            f"declared {declared_completed}/{declared_requested}, "
            f"actual {actual_completed}/{actual_requested}"
        )
    elif status == "complete":
        if actual_completed < 2 or actual_completed != actual_requested:
            error = (
                "A final consensus requires every requested agent to complete "
                "and at least two completed analyses"
            )
        elif indication is not None:
            error = "A final consensus cannot include a partial indication"
    elif status == "partial":
        if not (2 <= actual_completed < actual_requested):
            error = "A partial response requires 2 <= completed < requested"
        elif not isinstance(indication, dict):
            error = "A partial response requires a subordinate indication"
        elif (
            consensus.get("verdict") != "incomplete"
            or consensus.get("agreement_score") is not None
            or bool(consensus.get("consensus_points"))
        ):
            error = "A partial response cannot publish headline consensus fields"
        elif (
            indication.get("completed_analyses") != actual_completed
            or indication.get("requested_analyses") != actual_requested
        ):
            error = "Partial indication counts do not match agent results"
        else:
            expected_excluded = [
                {
                    "agent": output.get("agent"),
                    "unavailable_metrics": output.get("unavailable_metrics", []),
                }
                for output in agent_outputs
                if output.get("status", "complete") != "complete"
            ]
            if indication.get("excluded_agents") != expected_excluded:
                error = (
                    "Partial indication excluded agents must exactly match "
                    "incomplete agent identities and unavailable metrics"
                )
            else:
                direction_by_verdict = {
                    "strong_sell": -1,
                    "sell": -1,
                    "hold": 0,
                    "buy": 1,
                    "strong_buy": 1,
                }
                indication_verdict = indication.get("verdict")
                if indication_verdict == "mixed":
                    expected_supporting = 0
                elif indication_verdict in direction_by_verdict:
                    expected_direction = direction_by_verdict[indication_verdict]
                    expected_supporting = sum(
                        direction_by_verdict.get(str(output.get("verdict", "")))
                        == expected_direction
                        for output in completed_outputs
                    )
                else:
                    expected_supporting = None
                if indication.get("supporting_analyses") != expected_supporting:
                    error = (
                        "Partial indication supporting analyses must exactly match "
                        "completed votes on the indicated direction"
                    )
    elif status == "incomplete":
        if actual_completed >= 2:
            error = "An incomplete response requires fewer than two completions"
        elif (
            consensus.get("verdict") != "incomplete"
            or consensus.get("agreement_score") is not None
            or indication is not None
            or bool(consensus.get("consensus_points"))
        ):
            error = (
                "An incomplete response cannot publish a verdict, support, "
                "partial indication, or consensus points"
            )
    else:
        error = f"Unknown consensus status: {status!r}"

    return [EvalResult(
        name="consensus_structural_integrity",
        passed=not error,
        details=error,
    )]


def _fewer_than_two_contract_result(
    check_name: str,
    consensus: Dict[str, Any],
    agent_outputs: List[Dict[str, Any]],
) -> Optional[EvalResult]:
    """Abstain only when an unavailable consensus obeys the integrity contract."""
    if len(_completed_agent_outputs(agent_outputs)) >= 2:
        return None
    valid_abstention = (
        consensus.get("status") == "incomplete"
        and consensus.get("verdict") == "incomplete"
        and consensus.get("agreement_score") is None
        and consensus.get("partial_indication") is None
        and not consensus.get("consensus_points")
    )
    suffix = (
        "abstain_fewer_than_two"
        if valid_abstention
        else "integrity_fewer_than_two"
    )
    return EvalResult(
        name=f"{check_name}_{suffix}",
        passed=valid_abstention,
        details=(
            ""
            if valid_abstention
            else "Fewer than two completed analyses must not publish consensus"
        ),
    )


def eval_agreement_score_valid(
    consensus: Dict[str, Any],
    agent_outputs: Optional[List[Dict[str, Any]]] = None,
) -> List[EvalResult]:
    """
    Check the legacy agreement_score directional-support contract.

    Mixed consensus has no direction and therefore requires a null value.
    Other verdicts require a finite numeric share between 0.0 and 1.0.
    """
    if agent_outputs is not None:
        abstention = _fewer_than_two_contract_result(
            "agreement_score", consensus, agent_outputs
        )
        if abstention is not None:
            return [abstention]
    evaluated = _completed_vote_contract(consensus)
    score = evaluated.get("agreement_score")
    verdict = evaluated.get("verdict")
    if verdict == "mixed":
        return [EvalResult(
            name="mixed_directional_support_unavailable",
            passed=score is None,
            details="Mixed consensus must not report directional support"
            if score is not None else "",
        )]
    if score is None:
        return [EvalResult(
            name="agreement_score_exists",
            passed=False,
            details="No directional support for a directional consensus",
        )]

    ok = (
        not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(score)
        and 0 <= score <= 1
    )
    return [EvalResult(
        name="agreement_score_valid",
        passed=ok,
        details=f"agreement_score={score}, expected [0, 1]" if not ok else "",
    )]


def eval_directional_support_matches_agents(
    consensus: Dict[str, Any],
    agent_outputs: List[Dict[str, Any]],
) -> List[EvalResult]:
    """Independently verify directional support against completed verdicts."""
    abstention = _fewer_than_two_contract_result(
        "directional_support", consensus, agent_outputs
    )
    if abstention is not None:
        return [abstention]
    evaluated = _completed_vote_contract(consensus)
    verdict = evaluated.get("verdict")
    score = evaluated.get("agreement_score")
    completed_outputs = _completed_agent_outputs(agent_outputs)
    if verdict == "mixed":
        expected = None
    else:
        direction_by_verdict = {
            "strong_sell": -1,
            "sell": -1,
            "hold": 0,
            "buy": 1,
            "strong_buy": 1,
        }
        if verdict not in direction_by_verdict:
            return [EvalResult(
                name="directional_support_consensus_known",
                passed=False,
                details=f"Cannot verify support for verdict '{verdict}'",
            )]
        valid_agent_verdicts: List[str] = []
        for output in completed_outputs:
            candidate = output.get("verdict")
            if isinstance(candidate, str) and candidate in direction_by_verdict:
                valid_agent_verdicts.append(candidate)
        if not valid_agent_verdicts:
            return [EvalResult(
                name="directional_support_has_agent_verdicts",
                passed=False,
                details="No valid agent verdicts for directional support",
            )]
        consensus_direction = direction_by_verdict[verdict]
        expected = sum(
            direction_by_verdict[agent_verdict] == consensus_direction
            for agent_verdict in valid_agent_verdicts
        ) / len(valid_agent_verdicts)

    matches = (
        score is None
        if expected is None
        else not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(score)
        and math.isclose(score, expected, rel_tol=0.0, abs_tol=1e-9)
    )
    return [EvalResult(
        name="directional_support_matches_agents",
        passed=matches,
        details=f"directional support={score}, expected {expected}" if not matches else "",
    )]


def eval_consensus_matches_majority(
    consensus: Dict[str, Any],
    agent_outputs: List[Dict[str, Any]],
) -> List[EvalResult]:
    """
    Check the consensus against the independent ordinal-median contract.
    """
    abstention = _fewer_than_two_contract_result(
        "consensus_majority", consensus, agent_outputs
    )
    if abstention is not None:
        return [abstention]
    verdict_order = ["strong_sell", "sell", "hold", "buy", "strong_buy"]

    evaluated = _completed_vote_contract(consensus)
    consensus_verdict = evaluated.get("verdict", "")
    if consensus_verdict not in {*verdict_order, "mixed"}:
        return [EvalResult(
            name="consensus_verdict_known",
            passed=False,
            details=f"Unknown consensus verdict: '{consensus_verdict}'",
        )]

    # Get completed agent verdicts
    agent_verdicts = [
        output.get("verdict", "")
        for output in _completed_agent_outputs(agent_outputs)
    ]
    valid_verdicts = [v for v in agent_verdicts if v in verdict_order]

    if not valid_verdicts:
        return [EvalResult(
            name="consensus_has_agent_verdicts",
            passed=False,
            details="No valid agent verdicts to compare",
        )]

    _, expected_consensus = calculate_ordinal_consensus(
        [Verdict(verdict) for verdict in valid_verdicts]
    )
    expected = expected_consensus.value
    ok = consensus_verdict == expected

    return [EvalResult(
        name="consensus_matches_majority",
        passed=ok,
        details=(
            f"Consensus '{consensus_verdict}' does not match expected "
            f"ordinal-median verdict '{expected}'"
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
    abstention = _fewer_than_two_contract_result(
        "no_extreme_contradiction", consensus, agent_outputs
    )
    if abstention is not None:
        return [abstention]
    evaluated = _completed_vote_contract(consensus)
    consensus_verdict = evaluated.get("verdict", "")
    agent_verdicts = [
        output.get("verdict", "")
        for output in _completed_agent_outputs(agent_outputs)
    ]

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


def eval_consensus_points_exist(
    consensus: Dict[str, Any],
    agent_outputs: Optional[List[Dict[str, Any]]] = None,
) -> List[EvalResult]:
    """
    Check that consensus has at least one consensus or divergence point.
    
    WHY: The consensus section with no points looks empty to the user.
    The LLM should always find something agents agree or disagree on.
    """
    if agent_outputs is not None:
        abstention = _fewer_than_two_contract_result(
            "consensus_points", consensus, agent_outputs
        )
        if abstention is not None:
            return [abstention]
    evaluated = _completed_vote_contract(consensus)
    consensus_points = evaluated.get("consensus_points", [])
    divergence_points = evaluated.get("divergence_points", [])

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
    abstention = _fewer_than_two_contract_result(
        "consensus_coherence", consensus, agent_outputs
    )
    if abstention is not None:
        return [abstention]
    evaluated = _completed_vote_contract(consensus)

    # Build agent summaries
    summaries = []
    for output in _completed_agent_outputs(agent_outputs):
        name = output.get("agent", "Unknown")
        verdict = output.get("verdict", "unknown")
        score = output.get("score", 0)
        top_insight = output.get("insights", ["N/A"])[0] if output.get("insights") else "N/A"
        summaries.append(f"  {name} ({verdict}, score {score}): {top_insight[:100]}")

    prompt = CONSENSUS_COHERENCE_PROMPT.format(
        ticker=ticker,
        agent_summaries="\n".join(summaries),
        consensus_verdict=evaluated.get("verdict", "unknown"),
        agreement_score=evaluated.get("agreement_score", "N/A"),
        consensus_points="\n".join(
            f"  - {point}" for point in evaluated.get("consensus_points", [])
        ),
        divergence_points="\n".join(
            f"  - {point}" for point in evaluated.get("divergence_points", [])
        ),
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
    layer.results.extend(eval_consensus_structural_integrity(consensus, agent_outputs))
    layer.results.extend(eval_agreement_score_valid(consensus, agent_outputs))
    layer.results.extend(
        eval_directional_support_matches_agents(consensus, agent_outputs)
    )
    layer.results.extend(eval_consensus_matches_majority(consensus, agent_outputs))
    layer.results.extend(eval_no_extreme_contradiction(consensus, agent_outputs))
    layer.results.extend(eval_consensus_points_exist(consensus, agent_outputs))

    # LLM-as-judge (only if API key available)
    if llm_client:
        layer.results.extend(
            await eval_consensus_coherence(consensus, agent_outputs, ticker, llm_client)
        )

    return layer
