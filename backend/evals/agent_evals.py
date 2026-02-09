"""
Layer 2: Agent Evals — Code-based structure checks + LLM-as-Judge quality checks.

This layer has TWO parts:

1. CODE-BASED (fast, free, deterministic):
   - Score is between -100 and +100
   - Score is rounded to 2 decimal places
   - Verdict matches score thresholds
   - Output contains all required keys
   - Score falls within expected range (golden dataset)

2. LLM-AS-JUDGE (nuanced, costs money, approximate):
   - Financial Grounding: Are claims traceable to metrics?
   - Philosophy Adherence: Does the agent stay in character?
   - Actionable Quality: Would a retail investor find this useful?

WHY BOTH?
Code-based catches structural bugs instantly. LLM-as-judge catches
quality problems that humans would notice but regex can't.
The Arize article calls this "layered evaluation" — cheap checks first,
expensive checks only when needed.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional

from .data_pipeline_evals import EvalResult, LayerResults
from .golden_dataset import (
    GOLDEN_TICKERS,
    PHILOSOPHY_KEYWORDS,
    VERDICT_THRESHOLDS,
    REQUIRED_OUTPUT_KEYS,
)


# ─── Code-Based Evals ───


def eval_output_structure(agent_output: Dict[str, Any]) -> List[EvalResult]:
    """
    Check that the agent output contains all required keys.
    
    WHY: If an agent returns without 'score' or 'insights', the frontend
    will crash or show empty cards. This catches missing keys early.
    """
    results = []
    for key in REQUIRED_OUTPUT_KEYS:
        present = key in agent_output
        results.append(EvalResult(
            name=f"has_key:{key}",
            passed=present,
            details=f"Missing required key: '{key}'" if not present else "",
        ))
    return results


def eval_score_range(agent_output: Dict[str, Any]) -> List[EvalResult]:
    """
    Check that the score is between -100 and +100 and rounded to 2 decimals.
    
    WHY: We had a bug where scores weren't rounded. Users saw 17.5150009999
    instead of 17.52. This eval catches both range violations and rounding.
    """
    results = []
    score = agent_output.get("score")

    if score is None:
        results.append(EvalResult(
            name="score_exists",
            passed=False,
            details="No score in output",
        ))
        return results

    # Score must be numeric
    if not isinstance(score, (int, float)):
        results.append(EvalResult(
            name="score_is_numeric",
            passed=False,
            details=f"Score is {type(score).__name__}, expected number",
        ))
        return results

    # Score must be in [-100, 100]
    in_range = -100 <= score <= 100
    results.append(EvalResult(
        name="score_in_range",
        passed=in_range,
        details=f"Score {score} outside [-100, 100]" if not in_range else "",
    ))

    # Score must be rounded to max 2 decimal places
    rounded = round(score, 2)
    is_rounded = score == rounded
    results.append(EvalResult(
        name="score_rounded_2dp",
        passed=is_rounded,
        details=f"Score {score} not rounded to 2dp (expected {rounded})" if not is_rounded else "",
    ))

    return results


def eval_verdict_consistency(agent_output: Dict[str, Any]) -> List[EvalResult]:
    """
    Check that the verdict matches the score based on known thresholds.
    
    WHY: If the score says 65 (strong_buy territory) but verdict says "sell",
    the user gets contradictory signals. This must be deterministic.
    """
    score = agent_output.get("score")
    verdict = agent_output.get("verdict")

    if score is None or verdict is None:
        return [EvalResult(
            name="verdict_consistency",
            passed=False,
            details="Missing score or verdict",
        )]

    # Determine expected verdict from score
    expected = None
    if score >= 60:
        expected = "strong_buy"
    elif score >= 20:
        expected = "buy"
    elif score >= -20:
        expected = "hold"
    elif score >= -60:
        expected = "sell"
    else:
        expected = "strong_sell"

    ok = verdict == expected
    return [EvalResult(
        name="verdict_consistency",
        passed=ok,
        details=f"Score={score}, verdict='{verdict}', expected='{expected}'" if not ok else "",
    )]


def eval_golden_score_range(
    agent_output: Dict[str, Any],
    ticker: str,
) -> List[EvalResult]:
    """
    Check that the agent's score falls within the expected range for known tickers.
    
    WHY: If a code change suddenly makes Buffett hate Apple, something broke.
    The golden dataset captures invariants like "Buffett should always score
    Apple positively." This is regression detection.
    """
    ticker_upper = ticker.upper()
    agent_name = agent_output.get("agent", "").lower()

    # Extract the base agent name (e.g., "Warren Buffett" -> "buffett")
    agent_key = None
    for key in GOLDEN_TICKERS.get(ticker_upper, {}).keys():
        if key in agent_name.lower():
            agent_key = key
            break

    if agent_key is None or ticker_upper not in GOLDEN_TICKERS:
        return []  # Not in golden dataset, skip

    golden = GOLDEN_TICKERS[ticker_upper][agent_key]
    score = agent_output.get("score", 0)
    lo, hi = golden["score_range"]

    ok = lo <= score <= hi
    return [EvalResult(
        name=f"golden_range:{ticker_upper}:{agent_key}",
        passed=ok,
        details=(
            f"Score {score} outside expected [{lo}, {hi}]. "
            f"Rationale: {golden['rationale']}"
        ) if not ok else "",
        severity="warning",  # Warning not error — data changes daily
    )]


def eval_insights_not_empty(agent_output: Dict[str, Any]) -> List[EvalResult]:
    """
    Check that the agent produced at least one insight and one concern.
    
    WHY: An agent that returns zero insights is useless. The user
    sees an empty card. This catches "silent failure" in insight generation.
    """
    results = []
    insights = agent_output.get("insights", [])
    concerns = agent_output.get("concerns", [])

    results.append(EvalResult(
        name="has_insights",
        passed=len(insights) > 0,
        details=f"Agent produced 0 insights" if len(insights) == 0 else "",
    ))

    results.append(EvalResult(
        name="has_concerns",
        passed=len(concerns) > 0,
        details=f"Agent produced 0 concerns" if len(concerns) == 0 else "",
        severity="warning",  # Some stocks genuinely have no concerns
    ))

    return results


# ─── LLM-as-Judge Evals ───
#
# These require an LLM client. They are OPTIONAL — if no API key is set,
# the runner skips them and only runs code-based evals.


GROUNDING_PROMPT = """You are evaluating a financial analysis tool. You will be given:
1. The raw financial metrics the agent received
2. The insight or concern the agent produced

Determine whether the insight is GROUNDED in the provided metrics,
or whether it introduces claims not supported by the data.

[BEGIN DATA]
************
[Agent]: {agent_name}
[Metrics]: {metrics_json}
[Insight]: {insight_text}
[END DATA]

Consider:
- Does the insight reference specific numbers? If so, do they match the metrics?
- Does the insight make claims about trends or conditions not present in the data?
- Is the directional assessment correct? (e.g., if ROE is 8%, calling it "strong" is wrong)

Your response must be EXACTLY one of these two words on the first line:
grounded
hallucinated

Then on the next line, provide a brief explanation (1-2 sentences)."""


PHILOSOPHY_PROMPT = """You are evaluating whether an investment analysis agent stays true
to its assigned investment philosophy.

[BEGIN DATA]
************
[Agent]: {agent_name}
[Philosophy]: {philosophy}
[Key Principles]: {principles}
[Insight]: {insight_text}
[Concern]: {concern_text}
[END DATA]

Determine whether the analysis reflects the named investor's
known philosophy and principles.

Examples of IN-CHARACTER for Warren Buffett:
- Discussing moats, ROE, owner earnings, margin of safety
- Long-term language ("over decades", "durable advantage")

Examples of OUT-OF-CHARACTER for Warren Buffett:
- Discussing momentum, technical patterns, short-term trading
- Ignoring debt levels or management quality

Your response must be EXACTLY one of these two words on the first line:
in_character
out_of_character

Then on the next line, provide a brief explanation (1-2 sentences)."""


ACTIONABILITY_PROMPT = """You are evaluating whether an investment insight is actionable
for a retail investor with basic financial literacy.

[BEGIN DATA]
************
[Company]: {ticker}
[Insight]: {insight_text}
[Verdict]: {verdict}
[Score]: {score}
[END DATA]

Rate this insight on a scale of 1-5:

5 = Highly actionable: References specific numbers, explains WHY
    it matters, and connects to the investment decision.
    Example: "ROE of 28% has been consistent for 5 years,
    indicating a durable competitive moat worth paying a premium for."

3 = Somewhat actionable: States a fact but doesn't explain
    the implication or connect to the decision.
    Example: "The company has strong profit margins."

1 = Not actionable: Generic statement that could apply to any company.
    Example: "This is an interesting investment opportunity."

Your response must be a single number (1-5) on the first line,
followed by a brief explanation (1-2 sentences) on the next line."""


async def _call_llm_judge(prompt: str, llm_client) -> str:
    """Call the LLM judge and return the raw response."""
    try:
        response = await llm_client.analyze(
            prompt,
            persona="evaluator",
            verdict="hold",  # dummy — some clients require this
        )
        return response.strip()
    except Exception as e:
        return f"ERROR: {e}"


async def eval_financial_grounding(
    agent_output: Dict[str, Any],
    llm_client,
) -> List[EvalResult]:
    """
    LLM-as-Judge: Check if insights are grounded in actual metrics.
    
    This is Dimension 1 from our framework. The judge reads the metrics
    and the insight, then decides if the insight is making stuff up.
    """
    results = []
    metrics = agent_output.get("metrics", {})
    insights = agent_output.get("insights", [])
    agent_name = agent_output.get("agent", "Unknown")

    # Test a sample of insights (up to 3, to keep cost manageable)
    for i, insight in enumerate(insights[:3]):
        prompt = GROUNDING_PROMPT.format(
            agent_name=agent_name,
            metrics_json=json.dumps(metrics, indent=2, default=str),
            insight_text=insight,
        )
        response = await _call_llm_judge(prompt, llm_client)
        first_line = response.split("\n")[0].strip().lower()

        if first_line.startswith("error"):
            results.append(EvalResult(
                name=f"grounding:{agent_name}:insight_{i}",
                passed=False,
                details=f"LLM judge error: {response}",
                severity="warning",
            ))
        else:
            passed = "grounded" in first_line
            results.append(EvalResult(
                name=f"grounding:{agent_name}:insight_{i}",
                passed=passed,
                details=response if not passed else "",
            ))

    return results


async def eval_philosophy_adherence(
    agent_output: Dict[str, Any],
    llm_client,
) -> List[EvalResult]:
    """
    LLM-as-Judge: Check if the agent stays in character.
    
    This is Dimension 2. The judge reads the philosophy description
    and decides if the agent sounds like its namesake investor.
    """
    agent_name = agent_output.get("agent", "Unknown")
    insights = agent_output.get("insights", [])
    concerns = agent_output.get("concerns", [])

    # Find matching philosophy
    agent_key = None
    for key in PHILOSOPHY_KEYWORDS:
        if key in agent_name.lower():
            agent_key = key
            break

    if agent_key is None:
        return [EvalResult(
            name=f"philosophy:{agent_name}",
            passed=True,
            details="No philosophy defined for this agent",
            severity="warning",
        )]

    philosophy = PHILOSOPHY_KEYWORDS[agent_key]

    prompt = PHILOSOPHY_PROMPT.format(
        agent_name=agent_name,
        philosophy=philosophy["philosophy"],
        principles="\n".join(f"- {p}" for p in philosophy["principles"]),
        insight_text="\n".join(insights[:3]),
        concern_text="\n".join(concerns[:3]),
    )

    response = await _call_llm_judge(prompt, llm_client)
    first_line = response.split("\n")[0].strip().lower()

    if first_line.startswith("error"):
        return [EvalResult(
            name=f"philosophy:{agent_name}",
            passed=False,
            details=f"LLM judge error: {response}",
            severity="warning",
        )]

    passed = "in_character" in first_line
    return [EvalResult(
        name=f"philosophy:{agent_name}",
        passed=passed,
        details=response if not passed else "",
    )]


async def eval_actionable_quality(
    agent_output: Dict[str, Any],
    llm_client,
    ticker: str = "Unknown",
) -> List[EvalResult]:
    """
    LLM-as-Judge: Rate insight actionability on a 1-5 scale.
    
    This is Dimension 3. The judge reads the insight and asks:
    "Would a retail investor know what to do with this?"
    Score of 3+ = pass. Below 3 = the insight is too vague.
    """
    insights = agent_output.get("insights", [])
    verdict = agent_output.get("verdict", "hold")
    score = agent_output.get("score", 0)
    agent_name = agent_output.get("agent", "Unknown")

    if not insights:
        return [EvalResult(
            name=f"actionability:{agent_name}",
            passed=False,
            details="No insights to evaluate",
        )]

    # Combine top insights for evaluation
    combined_insights = "\n".join(insights[:5])

    prompt = ACTIONABILITY_PROMPT.format(
        ticker=ticker,
        insight_text=combined_insights,
        verdict=verdict,
        score=score,
    )

    response = await _call_llm_judge(prompt, llm_client)
    first_line = response.split("\n")[0].strip()

    if first_line.startswith("ERROR"):
        return [EvalResult(
            name=f"actionability:{agent_name}",
            passed=False,
            details=f"LLM judge error: {response}",
            severity="warning",
        )]

    # Parse the 1-5 score
    try:
        rating = int(first_line[0])
    except (ValueError, IndexError):
        return [EvalResult(
            name=f"actionability:{agent_name}",
            passed=False,
            details=f"Could not parse rating from: {first_line}",
            severity="warning",
        )]

    passed = rating >= 3  # 3+ is acceptable
    return [EvalResult(
        name=f"actionability:{agent_name}",
        passed=passed,
        details=f"Rating: {rating}/5. {response}" if not passed else f"Rating: {rating}/5",
    )]


# ─── Code-Based Philosophy Check (Keyword Scan) ───


def eval_philosophy_keywords(agent_output: Dict[str, Any]) -> List[EvalResult]:
    """
    Quick keyword scan: Do insights contain expected terms and avoid forbidden ones?
    
    This is a CHEAP supplement to the LLM philosophy check. It doesn't replace
    it — but it catches obvious problems instantly without an API call.
    
    Example: If Buffett's agent never mentions "moat" or "ROE" in any insight,
    something is probably wrong.
    """
    agent_name = agent_output.get("agent", "Unknown")
    insights_text = " ".join(agent_output.get("insights", [])).lower()
    concerns_text = " ".join(agent_output.get("concerns", [])).lower()
    all_text = insights_text + " " + concerns_text

    agent_key = None
    for key in PHILOSOPHY_KEYWORDS:
        if key in agent_name.lower():
            agent_key = key
            break

    if agent_key is None:
        return []

    keywords = PHILOSOPHY_KEYWORDS[agent_key]
    results = []

    # Check for at least ONE expected term
    found_expected = any(term.lower() in all_text for term in keywords["expected_terms"])
    results.append(EvalResult(
        name=f"keyword_expected:{agent_key}",
        passed=found_expected,
        details=(
            f"None of these terms found: {keywords['expected_terms'][:5]}"
            if not found_expected else ""
        ),
        severity="warning",
    ))

    # Check for forbidden terms
    found_forbidden = [
        term for term in keywords["forbidden_terms"]
        if term.lower() in all_text
    ]
    results.append(EvalResult(
        name=f"keyword_forbidden:{agent_key}",
        passed=len(found_forbidden) == 0,
        details=f"Forbidden terms found: {found_forbidden}" if found_forbidden else "",
        severity="warning",
    ))

    return results


# ─── Main Runner ───


async def run_agent_evals(
    agent_outputs: List[Dict[str, Any]],
    ticker: str,
    llm_client=None,
) -> LayerResults:
    """
    Run all Layer 2 agent evals.
    
    Args:
        agent_outputs: List of dicts from each agent's analyze() method
        ticker: The ticker symbol being analyzed
        llm_client: Optional LLM client for LLM-as-judge evals.
                     If None, only code-based evals run.
    
    Returns:
        LayerResults with all check results
    """
    layer = LayerResults(layer="Layer 2: Agent Quality")

    for output in agent_outputs:
        agent = output.get("agent", "Unknown")

        # Code-based checks (always run)
        layer.results.extend(eval_output_structure(output))
        layer.results.extend(eval_score_range(output))
        layer.results.extend(eval_verdict_consistency(output))
        layer.results.extend(eval_golden_score_range(output, ticker))
        layer.results.extend(eval_insights_not_empty(output))
        layer.results.extend(eval_philosophy_keywords(output))

        # LLM-as-judge checks (only if API key available)
        if llm_client:
            layer.results.extend(await eval_financial_grounding(output, llm_client))
            layer.results.extend(await eval_philosophy_adherence(output, llm_client))
            layer.results.extend(await eval_actionable_quality(output, llm_client, ticker=ticker))

    return layer
