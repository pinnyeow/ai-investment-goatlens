import pytest

from evals.consensus_evals import (
    eval_agreement_score_valid,
    eval_consensus_matches_majority,
    eval_directional_support_matches_agents,
)
from strategies import (
    ConsensusVerdict,
    StrategyResult,
    Verdict,
    calculate_consensus,
    calculate_consensus_with_llm,
)


def result(name: str, score: float, verdict: Verdict) -> StrategyResult:
    return StrategyResult(
        agent_name=name,
        verdict=verdict,
        confidence=1.0,
        key_insights=[],
        concerns=[],
        moat_assessment="",
        score=score,
    )


def test_mixed_pltr_votes_produce_low_agreement_hold_not_high_agreement_buy():
    results = [
        result("Buffett", 80.0, Verdict.STRONG_BUY),
        result("Lynch", 18.75, Verdict.HOLD),
        result("Graham", -30.0, Verdict.SELL),
        result("Munger", 75.0, Verdict.STRONG_BUY),
        result("Dalio", 16.61, Verdict.HOLD),
    ]

    consensus = calculate_consensus(results)

    assert consensus.consensus_verdict is ConsensusVerdict.HOLD
    assert consensus.agreement_score == pytest.approx(0.4)
    assert any(
        point.startswith("Mixed investor verdicts")
        for point in consensus.divergence_points
    )


def test_bullish_majority_uses_median_verdict_and_directional_agreement():
    results = [
        result("Buffett", 80.0, Verdict.STRONG_BUY),
        result("Lynch", 77.5, Verdict.STRONG_BUY),
        result("Graham", -30.0, Verdict.SELL),
        result("Munger", 75.0, Verdict.STRONG_BUY),
        result("Dalio", 32.06, Verdict.BUY),
    ]

    consensus = calculate_consensus(results)

    assert consensus.consensus_verdict is ConsensusVerdict.STRONG_BUY
    assert consensus.agreement_score == pytest.approx(0.8)


def test_bearish_majority_is_not_biased_toward_buy():
    results = [
        result("A", -80.0, Verdict.STRONG_SELL),
        result("B", -50.0, Verdict.SELL),
        result("C", -40.0, Verdict.SELL),
        result("D", -30.0, Verdict.SELL),
        result("E", -10.0, Verdict.HOLD),
    ]

    consensus = calculate_consensus(results)

    assert consensus.consensus_verdict is ConsensusVerdict.SELL
    assert consensus.agreement_score == pytest.approx(0.8)


def test_amzn_votes_produce_buy_with_sixty_percent_directional_support():
    results = [
        result("Buffett", 64.73, Verdict.STRONG_BUY),
        result("Lynch", 18.75, Verdict.HOLD),
        result("Graham", -30.0, Verdict.SELL),
        result("Munger", 45.0, Verdict.BUY),
        result("Dalio", 54.27, Verdict.BUY),
    ]

    consensus = calculate_consensus(results)

    assert consensus.consensus_verdict is ConsensusVerdict.BUY
    assert consensus.agreement_score == pytest.approx(0.6)


def test_same_direction_intensity_disagreement_has_full_directional_support():
    results = [
        result("A", 80.0, Verdict.STRONG_BUY),
        result("B", 70.0, Verdict.STRONG_BUY),
        result("C", 65.0, Verdict.STRONG_BUY),
        result("D", 40.0, Verdict.BUY),
        result("E", 30.0, Verdict.BUY),
    ]

    consensus = calculate_consensus(results)

    assert consensus.consensus_verdict is ConsensusVerdict.STRONG_BUY
    assert consensus.agreement_score == pytest.approx(1.0)


def test_even_cross_direction_split_is_mixed_without_directional_support():
    results = [
        result("A", -30.0, Verdict.SELL),
        result("B", 30.0, Verdict.BUY),
    ]

    consensus = calculate_consensus(results)

    assert consensus.consensus_verdict is ConsensusVerdict.MIXED
    assert consensus.agreement_score is None


def test_hold_majority_produces_hold_support():
    results = [
        result("A", 65.0, Verdict.STRONG_BUY),
        result("B", 10.0, Verdict.HOLD),
        result("C", 0.0, Verdict.HOLD),
        result("D", -10.0, Verdict.HOLD),
        result("E", -30.0, Verdict.SELL),
    ]

    consensus = calculate_consensus(results)

    assert consensus.consensus_verdict is ConsensusVerdict.HOLD
    assert consensus.agreement_score == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_llm_consensus_preserves_structural_no_majority_warning():
    class FakeLLM:
        async def analyze(self, *args, **kwargs):
            return "DEBATE1: Valuation differs.\nDEBATE2: Growth differs."

    results = [
        result("Buffett", 80.0, Verdict.STRONG_BUY),
        result("Lynch", 18.75, Verdict.HOLD),
        result("Graham", -30.0, Verdict.SELL),
        result("Munger", 75.0, Verdict.STRONG_BUY),
        result("Dalio", 16.61, Verdict.HOLD),
    ]

    consensus = await calculate_consensus_with_llm(results, FakeLLM(), "PLTR")

    assert consensus.divergence_points[0].startswith("Mixed investor verdicts")


def test_consensus_evals_accept_valid_mixed_contract():
    consensus = {"verdict": "mixed", "agreement_score": None}
    agents = [{"verdict": "sell"}, {"verdict": "buy"}]

    checks = [
        *eval_agreement_score_valid(consensus),
        *eval_directional_support_matches_agents(consensus, agents),
        *eval_consensus_matches_majority(consensus, agents),
    ]

    assert all(check.passed for check in checks)


def test_consensus_evals_reject_mixed_with_numeric_directional_support():
    checks = eval_agreement_score_valid(
        {"verdict": "mixed", "agreement_score": 0.5}
    )

    assert checks[0].passed is False


def test_consensus_evals_reject_fabricated_directional_support():
    consensus = {"verdict": "buy", "agreement_score": 0.01}
    agents = [
        {"verdict": "buy"},
        {"verdict": "strong_buy"},
        {"verdict": "buy"},
        {"verdict": "sell"},
        {"verdict": "strong_sell"},
    ]

    checks = eval_directional_support_matches_agents(consensus, agents)

    assert checks[0].passed is False
    assert "expected 0.6" in checks[0].details
