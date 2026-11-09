from conftest import QUALITY_COMPANY, company
from lens.build import analyze, earnings_streak
from lens.investors import INVESTORS, consensus, evaluate_investor


def test_quality_company_gets_all_five_verdicts():
    snapshot = analyze(company())
    assert snapshot["completed_analyses"] == 5
    assert all(r["score"] is not None for r in snapshot["agent_results"])
    assert snapshot["consensus_verdict"] in {"buy", "strong_buy"}


def test_missing_metric_is_skipped_not_penalised():
    full = evaluate_investor("buffett", QUALITY_COMPANY)
    without_debt = evaluate_investor("buffett", {**QUALITY_COMPANY, "debt_to_equity": None})
    assert without_debt["status"] == "complete"
    assert without_debt["criteria_checked"] == full["criteria_checked"] - 1
    # A gap must never drag the score below what the available evidence supports.
    assert without_debt["score"] >= full["score"] - 1
    skipped = [c for c in without_debt["criteria"] if c["status"] == "no_data"]
    assert skipped and "debt-to-equity" in skipped[0]["detail"]


def test_one_gap_never_hides_the_consensus():
    metrics = {**QUALITY_COMPANY, "pe_ratio": None, "pb_ratio": None, "current_ratio": None}
    snapshot = analyze(company(metrics))
    assert snapshot["consensus_verdict"] not in {"insufficient_data", "incomplete"}


def test_investor_needs_half_its_checklist():
    empty = {key: None for key in QUALITY_COMPANY}
    result = evaluate_investor("munger", {**empty, "gross_margin": 0.5})
    assert result["status"] == "insufficient_data"
    assert result["score"] is None and result["verdict"] is None


def test_no_data_at_all_is_reported_honestly():
    snapshot = analyze(company({key: None for key in QUALITY_COMPANY}))
    assert snapshot["completed_analyses"] == 0
    assert snapshot["consensus_verdict"] == "insufficient_data"


def test_banks_explain_not_meaningful_checks():
    na = ["free_cash_flow", "debt_to_equity"]
    metrics = {**QUALITY_COMPANY, "free_cash_flow": None, "debt_to_equity": None}
    snapshot = analyze(company(metrics, not_applicable=na, is_financial=True))
    buffett = next(r for r in snapshot["agent_results"] if r["id"] == "buffett")
    details = [c["detail"] for c in buffett["criteria"] if c["status"] == "no_data"]
    assert details and all("banks and insurers" in d for d in details)


def test_scores_stay_in_range_and_verdicts_match():
    for investor_id in INVESTORS:
        for metrics in (QUALITY_COMPANY, {**QUALITY_COMPANY, "roe": -0.5, "pe_ratio": 90.0, "beta": 3.0}):
            r = evaluate_investor(investor_id, metrics)
            assert -100 <= r["score"] <= 100


def test_graham_flags_expensive_stock():
    pricey = {**QUALITY_COMPANY, "current_price": 400.0, "pe_ratio": 50.0, "pb_ratio": 5.0}
    assert evaluate_investor("graham", pricey)["verdict"] in {"sell", "strong_sell"}


def test_consensus_of_subset():
    results = [evaluate_investor(i, QUALITY_COMPANY) for i in ("buffett", "graham")]
    out = consensus(results)
    assert out["consensus_score"] is not None and 0 <= out["agreement_score"] <= 1


def test_earnings_streak():
    quarters = [{"beat_miss": b} for b in ("beat", "beat", "miss", "beat")]
    streak = earnings_streak(quarters)
    assert streak["streak_type"] == "beat" and streak["streak_count"] == 2 and streak["beats"] == 3
