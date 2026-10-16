import json
import asyncio
import importlib
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main


def response_agent(name, *, complete=True, unavailable_metrics=None):
    return {
        "agent": name,
        "style": "Test style",
        "verdict": "buy" if complete else None,
        "score": 30.0 if complete else None,
        "insights": ["Shared view"] if complete else [],
        "concerns": [] if complete else ["Required evidence is unavailable."],
        "status": "complete" if complete else "incomplete",
        "unavailable_metrics": unavailable_metrics or [],
    }


def analysis_payload(**consensus_overrides):
    payload = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "data_status": {
            "fundamentals": {"available": True, "source": "sec", "as_of": "2026-09-30"},
            "market_data": {"available": True, "source": "yahoo", "as_of": "2026-10-15"},
            "normalized_ratios": {"available": True, "source": "yahoo", "as_of": "2026-10-15"},
            "price_dependent_analysis": {"status": "complete", "reason": None},
        },
        "fundamentals": {},
        "fundamentals_history": {},
        "metric_evidence": {},
        "consensus_status": "complete",
        "consensus_verdict": "buy",
        "agreement_score": 1.0,
        "completed_analyses": 2,
        "requested_analyses": 2,
        "consensus_points": ["Shared view"],
        "divergence_points": [],
        "moat_trend": "stable",
        "comparison_table": {"agents": []},
    }
    payload.update(consensus_overrides)
    if "agent_results" not in consensus_overrides:
        completed = payload["completed_analyses"]
        requested = payload["requested_analyses"]
        payload["agent_results"] = [
            response_agent(
                f"Agent {index + 1}",
                complete=index < completed,
                unavailable_metrics=(
                    [] if index < completed else [f"metric_{index + 1}"]
                ),
            )
            for index in range(requested)
        ]
    return payload


def test_api_rejects_partial_verdict_in_final_consensus_field():
    with pytest.raises(ValidationError, match="Partial analysis cannot publish a final consensus verdict"):
        main.AnalysisResponse.model_validate(
            analysis_payload(
                consensus_status="partial",
                consensus_verdict="buy",
                agreement_score=1.0,
                completed_analyses=2,
                requested_analyses=3,
            )
        )


def test_consensus_evaluators_check_the_subordinate_partial_indication():
    from evals.consensus_evals import (
        eval_agreement_score_valid,
        eval_consensus_matches_majority,
        eval_directional_support_matches_agents,
    )

    consensus = {
        "status": "partial",
        "verdict": "incomplete",
        "agreement_score": None,
        "partial_indication": {
            "verdict": "buy",
            "directional_support": 1.0,
            "supporting_analyses": 2,
        },
    }
    agents = [
        {"status": "complete", "verdict": "strong_buy"},
        {"status": "complete", "verdict": "buy"},
        {"status": "incomplete", "verdict": None},
    ]

    checks = [
        *eval_agreement_score_valid(consensus),
        *eval_directional_support_matches_agents(consensus, agents),
        *eval_consensus_matches_majority(consensus, agents),
    ]
    assert all(check.passed for check in checks)


def test_remaining_evaluators_use_partial_indication_and_completed_agents():
    from evals.consensus_evals import (
        eval_consensus_coherence,
        eval_consensus_points_exist,
        eval_no_extreme_contradiction,
    )

    consensus = {
        "status": "partial",
        "verdict": "incomplete",
        "agreement_score": None,
        "consensus_points": [],
        "divergence_points": [],
        "partial_indication": {
            "verdict": "buy",
            "directional_support": 0.0,
            "supporting_analyses": 0,
            "consensus_points": ["Completed-vote point"],
            "divergence_points": [],
        },
    }
    agents = [
        {"agent": "Completed A", "status": "complete", "verdict": "sell"},
        {"agent": "Completed B", "status": "complete", "verdict": "strong_sell"},
        {"agent": "Missing", "status": "incomplete", "verdict": None},
    ]

    contradiction = eval_no_extreme_contradiction(consensus, agents)[0]
    assert not contradiction.passed
    assert eval_consensus_points_exist(consensus, agents)[0].passed

    class CapturingLlm:
        prompt = None

        async def analyze(self, prompt, **kwargs):
            self.prompt = prompt
            return "coherent\nUses completed votes only."

    llm = CapturingLlm()
    result = asyncio.run(
        eval_consensus_coherence(consensus, agents, "AAPL", llm)
    )[0]
    assert result.passed
    assert isinstance(llm.prompt, str)
    assert "[Consensus Verdict]: buy" in llm.prompt
    assert "Completed-vote point" in llm.prompt
    assert "Completed A" in llm.prompt
    assert "Completed B" in llm.prompt
    assert "Missing" not in llm.prompt


def test_all_consensus_evaluators_abstain_when_fewer_than_two_complete():
    from evals.consensus_evals import run_consensus_evals

    class RejectingLlm:
        async def analyze(self, *args, **kwargs):
            raise AssertionError("LLM must not judge an unavailable consensus")

    layer = asyncio.run(
        run_consensus_evals(
            consensus={
                "status": "incomplete",
                "verdict": "incomplete",
                "agreement_score": None,
                "partial_indication": None,
                "consensus_points": [],
                "divergence_points": ["Fewer than two analyses completed."],
                "completed_analyses": 1,
                "requested_analyses": 1,
            },
            agent_outputs=[response_agent("Only completed agent")],
            ticker="AAPL",
            llm_client=RejectingLlm(),
        )
    )

    assert len(layer.results) == 7
    assert all(result.passed for result in layer.results)
    assert layer.results[0].name == "consensus_structural_integrity"
    assert all("abstain" in result.name for result in layer.results[1:])


def test_all_consensus_evaluators_reject_a_single_vote_published_as_consensus():
    from evals.consensus_evals import run_consensus_evals

    class RejectingLlm:
        async def analyze(self, *args, **kwargs):
            raise AssertionError("LLM must not judge an invalid single-vote consensus")

    layer = asyncio.run(
        run_consensus_evals(
            consensus={
                "status": "complete",
                "verdict": "buy",
                "agreement_score": 1.0,
                "partial_indication": None,
                "consensus_points": ["One vote"],
                "divergence_points": [],
                "completed_analyses": 1,
                "requested_analyses": 1,
            },
            agent_outputs=[response_agent("Only completed agent")],
            ticker="AAPL",
            llm_client=RejectingLlm(),
        )
    )

    assert len(layer.results) == 7
    assert all(not result.passed for result in layer.results)
    assert all("integrity" in result.name for result in layer.results)


def test_consensus_evaluator_rejects_final_consensus_when_an_agent_is_incomplete():
    from evals.consensus_evals import run_consensus_evals

    agents = [
        response_agent("Completed A"),
        response_agent("Completed B"),
        response_agent("Missing", complete=False, unavailable_metrics=["price"]),
    ]
    layer = asyncio.run(
        run_consensus_evals(
            consensus={
                "status": "complete",
                "verdict": "buy",
                "agreement_score": 1.0,
                "partial_indication": None,
                "consensus_points": ["Shared view"],
                "divergence_points": [],
                "completed_analyses": 2,
                "requested_analyses": 3,
            },
            agent_outputs=agents,
            ticker="AAPL",
        )
    )

    integrity = [
        result for result in layer.results
        if result.name == "consensus_structural_integrity"
    ]
    assert len(integrity) == 1
    assert not integrity[0].passed
    assert "final consensus" in integrity[0].details


def test_consensus_evaluator_rejects_malformed_partial_metadata():
    from evals.consensus_evals import run_consensus_evals

    agents = [
        response_agent("Completed A"),
        response_agent("Completed B"),
        response_agent("Missing", complete=False, unavailable_metrics=["price"]),
    ]
    layer = asyncio.run(
        run_consensus_evals(
            consensus={
                "status": "partial",
                "verdict": "incomplete",
                "agreement_score": None,
                "partial_indication": {
                    "verdict": "buy",
                    "directional_support": 1.0,
                    "supporting_analyses": 2,
                    "completed_analyses": 2,
                    "requested_analyses": 3,
                    "excluded_agents": [
                        {"agent": "Invented", "unavailable_metrics": ["price"]}
                    ],
                    "consensus_points": ["Shared view"],
                    "divergence_points": [],
                },
                "consensus_points": [],
                "divergence_points": ["Final consensus unavailable."],
                "completed_analyses": 2,
                "requested_analyses": 3,
            },
            agent_outputs=agents,
            ticker="AAPL",
        )
    )

    integrity = [
        result for result in layer.results
        if result.name == "consensus_structural_integrity"
    ]
    assert len(integrity) == 1
    assert not integrity[0].passed
    assert "excluded agents" in integrity[0].details


@pytest.mark.parametrize(
    ("verdict", "directional_support", "supporting_analyses"),
    [
        ("buy", 1.0, 1),
        ("mixed", None, 1),
    ],
)
def test_consensus_evaluator_rejects_incorrect_partial_support_numerator(
    verdict, directional_support, supporting_analyses
):
    from evals.consensus_evals import run_consensus_evals

    agents = [
        response_agent("Completed A"),
        response_agent("Completed B"),
        response_agent("Missing", complete=False, unavailable_metrics=["price"]),
    ]
    if verdict == "mixed":
        agents[1]["verdict"] = "sell"
        agents[1]["score"] = -30.0
    layer = asyncio.run(
        run_consensus_evals(
            consensus={
                "status": "partial",
                "verdict": "incomplete",
                "agreement_score": None,
                "partial_indication": {
                    "verdict": verdict,
                    "directional_support": directional_support,
                    "supporting_analyses": supporting_analyses,
                    "completed_analyses": 2,
                    "requested_analyses": 3,
                    "excluded_agents": [
                        {"agent": "Missing", "unavailable_metrics": ["price"]}
                    ],
                    "consensus_points": ["Completed-vote evidence"],
                    "divergence_points": [],
                },
                "consensus_points": [],
                "divergence_points": ["Final consensus unavailable."],
                "completed_analyses": 2,
                "requested_analyses": 3,
            },
            agent_outputs=agents,
            ticker="AAPL",
        )
    )

    assert any(
        not result.passed and "supporting analyses" in result.details.lower()
        for result in layer.results
    )


def test_eval_runner_preserves_flattened_partial_consensus_contract(monkeypatch):
    run_evals = importlib.import_module("evals.run_evals")
    partial_indication = {
        "verdict": "buy",
        "directional_support": 1.0,
        "supporting_analyses": 2,
        "completed_analyses": 2,
        "requested_analyses": 3,
        "excluded_agents": [
            {"agent": "Missing", "unavailable_metrics": ["price"]}
        ],
        "consensus_points": ["Shared view"],
        "divergence_points": [],
    }
    analysis = {
        "consensus_status": "partial",
        "consensus_verdict": "incomplete",
        "agreement_score": None,
        "partial_indication": partial_indication,
        "consensus_points": [],
        "divergence_points": ["Final consensus unavailable."],
        "completed_analyses": 2,
        "requested_analyses": 3,
        "agent_results": [],
    }
    captured = {}

    async def fake_fetch_analysis(ticker, base_url):
        return analysis

    async def fake_run_consensus_evals(**kwargs):
        captured.update(kwargs)
        return run_evals.LayerResults(layer="captured")

    monkeypatch.setattr(run_evals, "fetch_analysis", fake_fetch_analysis)
    monkeypatch.setattr(run_evals, "run_consensus_evals", fake_run_consensus_evals)
    monkeypatch.setattr(run_evals, "print_layer_results", lambda layer: None)

    asyncio.run(
        run_evals.run_evals_for_ticker(
            ticker="AAPL", layers=[3], base_url="http://test", llm_client=None
        )
    )

    assert captured["consensus"]["status"] == "partial"
    assert captured["consensus"]["partial_indication"] == partial_indication
    assert captured["consensus"]["completed_analyses"] == 2
    assert captured["consensus"]["requested_analyses"] == 3


def test_api_exposes_completed_votes_only_as_a_subordinate_partial_indication():
    response = main.AnalysisResponse.model_validate(
        analysis_payload(
            consensus_status="partial",
            consensus_verdict="incomplete",
            agreement_score=None,
            completed_analyses=2,
            requested_analyses=3,
            consensus_points=[],
            partial_indication={
                "verdict": "buy",
                "directional_support": 1.0,
                "supporting_analyses": 2,
                "completed_analyses": 2,
                "requested_analyses": 3,
                "excluded_agents": [
                    {
                        "agent": "Unavailable Agent",
                        "unavailable_metrics": ["current_price"],
                    }
                ],
            },
            agent_results=[
                response_agent("Agent 1"),
                response_agent("Agent 2"),
                response_agent(
                    "Unavailable Agent",
                    complete=False,
                    unavailable_metrics=["current_price"],
                ),
            ],
        )
    )

    assert response.consensus_verdict == "incomplete"
    assert response.agreement_score is None
    assert response.partial_indication is not None
    assert response.partial_indication.verdict == "buy"
    assert response.partial_indication.supporting_analyses == 2
    assert response.partial_indication.excluded_agents[0].agent == "Unavailable Agent"


@pytest.mark.parametrize(
    "overrides",
    [
        {"requested_analyses": 3},
        {"completed_analyses": 1},
        {
            "agent_results": [
                response_agent("Duplicate"),
                response_agent("Duplicate"),
            ]
        },
        {
            "consensus_status": "partial",
            "consensus_verdict": "incomplete",
            "agreement_score": None,
            "completed_analyses": 2,
            "requested_analyses": 4,
            "consensus_points": [],
            "agent_results": [
                response_agent("Agent 1"),
                response_agent("Agent 2"),
                response_agent("Missing 1", complete=False,
                               unavailable_metrics=["price"]),
                response_agent("Missing 2", complete=False,
                               unavailable_metrics=["beta"]),
            ],
            "partial_indication": {
                "verdict": "buy",
                "directional_support": 1.0,
                "supporting_analyses": 2,
                "completed_analyses": 2,
                "requested_analyses": 4,
                "excluded_agents": [
                    {"agent": "Missing 1", "unavailable_metrics": ["price"]},
                    {"agent": "Missing 1", "unavailable_metrics": ["price"]},
                ],
            },
        },
        {
            "consensus_status": "partial",
            "consensus_verdict": "incomplete",
            "agreement_score": None,
            "completed_analyses": 2,
            "requested_analyses": 3,
            "consensus_points": [],
            "partial_indication": {
                "verdict": "buy",
                "directional_support": 1.0,
                "supporting_analyses": 2,
                "completed_analyses": 2,
                "requested_analyses": 3,
                "excluded_agents": [
                    {"agent": "Invented", "unavailable_metrics": ["metric_3"]}
                ],
            },
        },
        {
            "consensus_status": "partial",
            "consensus_verdict": "incomplete",
            "agreement_score": None,
            "completed_analyses": 2,
            "requested_analyses": 3,
            "consensus_points": [],
            "partial_indication": {
                "verdict": "buy",
                "directional_support": 1.0,
                "supporting_analyses": 2,
                "completed_analyses": 2,
                "requested_analyses": 3,
                "excluded_agents": [
                    {"agent": "Agent 3", "unavailable_metrics": ["substituted"]}
                ],
            },
        },
    ],
)
def test_analysis_response_reconciles_counts_and_exact_agent_projection(overrides):
    with pytest.raises(ValidationError, match="agent|Agent|analyses|Consensus"):
        main.AnalysisResponse.model_validate(analysis_payload(**overrides))


@pytest.mark.parametrize("agreement_score", [None, 0.5])
def test_complete_directional_consensus_requires_exact_agent_support(agreement_score):
    with pytest.raises(ValidationError, match="support|Consensus"):
        main.AnalysisResponse.model_validate(
            analysis_payload(agreement_score=agreement_score)
        )


@pytest.mark.parametrize(
    ("agent_verdicts", "published_verdict", "agreement_score"),
    [
        (["buy", "strong_buy"], "mixed", None),
        (["buy", "strong_buy"], "strong_buy", 1.0),
        (["sell", "buy"], "buy", 0.5),
    ],
)
def test_analysis_response_rejects_wrong_complete_ordinal_verdict(
    agent_verdicts, published_verdict, agreement_score
):
    agents = [response_agent(f"Agent {index + 1}") for index in range(2)]
    for agent, verdict in zip(agents, agent_verdicts):
        agent["verdict"] = verdict
        agent["score"] = {
            "sell": -30.0,
            "buy": 30.0,
            "strong_buy": 70.0,
        }[verdict]

    with pytest.raises(ValidationError, match="ordinal consensus verdict"):
        main.AnalysisResponse.model_validate(
            analysis_payload(
                consensus_verdict=published_verdict,
                agreement_score=agreement_score,
                agent_results=agents,
            )
        )


@pytest.mark.parametrize(
    ("agent_verdicts", "published_verdict", "directional_support", "supporting"),
    [
        (["buy", "strong_buy"], "mixed", None, 0),
        (["buy", "strong_buy"], "strong_buy", 1.0, 2),
        (["sell", "buy"], "buy", 0.5, 1),
    ],
)
def test_analysis_response_rejects_wrong_partial_ordinal_verdict(
    agent_verdicts, published_verdict, directional_support, supporting
):
    agents = [response_agent(f"Agent {index + 1}") for index in range(2)]
    for agent, verdict in zip(agents, agent_verdicts):
        agent["verdict"] = verdict
        agent["score"] = {
            "sell": -30.0,
            "buy": 30.0,
            "strong_buy": 70.0,
        }[verdict]
    agents.append(
        response_agent("Missing", complete=False, unavailable_metrics=["price"])
    )

    with pytest.raises(ValidationError, match="ordinal consensus verdict"):
        main.AnalysisResponse.model_validate(
            analysis_payload(
                consensus_status="partial",
                consensus_verdict="incomplete",
                agreement_score=None,
                completed_analyses=2,
                requested_analyses=3,
                consensus_points=[],
                agent_results=agents,
                partial_indication={
                    "verdict": published_verdict,
                    "directional_support": directional_support,
                    "supporting_analyses": supporting,
                    "completed_analyses": 2,
                    "requested_analyses": 3,
                    "excluded_agents": [
                        {"agent": "Missing", "unavailable_metrics": ["price"]}
                    ],
                },
            )
        )


def test_unavailable_consensus_rejects_headline_points_and_agreement():
    with pytest.raises(ValidationError, match="Consensus"):
        main.AnalysisResponse.model_validate(
            analysis_payload(
                consensus_status="incomplete",
                consensus_verdict="incomplete",
                agreement_score=1.0,
                completed_analyses=1,
                requested_analyses=2,
                consensus_points=["Not a headline consensus"],
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "consensus_status": "partial",
            "consensus_verdict": "incomplete",
            "agreement_score": 0.5,
            "completed_analyses": 2,
            "requested_analyses": 3,
        },
        {
            "consensus_status": "partial",
            "consensus_verdict": "incomplete",
            "agreement_score": None,
            "completed_analyses": 2,
            "requested_analyses": 3,
        },
        {
            "partial_indication": {
                "verdict": "buy",
                "directional_support": 1.0,
                "supporting_analyses": 2,
                "completed_analyses": 2,
                "requested_analyses": 3,
                "excluded_agents": [
                    {"agent": "Excluded", "unavailable_metrics": ["price"]}
                ],
            }
        },
        {
            "consensus_status": "incomplete",
            "consensus_verdict": "incomplete",
            "agreement_score": None,
            "completed_analyses": 0,
            "requested_analyses": 3,
            "partial_indication": {
                "verdict": "buy",
                "directional_support": 1.0,
                "supporting_analyses": 1,
                "completed_analyses": 1,
                "requested_analyses": 3,
                "excluded_agents": [],
            },
        },
        {
            "consensus_status": "partial",
            "consensus_verdict": "incomplete",
            "agreement_score": None,
            "completed_analyses": 2,
            "requested_analyses": 3,
            "consensus_points": ["Completed votes agree"],
            "partial_indication": {
                "verdict": "buy",
                "directional_support": 1.0,
                "supporting_analyses": 2,
                "completed_analyses": 2,
                "requested_analyses": 3,
                "excluded_agents": [
                    {"agent": "Excluded", "unavailable_metrics": ["price"]}
                ],
            },
        },
    ],
)
def test_api_rejects_cross_field_consensus_contradictions(overrides):
    with pytest.raises(ValidationError):
        main.AnalysisResponse.model_validate(analysis_payload(**overrides))


AGENT_NAMES = ["Buffett", "Lynch", "Graham", "Munger", "Dalio"]


def agent_result(index, complete):
    if complete:
        return {
            "agent": AGENT_NAMES[index],
            "style": "Test style",
            "ticker": "AAPL",
            "score": 30.0,
            "verdict": "buy",
            "insights": ["Shared view"],
            "concerns": [],
            "status": "complete",
            "unavailable_metrics": [],
        }
    return {
        "agent": AGENT_NAMES[index],
        "style": "Test style",
        "ticker": "AAPL",
        "score": None,
        "verdict": None,
        "insights": [],
        "concerns": ["Required evidence is unavailable."],
        "status": "incomplete",
        "unavailable_metrics": [f"metric_{index}"],
    }


class ConsensusWorkflow:
    def __init__(self, results):
        self.results = results

    async def ainvoke(self, state):
        state.update(
            {
                "agent_results": self.results,
                "normalized_data": {"company_name": "Apple Inc.", "sector": "Technology"},
                "data_status": analysis_payload()["data_status"],
                "sec_fundamentals": {},
                "sec_fundamentals_history": {},
                "metric_evidence": {},
                "earnings_data": [],
                "earnings_streak": {},
                "temporal_results": {},
            }
        )
        return await main.synthesize_node(state)


@pytest.mark.parametrize("completed", range(6))
def test_public_api_keeps_partial_votes_out_of_the_final_consensus(
    monkeypatch, completed
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    results = [agent_result(index, index < completed) for index in range(5)]

    with TestClient(main.app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(main, "goat_workflow", ConsensusWorkflow(results))
        response = client.post(
            "/api/analyze",
            json={
                "ticker": "AAPL",
                "agents": ["buffett", "lynch", "graham", "munger", "dalio"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["completed_analyses"] == completed
    assert body["requested_analyses"] == 5
    if completed == 5:
        assert body["consensus_status"] == "complete"
        assert body["consensus_verdict"] == "buy"
        assert body["agreement_score"] == 1.0
        assert body["partial_indication"] is None
    elif completed < 2:
        assert body["consensus_status"] == "incomplete"
        assert body["consensus_verdict"] == "incomplete"
        assert body["agreement_score"] is None
        assert body["partial_indication"] is None
        assert body["consensus_points"] == []
        assert len(body["agent_results"]) == len(results)
        assert [item["agent"] for item in body["agent_results"]] == AGENT_NAMES[:len(results)]
        assert [item["status"] for item in body["agent_results"]] == [
            item["status"] for item in results
        ]
    else:
        assert body["consensus_status"] == "partial"
        assert body["consensus_verdict"] == "incomplete"
        assert body["agreement_score"] is None
        indication = body["partial_indication"]
        assert indication["verdict"] == "buy"
        assert indication["directional_support"] == 1.0
        assert indication["supporting_analyses"] == completed
        assert indication["completed_analyses"] == completed
        assert indication["requested_analyses"] == 5
        assert [item["agent"] for item in indication["excluded_agents"]] == AGENT_NAMES[completed:]


@pytest.mark.parametrize(
    ("selected_agents", "completed", "expected_status"),
    [
        (["buffett"], 1, "incomplete"),
        (["buffett", "lynch"], 1, "incomplete"),
        (["buffett", "lynch"], 2, "complete"),
    ],
)
def test_public_api_applies_consensus_integrity_to_selected_subsets(
    monkeypatch, selected_agents, completed, expected_status
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    results = [
        agent_result(index, index < completed)
        for index in range(len(selected_agents))
    ]

    with TestClient(main.app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(main, "goat_workflow", ConsensusWorkflow(results))
        response = client.post(
            "/api/analyze",
            json={"ticker": "AAPL", "agents": selected_agents},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["requested_analyses"] == len(selected_agents)
    assert body["completed_analyses"] == completed
    assert body["consensus_status"] == expected_status
    if expected_status == "incomplete":
        assert body["consensus_verdict"] == "incomplete"
        assert body["agreement_score"] is None
        assert body["partial_indication"] is None
        assert body["consensus_points"] == []
        assert len(body["agent_results"]) == len(results)
        assert [item["agent"] for item in body["agent_results"]] == AGENT_NAMES[:len(results)]
        assert [item["status"] for item in body["agent_results"]] == [
            item["status"] for item in results
        ]
    else:
        assert body["consensus_verdict"] == "buy"
        assert body["partial_indication"] is None



def test_api_rejects_directional_support_for_mixed_partial_indication():
    with pytest.raises(ValidationError, match="Mixed partial indication"):
        main.AnalysisResponse.model_validate(
            analysis_payload(
                consensus_status="partial",
                consensus_verdict="incomplete",
                agreement_score=None,
                completed_analyses=2,
                requested_analyses=3,
                consensus_points=[],
                partial_indication={
                    "verdict": "mixed",
                    "directional_support": 0.5,
                    "supporting_analyses": 0,
                    "completed_analyses": 2,
                    "requested_analyses": 3,
                    "excluded_agents": [
                        {"agent": "Excluded", "unavailable_metrics": ["price"]}
                    ],
                    "divergence_points": [
                        "Mixed investor verdicts span opposing directions."
                    ],
                },
            )
        )


def test_browser_renders_partial_vote_as_subordinate_indication():
    html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()
    marker = "function getConsensusUiState(data)"
    assert marker in html
    helper_source = marker + html.split(marker, 1)[1].split(
        "\n        function displayResults", 1
    )[0]
    verdict_source = "function formatVerdict(verdict)" + html.split(
        "function formatVerdict(verdict)", 1
    )[1].split("\n        function formatMoatTrend", 1)[0]
    script = f"""
const assert = require('assert');
{verdict_source}
{helper_source}
const ui = getConsensusUiState({{
  consensus_status: 'partial',
  consensus_verdict: 'incomplete',
  completed_analyses: 3,
  requested_analyses: 5,
  partial_indication: {{
    verdict: 'strong_buy',
    directional_support: 2 / 3,
    supporting_analyses: 2,
    completed_analyses: 3,
    requested_analyses: 5,
    excluded_agents: [
      {{agent: 'Graham', unavailable_metrics: ['current_price']}},
      {{agent: 'Dalio', unavailable_metrics: ['beta', 'volatility']}}
    ]
  }}
}});
assert.strictEqual(ui.headline, 'Incomplete');
assert.strictEqual(ui.status, 'Incomplete — 3 of 5 requested analyses complete.');
assert.strictEqual(
  ui.partialIndication,
  'Partial indication: Strong Buy among 3 completed investors. Directional support: 2 of 3 completed. Excluded: Graham (current price); Dalio (beta, volatility).'
);
assert.strictEqual(ui.showPartialIndication, true);
const mixedUi = getConsensusUiState({{
  consensus_status: 'partial',
  consensus_verdict: 'incomplete',
  completed_analyses: 2,
  requested_analyses: 3,
  partial_indication: {{
    verdict: 'mixed',
    directional_support: null,
    supporting_analyses: 0,
    completed_analyses: 2,
    requested_analyses: 3,
    excluded_agents: [{{agent: 'Dalio', unavailable_metrics: ['beta']}}]
  }}
}});
assert.match(mixedUi.partialIndication, /^Partial indication: Mixed among/);
assert.match(mixedUi.partialIndication, /No directional support for mixed completed votes/);
"""
    result = subprocess.run(
        ["node", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, json.dumps(
        {"stdout": result.stdout, "stderr": result.stderr}
    )

    display_results = html.split("function displayResults(data)", 1)[1].split(
        "\n        function renderEarnings", 1
    )[0]
    assert 'id="partialIndication"' in html
    assert "getConsensusUiState(data)" in display_results
    assert "partialIndication.textContent" in display_results


def test_browser_explains_single_completed_opinion_without_calling_it_consensus():
    html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()
    marker = "function getConsensusUiState(data)"
    helper_source = marker + html.split(marker, 1)[1].split(
        "\n        function displayResults", 1
    )[0]
    verdict_source = "function formatVerdict(verdict)" + html.split(
        "function formatVerdict(verdict)", 1
    )[1].split("\n        function formatMoatTrend", 1)[0]
    script = f"""
const assert = require('assert');
{verdict_source}
{helper_source}
const ui = getConsensusUiState({{
  consensus_status: 'incomplete',
  consensus_verdict: 'incomplete',
  completed_analyses: 1,
  requested_analyses: 3,
  partial_indication: null
}});
assert.strictEqual(ui.headline, 'Incomplete');
assert.strictEqual(
  ui.status,
  'Incomplete — 1 of 3 requested analyses complete; at least 2 are required for consensus.'
);
assert.strictEqual(ui.partialIndication, '');
assert.strictEqual(ui.showPartialIndication, false);
"""
    result = subprocess.run(
        ["node", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, json.dumps(
        {"stdout": result.stdout, "stderr": result.stderr}
    )


def test_verification_doc_preserves_buffett_result_without_claiming_consensus():
    verification = (
        Path(__file__).parents[2] / "docs" / "verification.md"
    ).read_text()

    assert '-d \'{"ticker":"AAPL","agents":["buffett"]}\'' in verification
    assert "a complete Buffett result" in verification
    assert "consensus unavailable/incomplete" in verification
    assert "complete Buffett-only consensus" not in verification


def test_partial_disagreement_warning_remains_subordinate_in_api_and_browser(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    results = [agent_result(0, True), agent_result(1, True), agent_result(2, False)]
    results[1]["score"] = -30.0
    results[1]["verdict"] = "sell"

    with TestClient(main.app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(main, "goat_workflow", ConsensusWorkflow(results))
        response = client.post(
            "/api/analyze",
            json={"ticker": "AAPL", "agents": ["buffett", "lynch", "graham"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["consensus_verdict"] == "incomplete"
    assert body["divergence_points"] == [
        "Final consensus unavailable because 2 of 3 requested analyses completed."
    ]
    indication = body["partial_indication"]
    assert indication["verdict"] == "mixed"
    assert indication["directional_support"] is None
    assert indication["supporting_analyses"] == 0
    assert indication["divergence_points"][0].startswith("Mixed investor verdicts")

    html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()
    marker = "function getConsensusUiState(data)"
    helper_source = marker + html.split(marker, 1)[1].split(
        "\n        function displayResults", 1
    )[0]
    verdict_source = "function formatVerdict(verdict)" + html.split(
        "function formatVerdict(verdict)", 1
    )[1].split("\n        function formatMoatTrend", 1)[0]
    script = f"""
const assert = require('assert');
{verdict_source}
{helper_source}
const ui = getConsensusUiState({json.dumps(body)});
assert.match(ui.partialIndication, /Mixed investor verdicts/);
"""
    result = subprocess.run(
        ["node", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, json.dumps(
        {"stdout": result.stdout, "stderr": result.stderr}
    )
