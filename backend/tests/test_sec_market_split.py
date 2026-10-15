"""Public API contracts for the SEC-fundamentals/Yahoo-market split."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from data_sources.sec import (
    FundamentalValue,
    SECFundamentalsClient,
    SECInvalidResponseError,
    SECIssuerNotFoundError,
    SECUnavailableError,
)
from data_sources.yahoo import YahooFinanceError


FIXTURES = Path(__file__).parent / "fixtures" / "sec"


class FixtureAAPLSEC(SECFundamentalsClient):
    calls = []

    def __init__(self, user_agent):
        ticker_map = json.loads((FIXTURES / "company_tickers.json").read_text())
        companyfacts = json.loads((FIXTURES / "companyfacts_aapl.json").read_text())

        async def handler(request):
            if request.url.path.endswith("/files/company_tickers.json"):
                return httpx.Response(200, json=ticker_map)
            if "/api/xbrl/companyfacts/" in request.url.path:
                return httpx.Response(200, json=companyfacts)
            return httpx.Response(404, json={})

        super().__init__(
            user_agent=user_agent,
            transport=httpx.MockTransport(handler),
        )

    async def get_fundamentals(self, ticker, as_of=None):
        self.calls.append(ticker)
        return await super().get_fundamentals(ticker, as_of=as_of)


class UnavailableYahoo:
    calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def get_company_data(self, ticker, years=10):
        self.calls.append(ticker)
        raise YahooFinanceError(
            "Yahoo Finance is temporarily unavailable.",
            code="market_data_temporarily_unavailable",
        )

    async def get_next_earnings_date(self, ticker):
        return None


class WorkingYahoo(UnavailableYahoo):
    async def get_company_data(self, ticker, years=10):
        self.calls.append(ticker)
        return {
            "ticker": ticker,
            "info": {
                "regularMarketPrice": 200.0,
                "longName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "beta": 1.2,
            },
            "earnings_dates": None,
            "fetched_at": "2026-10-14T12:00:00+00:00",
        }

    def normalize_earnings(self, raw_data):
        return []

    def get_earnings_streak(self, earnings):
        return {
            "streak_type": "unknown",
            "streak_count": 0,
            "beats": 0,
            "misses": 0,
            "inline": 0,
            "total": 0,
            "summary": "No earnings data available",
        }


class NoNews:
    async def should_use_tool(self, ticker, financials):
        return False


class NoRAG:
    async def retrieve_earnings_context(self, ticker, quarters=4):
        return []


class UnknownIssuerSEC:
    calls = []

    def __init__(self, user_agent):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def get_fundamentals(self, ticker, as_of=None):
        self.calls.append(ticker)
        raise SECIssuerNotFoundError(
            "Ticker is not present in the SEC company map."
        )


class UnavailableIssuerSEC(UnknownIssuerSEC):
    async def get_fundamentals(self, ticker, as_of=None):
        self.calls.append(ticker)
        raise SECUnavailableError("SEC data is temporarily unavailable.")


class InvalidResponseSEC(UnknownIssuerSEC):
    async def get_fundamentals(self, ticker, as_of=None):
        self.calls.append(ticker)
        raise SECInvalidResponseError("private provider body and request details")


class MissingEPSSEC(FixtureAAPLSEC):
    async def get_fundamentals(self, ticker, as_of=None):
        result = await super().get_fundamentals(ticker, as_of=as_of)
        result.latest["diluted_eps"] = FundamentalValue(
            value=None,
            provenance=None,
            availability_reason="issuer_does_not_report_a_standard_entity_wide_annual_fact",
        )
        return result


class ZeroEPSSEC(FixtureAAPLSEC):
    async def get_fundamentals(self, ticker, as_of=None):
        result = await super().get_fundamentals(ticker, as_of=as_of)
        original = result.latest["diluted_eps"]
        result.latest["diluted_eps"] = FundamentalValue(
            value=0,
            provenance=original.provenance,
            availability_reason=None,
        )
        return result


def test_analysis_continues_from_sec_when_yahoo_is_unavailable(monkeypatch):
    FixtureAAPLSEC.calls = []
    UnavailableYahoo.calls = []
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAAPLSEC)
    monkeypatch.setattr(main, "YahooFinanceClient", UnavailableYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"ticker": "AAPL", "time_period": "1y", "agents": ["buffett"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["company_name"] == "Apple Inc."
    assert body["data_status"] == {
        "fundamentals": {
            "available": True,
            "source": "sec",
            "as_of": "2025-09-27",
        },
        "market_data": {
            "available": False,
            "source": "yahoo",
            "as_of": None,
            "reason": "market_data_temporarily_unavailable",
        },
        "price_dependent_analysis": {
            "status": "incomplete",
            "reason": "current_market_price_unavailable",
        },
    }
    assert len(body["agent_results"]) == 1
    result = body["agent_results"][0]
    assert result["status"] == "complete"
    assert result["score"] is not None
    assert result["verdict"] is not None
    assert body["consensus_status"] == "complete"
    assert body["consensus_verdict"] == result["verdict"]
    assert body["agreement_score"] is None
    assert body["consensus_points"] == []
    revenue = body["fundamentals"]["revenue"]
    assert revenue["value"] == 416161000000
    assert revenue["provenance"] == {
        "namespace": "us-gaap",
        "concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "unit": "USD",
        "start": "2024-09-29",
        "end": "2025-09-27",
        "accession": "0000320193-25-000079",
        "form": "10-K",
        "filed": "2025-10-31",
        "fiscal_year": 2025,
        "derived": False,
        "inputs": [],
    }
    assert body["fundamentals_history"]["revenue"][0] == revenue
    assert FixtureAAPLSEC.calls == ["AAPL"]
    assert UnavailableYahoo.calls == ["AAPL"]


def test_analysis_reports_timestamped_yahoo_market_data_when_available(monkeypatch):
    FixtureAAPLSEC.calls = []
    WorkingYahoo.calls = []
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAAPLSEC)
    monkeypatch.setattr(main, "YahooFinanceClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"ticker": "AAPL", "agents": ["buffett"]},
        )

    assert response.status_code == 200
    status = response.json()["data_status"]
    assert status["fundamentals"] == {
        "available": True,
        "source": "sec",
        "as_of": "2025-09-27",
    }
    assert status["market_data"] == {
        "available": True,
        "source": "yahoo",
        "as_of": "2026-10-14T12:00:00+00:00",
        "reason": None,
    }
    assert status["price_dependent_analysis"] == {
        "status": "complete",
        "reason": None,
    }
    assert FixtureAAPLSEC.calls == ["AAPL"]
    assert WorkingYahoo.calls == ["AAPL"]


def test_price_present_does_not_overstate_missing_valuation_inputs(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", MissingEPSSEC)
    monkeypatch.setattr(main, "YahooFinanceClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "AAPL", "agents": ["buffett"]}
        )

    assert response.status_code == 200
    status = response.json()["data_status"]
    assert status["market_data"]["available"] is True
    assert status["price_dependent_analysis"] == {
        "status": "incomplete",
        "reason": "missing_inputs:eps",
    }


def test_zero_denominator_keeps_price_analysis_incomplete(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", ZeroEPSSEC)
    monkeypatch.setattr(main, "YahooFinanceClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "AAPL", "agents": ["buffett"]}
        )
    assert response.status_code == 200
    assert response.json()["data_status"]["price_dependent_analysis"] == {
        "status": "incomplete",
        "reason": "missing_outputs:pe_ratio",
    }


def test_all_agent_market_path_never_drops_or_crashes_an_agent(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAAPLSEC)
    monkeypatch.setattr(main, "YahooFinanceClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)
    with TestClient(main.app) as client:
        response = client.post("/api/analyze", json={"ticker": "AAPL"})
    assert response.status_code == 200
    results = response.json()["agent_results"]
    assert len(results) == 5
    assert all("agent_execution" not in item["unavailable_metrics"] for item in results)


def test_all_agents_return_explicit_partial_results_without_market_data(monkeypatch):
    FixtureAAPLSEC.calls = []
    UnavailableYahoo.calls = []
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAAPLSEC)
    monkeypatch.setattr(main, "YahooFinanceClient", UnavailableYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"ticker": "AAPL", "time_period": "1y"},
        )

    assert response.status_code == 200
    payload = response.json()
    results = payload["agent_results"]
    assert {item["agent"] for item in results} == {
        "Warren Buffett",
        "Peter Lynch",
        "Benjamin Graham",
        "Charlie Munger",
        "Ray Dalio",
    }
    buffett = next(item for item in results if item["agent"] == "Warren Buffett")
    unavailable = [item for item in results if item["agent"] != "Warren Buffett"]
    assert buffett["status"] == "complete"
    assert buffett["score"] is not None
    assert buffett["verdict"] is not None
    assert all(item["status"] == "incomplete" for item in unavailable)
    assert all(item["score"] is None for item in unavailable)
    assert all(item["verdict"] is None for item in unavailable)
    assert payload["consensus_status"] == "partial"
    combined_insights = " ".join(
        insight for item in results for insight in item["insights"]
    )
    assert "Conservative debt" not in combined_insights
    assert "Under-discovered by institutions" not in combined_insights
    assert "Low market correlation" not in combined_insights


def test_fundamentals_health_exercises_only_fixed_sec_path(monkeypatch):
    FixtureAAPLSEC.calls = []
    UnavailableYahoo.calls = []
    main._fundamentals_health_cache.clear()
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAAPLSEC)
    monkeypatch.setattr(main, "YahooFinanceClient", UnavailableYahoo)

    with TestClient(main.app) as client:
        response = client.get("/api/fundamentals-health")
        cached_response = client.get("/api/fundamentals-health")

    assert response.status_code == 200
    assert cached_response.json() == response.json()
    assert response.json() == {
        "status": "healthy",
        "ticker": "AAPL",
        "source": "sec",
        "as_of": "2025-09-27",
    }
    assert FixtureAAPLSEC.calls == ["AAPL"]
    assert UnavailableYahoo.calls == []


def test_failed_fundamentals_health_is_negative_cached(monkeypatch):
    UnavailableIssuerSEC.calls = []
    main._fundamentals_health_cache.clear()
    main._fundamentals_health_inflight.clear()
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.setattr(main, "SECFundamentalsClient", UnavailableIssuerSEC)

    with TestClient(main.app) as client:
        responses = [client.get("/api/fundamentals-health") for _ in range(3)]

    assert [response.status_code for response in responses] == [503, 503, 503]
    assert UnavailableIssuerSEC.calls == ["AAPL"]
    assert all(response.headers["Retry-After"] == "60" for response in responses)


@pytest.mark.asyncio
async def test_concurrent_failed_fundamentals_health_is_coalesced(monkeypatch):
    class SlowUnavailableSEC(UnavailableIssuerSEC):
        calls = []

        async def get_fundamentals(self, ticker, as_of=None):
            self.calls.append(ticker)
            await asyncio.sleep(0.01)
            raise SECUnavailableError("provider-private-detail")

    main._fundamentals_health_cache.clear()
    main._fundamentals_health_inflight.clear()
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.setattr(main, "SECFundamentalsClient", SlowUnavailableSEC)
    outcomes = await asyncio.gather(
        main._fundamentals_health_snapshot(),
        main._fundamentals_health_snapshot(),
        return_exceptions=True,
    )

    assert all(isinstance(item, SECUnavailableError) for item in outcomes)
    assert SlowUnavailableSEC.calls == ["AAPL"]


def test_canary_unexpected_exceptions_are_sanitized(monkeypatch, capsys):
    secret = "credential=PRIVATE-CANARY-SECRET internal-host.example"

    class ExplodingSEC(UnavailableIssuerSEC):
        calls = []

        async def get_fundamentals(self, ticker, as_of=None):
            self.calls.append(ticker)
            raise RuntimeError(secret)

    async def explode_market_health():
        raise RuntimeError(secret)

    main._fundamentals_health_cache.clear()
    main._fundamentals_health_inflight.clear()
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.setattr(main, "SECFundamentalsClient", ExplodingSEC)
    monkeypatch.setattr(main, "_market_data_health_snapshot", explode_market_health)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        fundamentals = client.get("/api/fundamentals-health")
        market = client.get("/api/market-data-health")

    captured = capsys.readouterr()
    assert fundamentals.status_code == 502
    assert fundamentals.json()["detail"]["code"] == "fundamentals_failure"
    assert market.status_code == 502
    assert market.json()["detail"]["code"] == "market_data_failure"
    assert secret not in fundamentals.text
    assert secret not in market.text
    assert secret not in captured.out
    assert secret not in captured.err


def test_real_server_canary_logs_do_not_contain_unexpected_exception_secret():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "tests.canary_probe_app:app",
            "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=Path(__file__).parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "SEC_USER_AGENT": "GOATlens tests tests@example.com"},
    )
    response = None
    analysis_response = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{port}/api/fundamentals-health",
                    timeout=1,
                )
                analysis_response = httpx.post(
                    f"http://127.0.0.1:{port}/api/analyze",
                    json={"ticker": "AAPL", "agents": ["buffett"]},
                    timeout=2,
                )
                break
            except httpx.ConnectError:
                time.sleep(0.05)
    finally:
        process.terminate()
        output, _ = process.communicate(timeout=5)

    assert response is not None
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "fundamentals_failure"
    assert analysis_response is not None
    assert analysis_response.status_code == 502
    assert analysis_response.json()["detail"] == {
        "code": "fundamentals_failure",
        "message": "SEC fundamental data request failed.",
        "retryable": False,
    }
    assert "PRIVATE-CANARY-SECRET" not in analysis_response.text
    assert "PRIVATE-CANARY-SECRET" not in output
    assert "Traceback" not in output


def test_unknown_sec_issuer_is_a_sanitized_404_without_calling_yahoo(monkeypatch):
    UnknownIssuerSEC.calls = []
    UnavailableYahoo.calls = []
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.setattr(main, "SECFundamentalsClient", UnknownIssuerSEC)
    monkeypatch.setattr(main, "YahooFinanceClient", UnavailableYahoo)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"ticker": "NOPE", "agents": ["buffett"]},
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "ticker_not_found",
            "message": "No SEC issuer found for ticker NOPE.",
            "retryable": False,
        }
    }
    assert UnknownIssuerSEC.calls == ["NOPE"]
    assert UnavailableYahoo.calls == []


def test_sec_transport_outage_is_a_sanitized_retryable_503(monkeypatch):
    UnavailableIssuerSEC.calls = []
    UnavailableYahoo.calls = []
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.setattr(main, "SECFundamentalsClient", UnavailableIssuerSEC)
    monkeypatch.setattr(main, "YahooFinanceClient", UnavailableYahoo)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"ticker": "AAPL", "agents": ["buffett"]},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "fundamentals_temporarily_unavailable",
            "message": "SEC fundamental data is temporarily unavailable. Please retry.",
            "retryable": True,
        }
    }
    assert response.headers["Retry-After"] == "60"
    assert UnavailableIssuerSEC.calls == ["AAPL"]
    assert UnavailableYahoo.calls == []


def test_missing_sec_operator_identity_is_a_sanitized_configuration_503(monkeypatch):
    UnavailableYahoo.calls = []
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", SECFundamentalsClient)
    monkeypatch.setattr(main, "YahooFinanceClient", UnavailableYahoo)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"ticker": "AAPL", "agents": ["buffett"]},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "fundamentals_configuration_error",
            "message": "SEC fundamental data is not configured.",
            "retryable": False,
        }
    }
    assert UnavailableYahoo.calls == []


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/api/analyze", {"ticker": "AAPL", "agents": ["buffett"]}),
        ("get", "/api/fundamentals-health", None),
    ],
)
def test_invalid_sec_payload_has_distinct_sanitized_public_contract(
    monkeypatch, method, path, body
):
    InvalidResponseSEC.calls = []
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.setattr(main, "SECFundamentalsClient", InvalidResponseSEC)

    with TestClient(main.app) as client:
        response = (
            getattr(client, method)(path, json=body)
            if body
            else getattr(client, method)(path)
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": {
            "code": "fundamentals_invalid_response",
            "message": "SEC returned invalid fundamental data.",
            "retryable": False,
        }
    }
    assert "private provider body" not in response.text


@pytest.mark.asyncio
async def test_agent_exception_is_preserved_and_consensus_is_partial(monkeypatch):
    class CompleteAgent:
        name = "Complete Agent"
        style = "Complete"

        def __init__(self, llm_client=None):
            pass

        async def analyze(self, *args, **kwargs):
            return {
                "agent": self.name,
                "style": self.style,
                "score": 30.0,
                "verdict": "buy",
                "insights": ["complete"],
                "concerns": [],
                "status": "complete",
                "unavailable_metrics": [],
            }

    class FailingAgent(CompleteAgent):
        name = "Failing Agent"

        async def analyze(self, *args, **kwargs):
            raise RuntimeError("private failure details")

    monkeypatch.setattr(
        main,
        "AGENT_REGISTRY",
        [("complete", CompleteAgent), ("failing", FailingAgent)],
    )
    monkeypatch.setattr(main, "_AGENT_REQUIRED_METRICS", {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = {
        "ticker": "AAPL",
        "selected_agents": [],
        "normalized_data": {},
        "earnings_data": [],
        "earnings_streak": {},
        "error": None,
    }

    await main.run_agents_node(state)
    assert len(state["agent_results"]) == 2
    failed = state["agent_results"][1]
    assert failed["agent"] == "Failing Agent"
    assert failed["status"] == "incomplete"
    assert failed["score"] is None
    assert failed["verdict"] is None
    assert "private failure details" not in str(failed)

    await main.synthesize_node(state)
    assert state["consensus"]["status"] == "partial"
    assert state["consensus"]["verdict"] == "incomplete"
    assert state["consensus"]["agreement_score"] is None
    assert state["consensus"]["consensus_points"] == []
    assert state["consensus"]["completed_analyses"] == 1
    assert state["consensus"]["requested_analyses"] == 2


@pytest.mark.asyncio
async def test_malformed_agent_return_is_preserved_as_incomplete(monkeypatch):
    class MalformedAgent:
        name = "Malformed Agent"
        style = "Malformed"

        def __init__(self, llm_client=None):
            pass

        async def analyze(self, *args, **kwargs):
            return None

    monkeypatch.setattr(main, "AGENT_REGISTRY", [("malformed", MalformedAgent)])
    monkeypatch.setattr(main, "_AGENT_REQUIRED_METRICS", {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = {
        "ticker": "AAPL",
        "selected_agents": [],
        "normalized_data": {},
        "earnings_data": [],
        "earnings_streak": {},
        "error": None,
    }

    await main.run_agents_node(state)
    assert state["agent_results"] == [{
        "agent": "Malformed Agent",
        "style": "Malformed",
        "ticker": "AAPL",
        "score": None,
        "verdict": None,
        "insights": [],
        "concerns": ["Analysis output was unavailable."],
        "status": "incomplete",
        "unavailable_metrics": ["agent_execution_failure"],
    }]
    await main.synthesize_node(state)
    assert state["consensus"]["status"] == "incomplete"


@pytest.mark.parametrize(
    "malformed_result",
    [
        {
            "agent": "Malformed Agent", "style": "Malformed", "score": None,
            "verdict": None, "insights": [], "concerns": [], "status": "complete",
        },
        {
            "agent": "Malformed Agent", "style": "Malformed", "score": float("nan"),
            "verdict": "buy", "insights": [], "concerns": [], "status": "complete",
        },
        {
            "agent": "Malformed Agent", "style": "Malformed", "score": 101,
            "verdict": "strong_buy", "insights": [], "concerns": [],
            "status": "complete",
        },
        {
            "agent": "Malformed Agent", "style": "Malformed", "score": True,
            "verdict": "hold", "insights": [], "concerns": [], "status": "complete",
        },
        {
            "agent": "Malformed Agent", "style": "Malformed", "score": 75,
            "verdict": "sell", "insights": [], "concerns": [], "status": "complete",
        },
        {
            "agent": "Spoofed Agent", "style": "Spoofed", "score": 75,
            "verdict": "strong_buy", "insights": [], "concerns": [],
            "status": "complete",
        },
        {
            "agent": "", "style": "", "score": 75, "verdict": "strong_buy",
            "insights": [], "concerns": [], "status": "complete",
        },
        {
            "agent": "Malformed Agent", "style": "Malformed", "score": 20,
            "verdict": "buy", "insights": [], "concerns": [],
            "status": "complete", "unavailable_metrics": ["revenue"],
        },
        {
            "agent": "Malformed Agent", "style": "Malformed", "score": None,
            "verdict": None, "insights": [], "concerns": [],
            "status": "incomplete", "unavailable_metrics": [],
        },
    ],
)
@pytest.mark.asyncio
async def test_semantically_malformed_complete_agent_is_incomplete(
    monkeypatch, malformed_result
):
    class MalformedAgent:
        name = "Malformed Agent"
        style = "Malformed"

        def __init__(self, llm_client=None):
            pass

        async def analyze(self, *args, **kwargs):
            return malformed_result

    monkeypatch.setattr(main, "AGENT_REGISTRY", [("malformed", MalformedAgent)])
    monkeypatch.setattr(main, "_AGENT_REQUIRED_METRICS", {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = {
        "ticker": "AAPL", "selected_agents": [], "normalized_data": {},
        "earnings_data": [], "earnings_streak": {}, "error": None,
    }

    await main.run_agents_node(state)
    result = state["agent_results"][0]
    assert result["status"] == "incomplete"
    assert result["score"] is None
    assert result["verdict"] is None
    await main.synthesize_node(state)
    assert state["consensus"]["status"] == "incomplete"


def test_invalid_requested_agent_is_rejected_before_provider_calls(monkeypatch):
    FixtureAAPLSEC.calls = []
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAAPLSEC)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "AAPL", "agents": ["unknown"]}
        )
    assert response.status_code == 422
    assert FixtureAAPLSEC.calls == []


def test_security_headers_and_same_origin_api_policy():
    with TestClient(main.app) as client:
        root = client.get("/")
        preflight = client.options(
            "/api/analyze",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert root.status_code == 200
    assert root.headers["X-Content-Type-Options"] == "nosniff"
    assert root.headers["X-Frame-Options"] == "DENY"
    assert root.headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in root.headers["Content-Security-Policy"]
    assert "access-control-allow-origin" not in preflight.headers
    assert "access-control-allow-credentials" not in preflight.headers


def test_openapi_health_routes_have_unique_operation_ids():
    main.app.openapi_schema = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        main.app.openapi()
    duplicate_warnings = [
        str(item.message) for item in caught if "Duplicate Operation ID" in str(item.message)
    ]
    assert duplicate_warnings == []


def test_openapi_constrains_sec_provenance_status_and_consensus_contracts():
    with TestClient(main.app) as client:
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
    response = schemas["AnalysisResponse"]["properties"]
    assert response["data_status"]["$ref"].endswith("/AnalysisDataStatus")
    assert response["fundamentals"]["additionalProperties"]["$ref"].endswith(
        "/FundamentalFactResponse"
    )
    assert response["fundamentals_history"]["additionalProperties"]["items"][
        "$ref"
    ].endswith("/FundamentalFactResponse")
    assert set(response["consensus_status"]["enum"]) == {
        "complete",
        "partial",
        "incomplete",
    }
    assert response["completed_analyses"]["type"] == "integer"
    assert response["completed_analyses"]["minimum"] == 0
    assert response["requested_analyses"]["type"] == "integer"
    assert response["requested_analyses"]["minimum"] == 1
    assert response["agreement_score"]["anyOf"][0]["minimum"] == 0
    assert response["agreement_score"]["anyOf"][0]["maximum"] == 1
    assert response["comparison_table"]["$ref"].endswith(
        "/ComparisonTableResponse"
    )
    agent_score = schemas["AgentResult"]["properties"]["score"]["anyOf"][0]
    assert agent_score["minimum"] == -100
    assert agent_score["maximum"] == 100
    assert set(
        schemas["AnalysisRequest"]["properties"]["time_period"]["enum"]
    ) == {"ytd", "3m", "6m", "1y", "5y"}
    assert set(
        schemas["ForwardGuidanceSummary"]["properties"]["growth_trend"]["enum"]
    ) == {"accelerating", "decelerating", "stable", "unknown"}
    provenance = schemas["FactProvenanceResponse"]
    assert set(provenance["required"]) >= {
        "concept",
        "unit",
        "end",
        "accession",
        "filed",
    }
