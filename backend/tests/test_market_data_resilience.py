"""Behavioral regression tests for resilient market-data analysis."""

import asyncio
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from curl_cffi.requests import exceptions as curl_exceptions
import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main
from data_sources import FMPClient, FMPError, YahooFinanceClient, YahooFinanceError


FALLBACK_FINANCIALS = {
    "ticker": "ACME",
    "company_name": "Fallback Corp",
    "sector": "Industrials",
    "industry": "Industrial Products",
    "market_cap": 1_000_000_000,
    "current_price": 42.0,
    "pe_ratio": 12.0,
    "pb_ratio": 1.5,
    "ps_ratio": 2.0,
    "peg_ratio": 1.0,
    "gross_margin": 0.40,
    "operating_margin": 0.20,
    "profit_margin": 0.15,
    "roe": 0.18,
    "roa": 0.10,
    "roic": 0.16,
    "revenue_growth": 0.08,
    "earnings_growth": 0.09,
    "revenue_growth_5y": 0.07,
    "current_ratio": 1.8,
    "quick_ratio": 1.4,
    "debt_to_equity": 0.3,
    "interest_coverage": 12.0,
    "eps": 3.5,
    "book_value_per_share": 28.0,
    "free_cash_flow_per_share": 4.0,
    "dividend_yield": 0.02,
    "beta": 0.9,
    "volatility_252d": 0.2,
    "insider_ownership": 0.05,
    "institutional_ownership": 0.60,
}


class RateLimitedYahoo:
    company_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def get_company_data(self, ticker, years=10):
        self.company_calls.append(ticker)
        raise YahooFinanceError("Too Many Requests. Rate limited.")

    async def get_next_earnings_date(self, ticker):
        return None


class UnknownTickerYahoo(RateLimitedYahoo):
    async def get_company_data(self, ticker, years=10):
        self.company_calls.append(ticker)
        raise YahooFinanceError(f"No data found for ticker: {ticker}")


class StructuredAuthFailureYahoo(RateLimitedYahoo):
    async def get_company_data(self, ticker, years=10):
        self.company_calls.append(ticker)
        error = YahooFinanceError("Yahoo authentication failed")
        setattr(error, "code", "market_data_temporarily_unavailable")
        raise error


class NonRetryableYahoo(RateLimitedYahoo):
    async def get_company_data(self, ticker, years=10):
        self.company_calls.append(ticker)
        raise YahooFinanceError("malformed provider payload with secret=private")


class TickerEchoingFailureYahoo(RateLimitedYahoo):
    async def get_company_data(self, ticker, years=10):
        self.company_calls.append(ticker)
        raise YahooFinanceError(f"malformed provider payload for {ticker}")


class WorkingFMP:
    company_calls = []

    def __init__(self, api_key=None):
        self.api_key = api_key

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def get_company_data(self, ticker, years=10):
        self.company_calls.append(ticker)
        return {
            "ticker": ticker,
            "profile": {
                "companyName": "Fallback Corp",
                "sector": "Industrials",
            },
        }

    def normalize_data(self, raw_data):
        return SimpleNamespace(**FALLBACK_FINANCIALS)


class UnavailableFMP(WorkingFMP):
    async def get_company_data(self, ticker, years=10):
        self.company_calls.append(ticker)
        raise RuntimeError("provider token and internal URL must stay private")


class NoNews:
    async def should_use_tool(self, ticker, financials):
        return False


class NoRAG:
    async def retrieve_earnings_context(self, ticker, quarters=4):
        return []


class NoMemory:
    def record_analysis(self, **kwargs):
        return None


class TestYahooProviderClassification:
    def test_yahoo_401_is_structured_as_transient(self, monkeypatch):
        """A crumb/auth failure must not become an unknown ticker."""

        class AuthFailureTicker:
            @property
            def info(self):
                error = RuntimeError("Yahoo request rejected")
                setattr(error, "response", SimpleNamespace(status_code=401))
                raise error

        client = YahooFinanceClient()
        monkeypatch.setattr(client, "_get_ticker", lambda ticker: AuthFailureTicker())

        with pytest.raises(YahooFinanceError) as exc_info:
            client._fetch_all_data_sync("AAPL")

        assert getattr(exc_info.value, "code", None) == (
            "market_data_temporarily_unavailable"
        )

    @pytest.mark.parametrize(
        "failure_kind",
        ["timeout", "connection", "curl_timeout", "curl_connection", "http_408"],
    )
    def test_network_capacity_failures_are_transient(self, monkeypatch, failure_kind):
        """Common retryable network failures must keep the fallback path open."""

        if failure_kind == "timeout":
            provider_error = TimeoutError("provider timed out")
        elif failure_kind == "connection":
            provider_error = ConnectionError("provider connection reset")
        elif failure_kind == "curl_timeout":
            provider_error = curl_exceptions.Timeout("provider timed out")
        elif failure_kind == "curl_connection":
            provider_error = curl_exceptions.ConnectionError(
                "provider connection reset"
            )
        else:
            provider_error = RuntimeError("provider request timeout")
            setattr(provider_error, "response", SimpleNamespace(status_code=408))

        class FailingTicker:
            @property
            def info(self):
                raise provider_error

        client = YahooFinanceClient()
        monkeypatch.setattr(client, "_get_ticker", lambda ticker: FailingTicker())

        with pytest.raises(YahooFinanceError) as exc_info:
            client._fetch_all_data_sync("AAPL")

        assert getattr(exc_info.value, "code", None) == (
            "market_data_temporarily_unavailable"
        )

    def test_provider_404_is_structured_as_unknown_ticker(self, monkeypatch):
        """A definitive provider 404 must not spend fallback quota."""
        provider_error = RuntimeError("provider returned not found")
        setattr(provider_error, "response", SimpleNamespace(status_code=404))

        class MissingTicker:
            @property
            def info(self):
                raise provider_error

        client = YahooFinanceClient()
        monkeypatch.setattr(client, "_get_ticker", lambda ticker: MissingTicker())

        with pytest.raises(YahooFinanceError) as exc_info:
            client._fetch_all_data_sync("ZZZZZ")

        assert getattr(exc_info.value, "code", None) == "ticker_not_found"

    def test_incomplete_known_ticker_is_transient(self, monkeypatch):
        """Partial valid provider data is an outage, not proof of an unknown symbol."""

        class IncompleteTicker:
            info = {"symbol": "AAPL", "longName": "Apple Inc."}

        client = YahooFinanceClient()
        monkeypatch.setattr(client, "_get_ticker", lambda ticker: IncompleteTicker())

        with pytest.raises(YahooFinanceError) as exc_info:
            client._fetch_all_data_sync("AAPL")

        assert getattr(exc_info.value, "code", None) == (
            "market_data_temporarily_unavailable"
        )


class TestFMPDiagnostics:
    def test_smoke_canary_calls_market_data_health_once(self):
        workflow = (
            Path(__file__).parents[2] / ".github" / "workflows" / "smoke-tests.yml"
        ).read_text()
        canary_step = workflow.split(
            "- name: Core market data path is healthy", maxsplit=1
        )[1].split("- name:", maxsplit=1)[0]

        assert canary_step.count("/api/market-data-health") == 1
        assert "--retry" not in canary_step

    @pytest.mark.asyncio
    async def test_http_failure_logs_safe_category_without_secret(self, caplog):
        secret = "diagnostic-secret-key"
        request = httpx.Request(
            "GET",
            f"https://financialmodelingprep.com/stable/profile?apikey={secret}",
        )
        response = httpx.Response(403, request=request)

        class FakeHTTPClient:
            async def get(self, url, params=None):
                return response

        client = object.__new__(FMPClient)
        client.api_key = secret
        setattr(client, "_client", FakeHTTPClient())

        with caplog.at_level("WARNING"), pytest.raises(FMPError) as exc_info:
            await client._request("profile", {"symbol": "AAPL"})

        formatted_traceback = "".join(
            traceback.format_exception(
                type(exc_info.value),
                exc_info.value,
                exc_info.value.__traceback__,
            )
        )
        assert "fmp_request_failed endpoint=profile category=http_status status=403" in (
            caplog.text
        )
        assert str(exc_info.value) == "FMP request failed."
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert secret not in caplog.text
        assert secret not in str(exc_info.value)
        assert secret not in formatted_traceback
        assert "financialmodelingprep.com" not in caplog.text

    @pytest.mark.asyncio
    async def test_transport_failure_does_not_copy_provider_url(self, caplog):
        secret = "diagnostic-secret-key"
        request = httpx.Request(
            "GET",
            f"https://financialmodelingprep.com/stable/profile?apikey={secret}",
        )

        class FakeHTTPClient:
            async def get(self, url, params=None):
                raise httpx.RequestError(
                    f"failed request {request.url}",
                    request=request,
                )

        client = object.__new__(FMPClient)
        client.api_key = secret
        setattr(client, "_client", FakeHTTPClient())

        with caplog.at_level("WARNING"), pytest.raises(FMPError) as exc_info:
            await client._request("profile", {"symbol": "AAPL"})

        formatted_traceback = "".join(
            traceback.format_exception(
                type(exc_info.value),
                exc_info.value,
                exc_info.value.__traceback__,
            )
        )
        assert "fmp_request_failed endpoint=profile category=transport" in caplog.text
        assert str(exc_info.value) == "FMP request failed."
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert secret not in caplog.text
        assert secret not in str(exc_info.value)
        assert secret not in formatted_traceback
        assert "financialmodelingprep.com" not in caplog.text

    @pytest.mark.asyncio
    async def test_api_error_logs_category_without_provider_message(self, caplog):
        secret = "diagnostic-secret-key"

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"Error Message": f"invalid key {secret}"}

        class FakeHTTPClient:
            async def get(self, url, params=None):
                return FakeResponse()

        client = object.__new__(FMPClient)
        client.api_key = secret
        setattr(client, "_client", FakeHTTPClient())

        with caplog.at_level("WARNING"), pytest.raises(FMPError) as exc_info:
            await client._request("profile", {"symbol": "AAPL"})

        assert "fmp_request_failed endpoint=profile category=api_error" in caplog.text
        assert str(exc_info.value) == "FMP request failed."
        assert secret not in caplog.text
        assert secret not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_malformed_endpoint_is_rejected_without_logging_it(self, caplog):
        secret = "diagnostic-secret-key"

        class FakeHTTPClient:
            async def get(self, url, params=None):
                raise AssertionError("invalid endpoint must not be requested")

        client = object.__new__(FMPClient)
        client.api_key = secret
        setattr(client, "_client", FakeHTTPClient())

        with caplog.at_level("WARNING"), pytest.raises(FMPError) as exc_info:
            await client._request(f"profile?apikey={secret}", {"symbol": "AAPL"})

        assert "fmp_request_failed endpoint=invalid category=invalid_endpoint" in (
            caplog.text
        )
        assert str(exc_info.value) == "FMP request failed."
        assert secret not in caplog.text
        assert secret not in str(exc_info.value)


class TestAnalyzeFallback:
    def test_analyze_returns_200_from_fmp_when_yahoo_is_rate_limited(self, monkeypatch):
        """A temporary Yahoo outage must not take down core stock analysis."""
        RateLimitedYahoo.company_calls = []
        WorkingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(main, "YahooFinanceClient", RateLimitedYahoo)
        monkeypatch.setattr(main, "FMPClient", WorkingFMP)
        monkeypatch.setattr(main, "NewsClient", NoNews)
        monkeypatch.setattr(main, "RAGRetriever", NoRAG)
        monkeypatch.setattr(main, "MemoryStore", NoMemory)

        with TestClient(main.app) as client:
            response = client.post(
                "/api/analyze",
                json={
                    "ticker": "acme",
                    "time_period": "1y",
                    "agents": ["buffett"],
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["ticker"] == "ACME"
        assert body["company_name"] == "Fallback Corp"
        assert body["sector"] == "Industrials"
        assert len(body["agent_results"]) == 1
        assert RateLimitedYahoo.company_calls == ["ACME"]
        assert WorkingFMP.company_calls == ["ACME"]

    def test_structured_yahoo_auth_failure_uses_fmp(self, monkeypatch):
        """A production Yahoo crumb/auth failure must activate bounded fallback."""
        StructuredAuthFailureYahoo.company_calls = []
        WorkingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(main, "YahooFinanceClient", StructuredAuthFailureYahoo)
        monkeypatch.setattr(main, "FMPClient", WorkingFMP)
        monkeypatch.setattr(main, "NewsClient", NoNews)
        monkeypatch.setattr(main, "RAGRetriever", NoRAG)
        monkeypatch.setattr(main, "MemoryStore", NoMemory)

        with TestClient(main.app) as client:
            response = client.post(
                "/api/analyze",
                json={
                    "ticker": "AAPL",
                    "time_period": "1y",
                    "agents": ["buffett"],
                },
            )

        assert response.status_code == 200
        assert response.json()["company_name"] == "Fallback Corp"
        assert StructuredAuthFailureYahoo.company_calls == ["AAPL"]
        assert WorkingFMP.company_calls == ["AAPL"]


class TestAnalyzeErrorReporting:
    def test_analysis_request_accepts_class_share_ticker(self):
        request = main.AnalysisRequest(ticker=" brk.b ")

        assert request.ticker == "BRK.B"

    def test_market_data_paths_reject_invalid_tickers_before_providers(
        self, monkeypatch
    ):
        """Every user-controlled ticker boundary applies the common grammar."""
        initializations = []

        class UnexpectedYahoo:
            def __init__(self):
                initializations.append(True)
                raise AssertionError("invalid ticker reached Yahoo")

        monkeypatch.setattr(main, "YahooFinanceClient", UnexpectedYahoo)

        with TestClient(main.app) as client:
            earnings = client.get("/api/earnings/ACME%20CONNECTION")
            prices = client.get("/api/price-history/ACME%20CONNECTION")

        assert earnings.status_code == 422
        assert prices.status_code == 422
        assert initializations == []

    def test_analyze_rejects_marker_ticker_before_calling_providers(self, monkeypatch):
        """Invalid ticker text cannot influence provider error classification."""
        RateLimitedYahoo.company_calls = []
        WorkingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setattr(main, "YahooFinanceClient", RateLimitedYahoo)
        monkeypatch.setattr(main, "FMPClient", WorkingFMP)

        with TestClient(main.app) as client:
            response = client.post(
                "/api/analyze",
                json={"ticker": "ACME TIMEOUT", "agents": ["buffett"]},
            )

        assert response.status_code == 422
        assert RateLimitedYahoo.company_calls == []
        assert WorkingFMP.company_calls == []

    def test_analyze_reports_provider_outage_as_retryable_503(self, monkeypatch):
        """Provider capacity failures are server outages, not bad user input."""
        RateLimitedYahoo.company_calls = []
        UnavailableFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(main, "YahooFinanceClient", RateLimitedYahoo)
        monkeypatch.setattr(main, "FMPClient", UnavailableFMP)

        with TestClient(main.app) as client:
            response = client.post(
                "/api/analyze",
                json={"ticker": "ACME", "agents": ["buffett"]},
            )

        assert response.status_code == 503
        assert response.json() == {
            "detail": {
                "code": "market_data_temporarily_unavailable",
                "message": "Market data is temporarily unavailable. Please retry.",
                "retryable": True,
            }
        }
        assert response.headers["Retry-After"] == "60"
        assert "provider token" not in response.text
        assert "Too Many Requests" not in response.text

    def test_analyze_reports_unknown_ticker_without_spending_fallback_quota(
        self, monkeypatch
    ):
        """A definite unknown ticker is a 404 and must not call FMP."""
        UnknownTickerYahoo.company_calls = []
        WorkingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(main, "YahooFinanceClient", UnknownTickerYahoo)
        monkeypatch.setattr(main, "FMPClient", WorkingFMP)

        with TestClient(main.app) as client:
            response = client.post(
                "/api/analyze",
                json={"ticker": "NOPE", "agents": ["buffett"]},
            )

        assert response.status_code == 404
        assert response.json() == {
            "detail": {
                "code": "ticker_not_found",
                "message": "No market data found for ticker NOPE.",
                "retryable": False,
            }
        }
        assert UnknownTickerYahoo.company_calls == ["NOPE"]
        assert WorkingFMP.company_calls == []

    def test_analyze_does_not_fallback_or_expose_non_retryable_yahoo_failure(
        self, monkeypatch
    ):
        """Permanent/provider-shape failures must not spend quota or leak details."""
        NonRetryableYahoo.company_calls = []
        WorkingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setattr(main, "YahooFinanceClient", NonRetryableYahoo)
        monkeypatch.setattr(main, "FMPClient", WorkingFMP)

        with TestClient(main.app) as client:
            response = client.post(
                "/api/analyze",
                json={"ticker": "ACME", "agents": ["buffett"]},
            )

        assert response.status_code == 502
        assert response.json() == {
            "detail": {
                "code": "market_data_failure",
                "message": "Market data request failed.",
                "retryable": False,
            }
        }
        assert "secret=private" not in response.text
        assert NonRetryableYahoo.company_calls == ["ACME"]
        assert WorkingFMP.company_calls == []

    def test_ticker_text_does_not_make_permanent_failure_retryable(self, monkeypatch):
        """A valid marker-like symbol cannot trigger fallback by appearing in an error."""
        TickerEchoingFailureYahoo.company_calls = []
        WorkingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setattr(main, "YahooFinanceClient", TickerEchoingFailureYahoo)
        monkeypatch.setattr(main, "FMPClient", WorkingFMP)

        with TestClient(main.app) as client:
            response = client.post(
                "/api/analyze",
                json={"ticker": "TIMEOUT", "agents": ["buffett"]},
            )

        assert response.status_code == 502
        assert TickerEchoingFailureYahoo.company_calls == ["TIMEOUT"]
        assert WorkingFMP.company_calls == []

    def test_analyze_sanitizes_unhandled_workflow_exceptions(self, monkeypatch):
        """Unhandled internal/provider details must not cross the HTTP boundary."""
        class ExplodingWorkflow:
            async def ainvoke(self, state):
                raise RuntimeError("credential=private internal-host.example")

        with TestClient(main.app) as client:
            monkeypatch.setattr(main, "goat_workflow", ExplodingWorkflow())
            response = client.post(
                "/api/analyze",
                json={"ticker": "ACME", "agents": ["buffett"]},
            )

        assert response.status_code == 500
        assert response.json() == {"detail": "Analysis failed"}
        assert "credential=private" not in response.text


class TestDeploymentReadiness:
    def test_health_identifies_deployed_commit(self, monkeypatch):
        """Smoke tests must know they are exercising the intended release."""
        commit = "c053e1b56d6ac92b83a6bab1ebd879a988913194"
        monkeypatch.setenv("RENDER_GIT_COMMIT", commit)

        with TestClient(main.app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["release"] == commit


class TestMarketDataCanary:
    def test_canary_reports_healthy_when_fallback_provider_works(self, monkeypatch):
        """Monitoring must exercise live market data without running the LLMs."""
        RateLimitedYahoo.company_calls = []
        WorkingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setattr(main, "YahooFinanceClient", RateLimitedYahoo)
        monkeypatch.setattr(main, "FMPClient", WorkingFMP)

        class UnexpectedNews:
            def __init__(self):
                raise AssertionError("canary must not initialize news")

        class UnexpectedRAG:
            def __init__(self):
                raise AssertionError("canary must not initialize RAG")

        monkeypatch.setattr(main, "NewsClient", UnexpectedNews)
        monkeypatch.setattr(main, "RAGRetriever", UnexpectedRAG)

        with TestClient(main.app) as client:
            response = client.get("/api/market-data-health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "ticker": "AAPL",
            "source": "fmp",
        }
        assert RateLimitedYahoo.company_calls == ["AAPL"]
        assert WorkingFMP.company_calls == ["AAPL"]

    def test_canary_does_not_accept_arbitrary_tickers(self, monkeypatch):
        """The unauthenticated canary cannot spend provider quota for user input."""
        RateLimitedYahoo.company_calls = []
        WorkingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setattr(main, "YahooFinanceClient", RateLimitedYahoo)
        monkeypatch.setattr(main, "FMPClient", WorkingFMP)

        with TestClient(main.app) as client:
            response = client.get("/api/market-data-health/EXPENSIVE")

        assert response.status_code == 404
        assert RateLimitedYahoo.company_calls == []
        assert WorkingFMP.company_calls == []

    def test_canary_reports_unknown_fixed_ticker_as_404(self, monkeypatch):
        """A definite missing canary ticker is not an upstream gateway failure."""
        UnknownTickerYahoo.company_calls = []
        WorkingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setattr(main, "YahooFinanceClient", UnknownTickerYahoo)
        monkeypatch.setattr(main, "FMPClient", WorkingFMP)

        with TestClient(main.app) as client:
            response = client.get("/api/market-data-health")

        assert response.status_code == 404
        assert response.json() == {
            "detail": {
                "code": "ticker_not_found",
                "message": "No market data found for ticker AAPL.",
                "retryable": False,
            }
        }
        assert WorkingFMP.company_calls == []

    @pytest.mark.asyncio
    async def test_concurrent_canary_fallbacks_share_one_fmp_fetch(self, monkeypatch):
        """Overlapping canary requests must spend fallback quota only once."""
        fmp_started = asyncio.Event()
        release_fmp = asyncio.Event()

        class BlockingFMP(WorkingFMP):
            company_calls = []

            async def get_company_data(self, ticker, years=10):
                self.company_calls.append(ticker)
                fmp_started.set()
                await release_fmp.wait()
                return {
                    "ticker": ticker,
                    "profile": {
                        "companyName": "Fallback Corp",
                        "sector": "Industrials",
                    },
                }

        RateLimitedYahoo.company_calls = []
        BlockingFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setattr(main, "YahooFinanceClient", RateLimitedYahoo)
        monkeypatch.setattr(main, "FMPClient", BlockingFMP)

        first = asyncio.create_task(main.market_data_health())
        await fmp_started.wait()
        second = asyncio.create_task(main.market_data_health())
        await asyncio.sleep(0)
        release_fmp.set()
        first_result, second_result = await asyncio.gather(first, second)
        results = [first_result, second_result, await main.market_data_health()]

        assert results == [
            {"status": "healthy", "ticker": "AAPL", "source": "fmp"},
            {"status": "healthy", "ticker": "AAPL", "source": "fmp"},
            {"status": "healthy", "ticker": "AAPL", "source": "fmp"},
        ]
        assert BlockingFMP.company_calls == ["AAPL"]

    def test_failed_canary_snapshot_is_cached_for_ttl(self, monkeypatch):
        """An outage must not spend fallback quota again on every poll."""
        RateLimitedYahoo.company_calls = []
        UnavailableFMP.company_calls = []
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setattr(main, "YahooFinanceClient", RateLimitedYahoo)
        monkeypatch.setattr(main, "FMPClient", UnavailableFMP)

        with TestClient(main.app) as client:
            first = client.get("/api/market-data-health")
            second = client.get("/api/market-data-health")

        assert first.status_code == 503
        assert second.status_code == 503
        assert RateLimitedYahoo.company_calls == ["AAPL"]
        assert UnavailableFMP.company_calls == ["AAPL"]


class TestInitialBrowserRequest:
    def test_analyze_click_starts_only_the_core_analysis_request(self):
        """Optional charts must not stampede Yahoo alongside core analysis."""
        html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()
        run_analysis = html.split("async function runAnalysis()", 1)[1].split(
            "async function runDemo()", 1
        )[0]

        assert run_analysis.count("fetch(") == 1
        assert "/api/analyze" in run_analysis
        assert "/api/price-history" not in run_analysis
        assert "/api/earnings" not in run_analysis

    def test_structured_analysis_error_shows_human_message(self):
        """Structured API errors must not render as '[object Object]'."""
        html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()
        function_source = "async function runAnalysis()" + html.split(
            "async function runAnalysis()", 1
        )[1].split("\n        let _loadingTimer", 1)[0]
        script = f"""
const assert = require('assert');
const tickerInput = {{ value: 'AAPL' }};
const selectedInvestors = ['buffett'];
const selectedPeriod = '1y';
const API_BASE = 'https://example.test';
let _loadingTimer = null;
let shownError = '';
global.showLoading = () => {{}};
global.hideError = () => {{}};
global.setLoadingStep = () => {{}};
global.displayResults = () => {{ throw new Error('unexpected success'); }};
global.showError = (message) => {{ shownError = message; }};
global.hideLoading = () => {{}};
global.fetch = async () => ({{
  ok: false,
  json: async () => ({{ detail: {{ message: 'Market data is temporarily unavailable. Please retry.' }} }})
}});
{function_source}
(async () => {{
  await runAnalysis();
  assert.strictEqual(shownError, 'Market data is temporarily unavailable. Please retry.');
}})().catch((error) => {{ console.error(error); process.exit(1); }});
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

    def test_lazy_earnings_deduplicates_requests_and_ignores_stale_ticker(self):
        """Repeated clicks share a request and old ticker data cannot overwrite new."""
        html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()
        function_source = "async function loadEarningsInsights(ticker)" + html.split(
            "async function loadEarningsInsights(ticker)", 1
        )[1].split("\n        /**", 1)[0]
        script = f"""
const assert = require('assert');
let currentTicker = 'AAPL';
let earningsInsightsLoadedTicker = '';
let earningsInsightsGeneration = 1;
const earningsInsightsRequests = new Map();
const API_BASE = 'https://example.test';
const elements = {{
  guidanceCard: {{ style: {{ display: 'block' }}, innerHTML: 'OLD GUIDANCE' }},
  reactionInsightsList: {{ innerHTML: '' }}
}};
global.document = {{ getElementById: (id) => elements[id] }};
const rendered = [];
global.renderForwardGuidance = (value) => rendered.push(value.marker);
global.renderReactionInsights = (value) => rendered.push(value[0].marker);
const pending = new Map();
const fetchCalls = [];
global.fetch = (url) => {{
  const ticker = url.split('/').pop();
  fetchCalls.push(ticker);
  let resolve;
  const promise = new Promise((done) => {{ resolve = done; }});
  pending.set(ticker, {{ resolve, promise }});
  return promise;
}};
{function_source}
(async () => {{
  const firstA = loadEarningsInsights('AAPL');
  const secondA = loadEarningsInsights('AAPL');
  assert.deepStrictEqual(fetchCalls, ['AAPL'], 'duplicate AAPL request');
  currentTicker = 'MSFT';
  earningsInsightsGeneration += 1;
  const firstM = loadEarningsInsights('MSFT');
  assert.strictEqual(pending.size, 2);
  pending.get('MSFT').resolve({{ ok: true, json: async () => ({{
    forward_guidance: null,
    reaction_insights: [{{ marker: 'MSFT' }}]
  }}) }});
  await firstM;
  assert.strictEqual(elements.guidanceCard.style.display, 'none');
  assert.strictEqual(elements.guidanceCard.innerHTML, '');
  pending.get('AAPL').resolve({{ ok: true, json: async () => ({{
    forward_guidance: {{ marker: 'AAPL' }},
    reaction_insights: [{{ marker: 'AAPL' }}]
  }}) }});
  await Promise.all([firstA, secondA]);
  assert.deepStrictEqual(rendered, ['MSFT']);
  assert.strictEqual(earningsInsightsLoadedTicker, 'MSFT');
}})().catch((error) => {{ console.error(error); process.exit(1); }});
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


class TestRequestCoalescing:
    @pytest.mark.asyncio
    async def test_concurrent_company_requests_share_one_provider_fetch(self, monkeypatch):
        """Concurrent cold requests for one ticker must share one upstream call."""
        from data_sources.yahoo import YahooFinanceClient, clear_cache

        clear_cache()
        provider_started = asyncio.Event()
        release_provider = asyncio.Event()
        calls = []
        fixture = {"ticker": "AAPL", "info": {"longName": "Apple Fixture"}}

        async def blocking_run_sync(self, func, *args):
            calls.append(args[0])
            provider_started.set()
            await release_provider.wait()
            return fixture

        monkeypatch.setattr(YahooFinanceClient, "_run_sync", blocking_run_sync)

        async with YahooFinanceClient() as first, YahooFinanceClient() as second:
            first_request = asyncio.create_task(first.get_company_data("aapl"))
            await provider_started.wait()
            second_request = asyncio.create_task(second.get_company_data(" AAPL "))
            await asyncio.sleep(0)
            release_provider.set()
            results = await asyncio.gather(first_request, second_request)

        assert calls == ["AAPL"]
        assert [result["info"]["longName"] for result in results] == [
            "Apple Fixture",
            "Apple Fixture",
        ]

    @pytest.mark.asyncio
    async def test_cancelled_only_waiter_is_cleaned_after_provider_completes(
        self, monkeypatch
    ):
        """Cancellation must not pin a completed provider task in the loop map."""
        from data_sources.yahoo import YahooFinanceClient, clear_cache

        clear_cache()
        provider_started = asyncio.Event()
        release_provider = asyncio.Event()
        calls = []

        async def controlled_run_sync(self, func, *args):
            calls.append(args[0])
            if len(calls) == 1:
                provider_started.set()
                await release_provider.wait()
                return {"ticker": "AAPL", "info": {"longName": "stale"}}
            return {"ticker": "AAPL", "info": {"longName": "fresh"}}

        monkeypatch.setattr(YahooFinanceClient, "_run_sync", controlled_run_sync)

        async with YahooFinanceClient() as client:
            waiter = asyncio.create_task(client.get_company_data("AAPL"))
            await provider_started.wait()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            release_provider.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            result = await client.get_company_data("AAPL")

        assert calls == ["AAPL", "AAPL"]
        assert result["info"]["longName"] == "fresh"

    @pytest.mark.asyncio
    async def test_failed_provider_task_is_cleaned_after_waiter_cancellation(
        self, monkeypatch
    ):
        """An unobserved provider failure must be consumed and allow a retry."""
        from data_sources.yahoo import YahooFinanceClient, clear_cache

        clear_cache()
        provider_started = asyncio.Event()
        release_provider = asyncio.Event()
        calls = []

        async def controlled_run_sync(self, func, *args):
            calls.append(args[0])
            if len(calls) == 1:
                provider_started.set()
                await release_provider.wait()
                raise YahooFinanceError("provider failed after cancellation")
            return {"ticker": "AAPL", "info": {"longName": "retry"}}

        monkeypatch.setattr(YahooFinanceClient, "_run_sync", controlled_run_sync)

        async with YahooFinanceClient() as client:
            waiter = asyncio.create_task(client.get_company_data("AAPL"))
            await provider_started.wait()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            release_provider.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            result = await client.get_company_data("AAPL")

        assert calls == ["AAPL", "AAPL"]
        assert result["info"]["longName"] == "retry"
