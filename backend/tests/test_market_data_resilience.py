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
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main
from data_sources import YahooFinanceClient, YahooFinanceError


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
        formatted = "".join(
            traceback.format_exception(
                type(exc_info.value), exc_info.value, exc_info.value.__traceback__
            )
        )
        assert "Yahoo request rejected" not in formatted
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

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
    def test_canary_reports_healthy_when_yahoo_works(self, monkeypatch):
        class WorkingYahoo:
            calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

            async def get_company_data(self, ticker, years=1):
                self.calls.append(ticker)
                return {"ticker": ticker, "info": {"regularMarketPrice": 42.0}}

        monkeypatch.setattr(main, "YahooFinanceClient", WorkingYahoo)

        with TestClient(main.app) as client:
            response = client.get("/api/market-data-health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "ticker": "AAPL",
            "source": "yahoo",
        }
        assert WorkingYahoo.calls == ["AAPL"]

    def test_canary_does_not_accept_arbitrary_tickers(self, monkeypatch):
        RateLimitedYahoo.company_calls = []
        monkeypatch.setattr(main, "YahooFinanceClient", RateLimitedYahoo)

        with TestClient(main.app) as client:
            response = client.get("/api/market-data-health/EXPENSIVE")

        assert response.status_code == 404
        assert RateLimitedYahoo.company_calls == []

    def test_canary_reports_unknown_fixed_ticker_as_404(self, monkeypatch):
        UnknownTickerYahoo.company_calls = []
        monkeypatch.setattr(main, "YahooFinanceClient", UnknownTickerYahoo)

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

    def test_failed_canary_snapshot_is_cached_for_ttl(self, monkeypatch):
        RateLimitedYahoo.company_calls = []
        monkeypatch.setattr(main, "YahooFinanceClient", RateLimitedYahoo)

        with TestClient(main.app) as client:
            first = client.get("/api/market-data-health")
            second = client.get("/api/market-data-health")

        assert first.status_code == 503
        assert second.status_code == 503
        assert RateLimitedYahoo.company_calls == ["AAPL"]


class TestInitialBrowserRequest:
    def test_market_status_and_incomplete_agent_helpers_are_explicit(self):
        """The UI must expose degraded market data and incomplete agent inputs."""
        html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()
        market_source = "function getMarketDataUiState(dataStatus)" + html.split(
            "function getMarketDataUiState(dataStatus)", 1
        )[1].split("\n        function formatAgentAvailability", 1)[0]
        agent_source = "function formatAgentAvailability(agent)" + html.split(
            "function formatAgentAvailability(agent)", 1
        )[1].split("\n        async function runAnalysis", 1)[0]
        verdict_source = "function formatVerdict(verdict)" + html.split(
            "function formatVerdict(verdict)", 1
        )[1].split("\n        function formatMoatTrend", 1)[0]
        escape_source = "function escapeHtml(value)" + html.split(
            "function escapeHtml(value)", 1
        )[1].split("\n        function getMarketDataUiState", 1)[0]
        script = f"""
const assert = require('assert');
{market_source}
{agent_source}
{verdict_source}
{escape_source}
const unavailable = getMarketDataUiState({{
  market_data: {{ available: false, reason: 'transient_provider_failure' }}
}});
assert.strictEqual(unavailable.shouldLoadPrice, false);
assert.match(unavailable.message, /Market data unavailable/);
assert.strictEqual(getMarketDataUiState({{ market_data: {{ available: true }} }}).shouldLoadPrice, true);
assert.strictEqual(getMarketDataUiState(undefined).shouldLoadPrice, true);
assert.strictEqual(
  formatAgentAvailability({{ status: 'incomplete', unavailable_metrics: ['current_price', 'pe_ratio'] }}),
  'Incomplete — unavailable: current price, PE ratio'
);
assert.strictEqual(formatAgentAvailability({{ status: 'complete' }}), '');
assert.strictEqual(formatVerdict('incomplete'), 'Incomplete');
assert.strictEqual(formatVerdict(null), 'Incomplete');
assert.strictEqual(formatAgentScore({{ status: 'incomplete', score: null }}), 'Unavailable');
assert.strictEqual(formatAgentScore({{ status: 'complete', score: 45 }}), '+45');
assert.strictEqual(
  escapeHtml('<img src=x onerror=alert(1)>'),
  '&lt;img src=x onerror=alert(1)&gt;'
);
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

    def test_display_results_wires_market_and_agent_availability_into_ui(self):
        html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()
        display_results = html.split("function displayResults(data)", 1)[1].split(
            "\n        function renderEarnings", 1
        )[0]

        assert 'getMarketDataUiState(data.data_status)' in display_results
        assert "marketDataNotice" in display_results
        assert "if (marketUi.shouldLoadPrice)" in display_results
        assert "formatAgentAvailability(agent)" in display_results
        assert 'class="agent-availability"' in display_results
        assert "consensusStatusNotice" in display_results
        assert "Partial consensus" in display_results
        assert "agreement is among complete analyses only" in display_results
        assert "data.completed_analyses" in display_results
        assert "data.requested_analyses" in display_results
        assert "data.agreement_score == null" in display_results
        assert "agreementValue.textContent = 'Unavailable'" in display_results
        assert 'id="consensusStatusNotice"' in html
        assert "map(p => `<li>${escapeHtml(p)}</li>`)" in display_results
        assert "<h3>${escapeHtml(agent.agent)}</h3>" in display_results
        assert "<div class=\"agent-style\">${escapeHtml(agent.style)}</div>" in display_results
        assert "map(i => `<li>${escapeHtml(i)}</li>`)" in display_results
        assert "map(c => `<li>${escapeHtml(c)}</li>`)" in display_results

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

    def test_provider_and_error_strings_are_escaped_before_inner_html(self):
        html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()
        assert "${escapeHtml(e.quarter || '—')}" in html
        assert "${escapeHtml(e.date || '—')}" in html
        assert "${escapeHtml(e.quarter || e.date)}" in html
        assert "${escapeHtml(r.period)}" in html
        assert "${escapeHtml(fg.summary)}" in html
        assert "['accelerating', 'decelerating', 'stable'].includes(fg.growth_trend)" in html
        assert "${escapeHtml(ins.quarter || ins.date || '—')}" in html
        assert "${escapeHtml(ins.insight)}" in html
        assert "${escapeHtml(streak.beats)}" in html
        assert "${escapeHtml(streak.streak_type.toUpperCase())}" in html
        assert "${escapeHtml(r.eps_num_analysts)} analysts" in html
        assert "${escapeHtml(s.value)}" in html
        assert "const gs = ['raised', 'lowered', 'maintained'].includes(ins.guidance_signal)" in html
        assert "Failed to load chart: ${escapeHtml(err.message)}" in html
        assert "Failed to load insights: ${escapeHtml(err.message)}" in html
        assert "<strong>Couldn't find that ticker.</strong> ${escapeHtml(message)}" in html

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