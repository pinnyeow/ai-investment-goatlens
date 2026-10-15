"""Public API contracts for the SEC-fundamentals/Yahoo-market split."""

import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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
from data_sources.yahoo_chart import (
    YahooChartClient,
    YahooChartTemporarilyUnavailable,
    YahooPricePoint,
)
from data_sources.yahoo_ratios import (
    YahooRatioSnapshot,
    YahooRatioTemporarilyUnavailable,
)


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


class FixtureAMZNSEC(SECFundamentalsClient):
    def __init__(self, user_agent):
        ticker_map = json.loads((FIXTURES / "company_tickers.json").read_text())
        companyfacts = json.loads((FIXTURES / "companyfacts_amzn.json").read_text())
        companyfacts["facts"]["us-gaap"]["CostOfGoodsAndServicesSold"] = {
            "units": {
                "USD": [{
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                    "val": 356_414_000_000,
                    "accn": "0001018724-26-000004",
                    "fy": 2025,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2026-02-06",
                    "frame": "CY2025",
                }]
            }
        }

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


class FixtureCompleteMetricsSEC(SECFundamentalsClient):
    """Six annual periods that exercise genuine SEC normalization end to end."""

    def __init__(self, user_agent):
        def duration(year, value):
            return {
                "start": f"{year}-01-01",
                "end": f"{year}-12-31",
                "val": value,
                "accn": f"filing-{year}",
                "fy": year,
                "fp": "FY",
                "form": "10-K",
                "filed": f"{year + 1}-02-01",
            }

        def instant(year, value):
            return {**duration(year, value), "start": None}

        years = [2019, 2020, 2021, 2022, 2023, 2024]
        revenue = [90, 100, 110, 120, 130, 140]
        operating_income = [8, 10, 12, 15, 18, 21]
        net_income = [5, 6, 7, 8, 10, 12]
        equity = [48, 50, 55, 60, 65, 70]
        cash = [9, 10, 10, 12, 13, 14]
        noncurrent_debt = [32, 30, 28, 26, 24, 22]
        dividends = [0, 1, 1, 1, 1, 1]
        facts = {
            "Revenues": {"units": {"USD": [duration(y, v) for y, v in zip(years, revenue)]}},
            "OperatingIncomeLoss": {"units": {"USD": [duration(y, v) for y, v in zip(years, operating_income)]}},
            "NetIncomeLoss": {"units": {"USD": [duration(y, v) for y, v in zip(years, net_income)]}},
            "StockholdersEquity": {"units": {"USD": [instant(y, v) for y, v in zip(years, equity)]}},
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [instant(y, v) for y, v in zip(years, cash)]}},
            "LongTermDebtCurrent": {"units": {"USD": [{**instant(y, 5), "accn": f"debt-{y}" if y <= 2021 else f"filing-{y}"} for y in years]}},
            "LongTermDebtNoncurrent": {"units": {"USD": [{**instant(y, v), "accn": f"debt-{y}" if y <= 2021 else f"filing-{y}"} for y, v in zip(years, noncurrent_debt)]}},
            "ShortTermBorrowings": {"units": {"USD": [{**instant(y, 2), "accn": f"debt-{y}" if y <= 2021 else f"filing-{y}"} for y in years]}},
            "GrossProfit": {"units": {"USD": [duration(2024, 70)]}},
            "Assets": {"units": {"USD": [instant(2024, 150)]}},
            "Liabilities": {"units": {"USD": [instant(2024, 80)]}},
            "AssetsCurrent": {"units": {"USD": [instant(2024, 60)]}},
            "LiabilitiesCurrent": {"units": {"USD": [instant(2024, 30)]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [duration(2024, 30)]}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [duration(2024, 8)]}},
            "EarningsPerShareDiluted": {"units": {"USD/shares": [duration(2024, 2)]}},
            "CommonStockSharesOutstanding": {"units": {"shares": [instant(2024, 10)]}},
            "IncomeTaxExpenseBenefit": {"units": {"USD": [duration(2024, 5)]}},
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": {"units": {"USD": [duration(2024, 25)]}},
            "InterestExpenseNonoperating": {"units": {"USD": [duration(2024, 3)]}},
            "CommonStockDividendsPerShareDeclared": {"units": {"USD/shares": [duration(y, v) for y, v in zip(years, dividends)]}},
        }
        payload = {"entityName": "Metric Corp", "facts": {"us-gaap": facts}}
        ticker_map = {"0": {"cik_str": 1, "ticker": "METR", "title": "Metric Corp"}}

        async def handler(request):
            if request.url.path.endswith("/files/company_tickers.json"):
                return httpx.Response(200, json=ticker_map)
            if "/api/xbrl/companyfacts/" in request.url.path:
                return httpx.Response(200, json=payload)
            return httpx.Response(404, json={})

        super().__init__(user_agent=user_agent, transport=httpx.MockTransport(handler))


class UnavailableYahoo:
    calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def get_snapshot(self, ticker):
        self.calls.append(ticker)
        raise YahooChartTemporarilyUnavailable(
            "Yahoo Finance is temporarily unavailable."
        )


class WorkingYahoo(UnavailableYahoo):
    async def get_snapshot(self, ticker):
        self.calls.append(ticker)
        end = datetime(2026, 10, 14, 20, tzinfo=timezone.utc)

        def points(returns):
            adjusted = [100.0]
            for change in returns:
                adjusted.append(adjusted[-1] * (1 + change))
            sessions = []
            cursor = end
            while len(sessions) < len(adjusted):
                if cursor.weekday() < 5:
                    sessions.append(cursor)
                cursor -= timedelta(days=1)
            sessions.reverse()
            return [
                YahooPricePoint(
                    session_date=session.date().isoformat(),
                    timestamp=int(session.timestamp()),
                    open=value,
                    high=value,
                    low=value,
                    close=value,
                    adjusted_close=value,
                    volume=1_000_000,
                )
                for session, value in zip(sessions, adjusted)
            ]

        benchmark_returns = [
            0.001 if index % 2 == 0 else -0.001 for index in range(252)
        ]
        asset_returns = [change * 2 for change in benchmark_returns]
        asset_points = points(asset_returns)
        benchmark_points = points(benchmark_returns)
        asset_logs, _, asset_intervals = YahooChartClient._returns(asset_points)
        beta, correlation, aligned_count, aligned_start, aligned_end = (
            YahooChartClient._aligned_market_metrics(
                asset_points, benchmark_points
            )
        )
        return SimpleNamespace(
            symbol=ticker,
            currency="USD",
            current_price=200.0,
            current_price_as_of="2026-10-14T12:00:00+00:00",
            fetched_at="2026-10-14T12:00:01+00:00",
            history=asset_points,
            beta_sp500=beta,
            volatility_252d=statistics.stdev(asset_logs) * (252 ** 0.5),
            correlation_sp500=correlation,
            realized_total_return_252d=(
                asset_points[-1].adjusted_close
                / asset_points[0].adjusted_close
                - 1
            ),
            metric_session_start=asset_intervals[0][0],
            metric_session_end="2026-10-14",
            aligned_metric_session_start=aligned_start,
            aligned_metric_session_end=aligned_end,
            asset_return_observations=len(asset_logs),
            aligned_return_observations=aligned_count,
        )


class WorkingYahooRatios:
    calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def get_snapshot(self, ticker):
        self.calls.append(ticker)
        return YahooRatioSnapshot(
            symbol=ticker,
            fetched_at="2026-10-15T18:00:00+00:00",
            values={
                "roe": 0.31,
                "profit_margin": 0.17,
                "debt_to_equity": 0.46,
                "pe_ratio": 20.75,
                "pb_ratio": 4.95,
                "current_ratio": 1.03,
                "dividend_yield": 0.0,
                "gross_margin": 0.51,
                "operating_margin": 0.14,
            },
            provider_fields={
                "roe": "returnOnEquity",
                "profit_margin": "profitMargins",
                "debt_to_equity": "debtToEquity",
                "pe_ratio": "trailingPE",
                "pb_ratio": "priceToBook",
                "current_ratio": "currentRatio",
                "dividend_yield": "trailingAnnualDividendYield",
                "gross_margin": "grossMargins",
                "operating_margin": "operatingMargins",
            },
        )


class UnavailableYahooRatios:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def get_snapshot(self, ticker):
        raise YahooRatioTemporarilyUnavailable("Ratio data is unavailable.")


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


class NegativeValuationSEC(FixtureAAPLSEC):
    async def get_fundamentals(self, ticker, as_of=None):
        result = await super().get_fundamentals(ticker, as_of=as_of)
        eps = result.latest["diluted_eps"]
        equity = result.latest["equity"]
        result.latest["diluted_eps"] = FundamentalValue(
            value=-5.0,
            provenance=eps.provenance,
            availability_reason=None,
        )
        result.latest["equity"] = FundamentalValue(
            value=-1.0,
            provenance=equity.provenance,
            availability_reason=None,
        )
        return result


def test_analysis_continues_from_sec_when_yahoo_is_unavailable(monkeypatch):
    FixtureAAPLSEC.calls = []
    UnavailableYahoo.calls = []
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAAPLSEC)
    monkeypatch.setattr(main, "YahooChartClient", UnavailableYahoo)
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
            "as_of": "2025-10-31",
        },
        "market_data": {
            "available": False,
            "source": "yahoo",
            "as_of": None,
            "reason": "market_data_temporarily_unavailable",
        },
        "normalized_ratios": {
            "available": False,
            "source": "yahoo",
            "as_of": None,
            "reason": "provider_temporarily_unavailable",
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
    assert body["consensus_status"] == "incomplete"
    assert body["consensus_verdict"] == "incomplete"
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
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
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
        "as_of": "2025-10-31",
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


def test_available_yahoo_ratios_are_used_with_provider_provenance(monkeypatch):
    WorkingYahooRatios.calls = []
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "YahooRatiosClient", WorkingYahooRatios)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"ticker": "METR", "agents": ["buffett", "munger"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert WorkingYahooRatios.calls == ["METR"]
    assert body["metric_evidence"]["gross_margin"] == {
        "value": 0.51,
        "available": True,
        "source": "yahoo",
        "as_of": None,
        "formula": "provider_normalized_ratio",
        "formula_version": "yahoo_quote_summary_v1",
        "inputs": [
            {
                "kind": "provider_metric",
                "role": "ratio.gross_margin",
                "source": "yahoo",
                "symbol": "METR",
                "provider_field": "grossMargins",
                "value": 0.51,
                "retrieved_at": "2026-10-15T18:00:00+00:00",
            }
        ],
        "observation_count": 1,
        "availability_reason": None,
    }
    assert body["metric_evidence"]["roe"]["value"] == 0.31
    assert body["metric_evidence"]["roe"]["source"] == "yahoo"
    assert body["fundamentals"]["revenue"]["value"] == 140.0
    assert body["fundamentals"]["revenue"]["provenance"]["accession"] == "filing-2024"
    assert body["data_status"]["fundamentals"]["source"] == "sec"
    assert body["data_status"]["normalized_ratios"] == {
        "available": True,
        "source": "yahoo",
        "as_of": "2026-10-15T18:00:00+00:00",
        "reason": None,
    }


def test_amzn_public_analysis_completes_all_strategy_branches_without_fake_zeroes(
    monkeypatch,
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAMZNSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "YahooRatiosClient", WorkingYahooRatios)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={
                "ticker": "AMZN",
                "agents": ["buffett", "lynch", "graham", "munger", "dalio"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["requested_analyses"] == 5
    assert len(body["agent_results"]) == 5
    assert body["completed_analyses"] == 5
    assert all(result["status"] == "complete" for result in body["agent_results"])
    assert body["metric_evidence"]["gross_margin"]["source"] == "yahoo"
    assert body["metric_evidence"]["dividend_yield"]["value"] == 0.0
    assert body["metric_evidence"]["earnings_growth"]["value"] is None
    assert body["metric_evidence"]["earnings_growth"]["available"] is False
    lynch = next(result for result in body["agent_results"] if result["agent"] == "Peter Lynch")
    assert any("CAGR and PEG are not applicable" in item for item in lynch["insights"])


def test_amzn_remains_complete_when_quote_summary_is_blocked(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAMZNSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "YahooRatiosClient", UnavailableYahooRatios)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={
                "ticker": "AMZN",
                "agents": ["buffett", "lynch", "graham", "munger", "dalio"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["completed_analyses"] == 5
    assert all(result["status"] == "complete" for result in body["agent_results"])
    assert body["data_status"]["normalized_ratios"] == {
        "available": False,
        "source": "yahoo",
        "as_of": None,
        "reason": "provider_temporarily_unavailable",
    }
    assert body["metric_evidence"]["gross_margin"]["source"] == "sec"
    assert body["metric_evidence"]["gross_margin"]["formula"] == (
        "revenue_minus_cost_of_revenue_divided_by_revenue"
    )
    assert body["metric_evidence"]["debt_growth_3y"]["available"] is False
    assert body["metric_evidence"]["debt_growth_3y"]["value"] is None
    dividend = body["metric_evidence"]["dividend_yield"]
    assert dividend["available"] is False
    assert dividend["value"] is None
    assert dividend["availability_reason"] == "required_inputs_unavailable"
    graham = next(
        result for result in body["agent_results"] if result["agent"] == "Benjamin Graham"
    )
    assert graham["unavailable_metrics"] == []
    assert "Current dividend yield is unavailable" in graham["concerns"]


def test_all_five_agents_complete_from_sec_and_narrow_market_data(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={
                "ticker": "METR",
                "agents": ["buffett", "lynch", "graham", "munger", "dalio"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["requested_analyses"] == 5
    assert body["completed_analyses"] == 5
    assert body["consensus_status"] == "complete"
    assert body["agreement_score"] is not None
    required_metrics = {
        "roe",
        "profit_margin",
        "debt_to_equity",
        "free_cash_flow",
        "pe_ratio",
        "earnings_growth",
        "cash_and_equivalents",
        "pb_ratio",
        "current_ratio",
        "current_price",
        "current_assets",
        "total_liabilities",
        "shares_outstanding",
        "eps",
        "book_value_per_share",
        "dividend_yield",
        "consecutive_dividend_years",
        "gross_margin",
        "operating_margin",
        "roic",
        "revenue_growth_5y",
        "capex_to_revenue",
        "margin_trend",
        "beta",
        "volatility_252d",
        "correlation_sp500",
        "debt_growth_3y",
        "realized_total_return_252d",
    }
    assert required_metrics <= body["metric_evidence"].keys()
    assert all(body["metric_evidence"][key]["available"] for key in required_metrics)
    earnings_inputs = body["metric_evidence"]["earnings_growth"]["inputs"]
    assert [(item["role"], item["provenance"]["accession"], item["provenance"]["end"]) for item in earnings_inputs] == [
        ("income.begin", "filing-2021", "2021-12-31"),
        ("income.current", "filing-2024", "2024-12-31"),
    ]
    assert len(body["metric_evidence"]["margin_trend"]["inputs"]) == 10
    debt_inputs = body["metric_evidence"]["debt_growth_3y"]["inputs"]
    assert len(debt_inputs) == 6
    assert {
        item["provenance"]["accession"]
        for item in debt_inputs
        if item["role"].startswith("debt.begin.")
    } == {"debt-2021"}
    assert body["metric_evidence"]["consecutive_dividend_years"]["inputs"][-1] == {
        "kind": "sec_fact",
        "role": "streak.2019.terminator",
        "value": 0.0,
        "provenance": {
            "namespace": "us-gaap",
            "concept": "CommonStockDividendsPerShareDeclared",
            "unit": "USD/shares",
            "start": "2019-01-01",
            "end": "2019-12-31",
            "accession": "filing-2019",
            "form": "10-K",
            "filed": "2020-02-01",
            "fiscal_year": 2019,
            "derived": False,
            "inputs": [],
        },
    }
    assert {item["kind"] for item in body["metric_evidence"]["pe_ratio"]["inputs"]} == {
        "sec_fact",
        "market_observation",
    }
    assert body["metric_evidence"]["volatility_252d"]["inputs"] == [
        {
            "kind": "market_series",
            "role": "asset.adjusted_close_return_intervals",
            "source": "yahoo",
            "symbol": "METR",
            "start": "2025-10-27",
            "end": "2026-10-14",
            "observation_count": 252,
        }
    ]
    assert {
        (item["role"], item["symbol"], item["observation_count"])
        for item in body["metric_evidence"]["beta"]["inputs"]
    } == {
        ("asset.adjusted_close_return_intervals", "METR", 252),
        ("benchmark.sp500.adjusted_close_return_intervals", "^GSPC", 252),
    }
    roic_roles = {item["role"] for item in body["metric_evidence"]["roic"]["inputs"]}
    assert {
        "income.current.operating_income",
        "income.current.tax",
        "income.current.pretax_income",
        "capital.begin.equity",
        "capital.end.equity",
        "capital.begin.cash",
        "capital.end.cash",
        "capital.begin.debt.long_term_current",
        "capital.begin.debt.long_term_noncurrent",
        "capital.begin.debt.short_term_borrowings",
        "capital.end.debt.long_term_current",
        "capital.end.debt.long_term_noncurrent",
        "capital.end.debt.short_term_borrowings",
    } == roic_roles
    assert body["metric_evidence"]["current_price"] == {
        "value": 200.0,
        "available": True,
        "source": "yahoo",
        "as_of": "2026-10-14T12:00:00+00:00",
        "formula": "provider_regular_market_price",
        "formula_version": "yahoo_chart_v1",
        "inputs": [
            {
                "kind": "market_observation",
                "role": "price.current",
                "source": "yahoo",
                "symbol": "METR",
                "value": 200.0,
                "observed_at": "2026-10-14T12:00:00+00:00",
            }
        ],
        "observation_count": 1,
        "availability_reason": None,
    }
    assert body["metric_evidence"]["debt_growth_3y"]["formula"] == (
        "three_year_funded_debt_cagr"
    )
    assert [result["status"] for result in body["agent_results"]] == [
        "complete",
        "complete",
        "complete",
        "complete",
        "complete",
    ]


def test_price_present_does_not_overstate_missing_valuation_inputs(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", MissingEPSSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
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
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
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


def test_negative_eps_and_book_value_do_not_create_scoreable_valuation_ratios(
    monkeypatch,
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", NegativeValuationSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "AAPL", "agents": ["lynch", "graham"]}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["metric_evidence"]["pe_ratio"]["available"] is False
    assert body["metric_evidence"]["pb_ratio"]["available"] is False
    assert "pe_ratio" in body["agent_results"][0]["unavailable_metrics"]
    assert {"pe_ratio", "pb_ratio"} <= set(
        body["agent_results"][1]["unavailable_metrics"]
    )


def test_provider_valuation_ratios_recompute_price_analysis_status(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", NegativeValuationSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "YahooRatiosClient", WorkingYahooRatios)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "AAPL", "agents": ["lynch", "graham"]}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["metric_evidence"]["pe_ratio"]["available"] is True
    assert body["metric_evidence"]["pb_ratio"]["available"] is True
    assert body["data_status"]["price_dependent_analysis"] == {
        "status": "complete",
        "reason": None,
    }


def test_all_agent_market_path_never_drops_or_crashes_an_agent(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAAPLSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
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
    monkeypatch.setattr(main, "YahooChartClient", UnavailableYahoo)
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
    completed = [
        item
        for item in results
        if item["agent"] in {"Warren Buffett", "Charlie Munger"}
    ]
    unavailable = [item for item in results if item not in completed]
    assert all(item["status"] == "complete" for item in completed)
    assert all(item["score"] is not None for item in completed)
    assert all(item["verdict"] is not None for item in completed)
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
    monkeypatch.setattr(main, "YahooChartClient", UnavailableYahoo)

    with TestClient(main.app) as client:
        response = client.get("/api/fundamentals-health")
        cached_response = client.get("/api/fundamentals-health")

    assert response.status_code == 200
    assert cached_response.json() == response.json()
    assert response.json() == {
        "status": "healthy",
        "ticker": "AAPL",
        "source": "sec",
        "as_of": "2025-10-31",
    }
    assert FixtureAAPLSEC.calls == ["AAPL"]
    assert UnavailableYahoo.calls == []


@pytest.mark.asyncio
async def test_fundamentals_health_does_not_retain_provider_exception_chain(
    monkeypatch,
):
    secret = "SENTINEL_SEC_PRIVATE"

    async def unavailable():
        raise SECUnavailableError(secret)

    monkeypatch.setattr(main, "_fundamentals_health_snapshot", unavailable)
    with pytest.raises(main.HTTPException) as exc_info:
        await main.fundamentals_health()

    formatted = "".join(
        traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__
        )
    )
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert secret not in formatted


@pytest.mark.asyncio
async def test_market_health_does_not_retain_provider_exception_chain(monkeypatch):
    secret = "SENTINEL_MARKET_PRIVATE"

    async def unavailable():
        raise RuntimeError(secret)

    monkeypatch.setattr(main, "_market_data_health_snapshot", unavailable)
    with pytest.raises(main.HTTPException) as exc_info:
        await main.market_data_health()

    formatted = "".join(
        traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__
        )
    )
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert secret not in formatted


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
    monkeypatch.setattr(main, "YahooChartClient", UnavailableYahoo)

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
    monkeypatch.setattr(main, "YahooChartClient", UnavailableYahoo)

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
    monkeypatch.setattr(main, "YahooChartClient", UnavailableYahoo)

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
@pytest.mark.parametrize(
    ("include_incomplete", "expected_status", "expected_requested"),
    [
        (False, "complete", 2),
        (True, "partial", 3),
    ],
)
async def test_multi_agent_agreement_uses_all_and_only_complete_results(
    monkeypatch, include_incomplete, expected_status, expected_requested
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    complete_results = [
        {
            "agent": "High Conviction Agent",
            "style": "Quality",
            "ticker": "AAPL",
            "score": 80.0,
            "verdict": "strong_buy",
            "insights": ["Shared durable advantage", "High returns"],
            "concerns": [],
            "status": "complete",
            "unavailable_metrics": [],
        },
        {
            "agent": "Value Agent",
            "style": "Value",
            "ticker": "AAPL",
            "score": 40.0,
            "verdict": "buy",
            "insights": ["Shared durable advantage", "Valuation risk"],
            "concerns": [],
            "status": "complete",
            "unavailable_metrics": [],
        },
    ]
    if include_incomplete:
        complete_results.append({
            "agent": "Unavailable Agent",
            "style": "Market dependent",
            "ticker": "AAPL",
            "score": None,
            "verdict": None,
            "insights": [],
            "concerns": [],
            "status": "incomplete",
            "unavailable_metrics": ["current_price"],
        })
    state = {
        "ticker": "AAPL",
        "agent_results": complete_results,
        "normalized_data": {},
        "error": None,
    }

    await main.synthesize_node(state)  # type: ignore[arg-type]

    consensus = state["consensus"]
    assert consensus["status"] == expected_status
    assert consensus["verdict"] == "buy"
    assert consensus["agreement_score"] == pytest.approx(1.0)
    assert consensus["consensus_points"] == ["Shared durable advantage"]
    assert consensus["completed_analyses"] == 2
    assert consensus["requested_analyses"] == expected_requested


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
    assert "mixed" in response["consensus_verdict"]["enum"]
    assert "direction" in response["agreement_score"]["description"].lower()
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
