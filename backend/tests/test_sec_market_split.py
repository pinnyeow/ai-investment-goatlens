"""Public API contracts for the SEC-fundamentals/Yahoo-market split."""

import asyncio
import copy
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

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
    YahooSplitEvent,
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


class FixtureNVDASEC(SECFundamentalsClient):
    """Public FY2026 NVDA facts with no reported ShortTermBorrowings fact."""

    def __init__(self, user_agent):
        ticker_map = {
            "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}
        }
        companyfacts = json.loads(
            (FIXTURES / "companyfacts_nvda_debt.json").read_text()
        )

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


class FixtureReducedIssuerSEC(SECFundamentalsClient):
    ticker = ""
    cik = 0
    title = ""
    fixture = ""

    def __init__(self, user_agent):
        ticker_map = {
            "0": {
                "cik_str": self.cik,
                "ticker": self.ticker,
                "title": self.title,
            }
        }
        companyfacts = self.prepare_companyfacts(
            json.loads((FIXTURES / self.fixture).read_text())
        )

        async def handler(request):
            if request.url.path.endswith("/files/company_tickers.json"):
                return httpx.Response(200, json=ticker_map)
            if "/api/xbrl/companyfacts/" in request.url.path:
                return httpx.Response(200, json=companyfacts)
            return httpx.Response(404, json={})

        super().__init__(user_agent=user_agent, transport=httpx.MockTransport(handler))

    def prepare_companyfacts(self, companyfacts):
        return companyfacts


class FixtureTSLASEC(FixtureReducedIssuerSEC):
    ticker = "TSLA"
    cik = 1318605
    title = "TESLA, INC."
    fixture = "companyfacts_tsla_reduced.json"


class FixtureNFLXSEC(FixtureReducedIssuerSEC):
    ticker = "NFLX"
    cik = 1065280
    title = "NETFLIX INC"
    fixture = "companyfacts_nflx_reduced.json"


class FixtureWMTSEC(FixtureReducedIssuerSEC):
    ticker = "WMT"
    cik = 104169
    title = "WALMART INC."
    fixture = "companyfacts_wmt_reduced.json"


class FixtureWMTConflictingRevenuesSEC(FixtureWMTSEC):
    def prepare_companyfacts(self, companyfacts):
        revenues = companyfacts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
        conflicting = dict(revenues[-1])
        conflicting["val"] += 1
        revenues.append(conflicting)
        return companyfacts


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


class FixtureRepresentativeUnreportedShortTermSEC(FixtureCompleteMetricsSEC):
    """Current filing omits a standard fact that exists only in stale history."""

    async def get_fundamentals(self, ticker, as_of=None):
        result = await super().get_fundamentals("METR", as_of=as_of)
        historical_short_term = []
        for item in result.history["short_term_borrowings"]:
            assert item.provenance is not None
            historical_short_term.append(
                replace(
                    item,
                    provenance=replace(
                        item.provenance,
                        end="2018-06-30",
                        accession="historical-short-term",
                        filed=date(2018, 8, 3),
                        fiscal_year=2018,
                    ),
                )
            )
        latest = dict(result.latest)
        latest["short_term_borrowings"] = historical_short_term[0]
        history = {key: list(items) for key, items in result.history.items()}
        history["short_term_borrowings"] = historical_short_term
        return replace(result, ticker=ticker, latest=latest, history=history)


class FixtureConflictingShortTermSEC(FixtureRepresentativeUnreportedShortTermSEC):
    async def get_fundamentals(self, ticker, as_of=None):
        result = await super().get_fundamentals(ticker, as_of=as_of)
        latest = dict(result.latest)
        latest["short_term_borrowings"] = replace(
            latest["short_term_borrowings"],
            availability_reason="conflicting_standard_concepts",
        )
        return replace(result, latest=latest)


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
            cache_age_seconds=0.0,
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
            split_events=[],
            benchmark_history=benchmark_points,
        )


def _refresh_market_metrics(snapshot):
    asset_logs, _, asset_intervals = YahooChartClient._returns(snapshot.history)
    beta, correlation, aligned_count, aligned_start, aligned_end = (
        YahooChartClient._aligned_market_metrics(
            snapshot.history, snapshot.benchmark_history
        )
    )
    snapshot.beta_sp500 = beta
    snapshot.correlation_sp500 = correlation
    snapshot.aligned_return_observations = aligned_count
    snapshot.aligned_metric_session_start = aligned_start
    snapshot.aligned_metric_session_end = aligned_end
    snapshot.asset_return_observations = len(asset_logs)
    snapshot.metric_session_start = asset_intervals[0][0]
    snapshot.metric_session_end = asset_intervals[-1][1]
    snapshot.volatility_252d = statistics.stdev(asset_logs) * (252 ** 0.5)
    snapshot.realized_total_return_252d = (
        snapshot.history[-1].adjusted_close
        / snapshot.history[0].adjusted_close
        - 1
    )
    return snapshot


class ZeroBenchmarkVarianceYahoo(WorkingYahoo):
    async def get_snapshot(self, ticker):
        snapshot = await super().get_snapshot(ticker)
        snapshot.benchmark_history = [
            replace(point, adjusted_close=100.0)
            for point in snapshot.benchmark_history
        ]
        return _refresh_market_metrics(snapshot)


class ZeroAssetVarianceYahoo(WorkingYahoo):
    async def get_snapshot(self, ticker):
        snapshot = await super().get_snapshot(ticker)
        snapshot.history = [
            replace(point, adjusted_close=100.0) for point in snapshot.history
        ]
        return _refresh_market_metrics(snapshot)


class SparseAlignedHistoryYahoo(WorkingYahoo):
    async def get_snapshot(self, ticker):
        snapshot = await super().get_snapshot(ticker)
        snapshot.benchmark_history = [
            replace(point, session_date=f"2020-{index // 28 + 1:02d}-{index % 28 + 1:02d}")
            if index < 127
            else point
            for index, point in enumerate(snapshot.benchmark_history)
        ]
        return _refresh_market_metrics(snapshot)


class NonUSDYahoo(WorkingYahoo):
    async def get_snapshot(self, ticker):
        snapshot = await super().get_snapshot(ticker)
        snapshot.currency = "EUR"
        return snapshot


class PostFilingSplitYahoo(WorkingYahoo):
    async def get_snapshot(self, ticker):
        snapshot = await super().get_snapshot(ticker)
        snapshot.split_events = [
            YahooSplitEvent(
                effective_date="2025-06-01",
                timestamp=1_748_739_600,
                numerator=2.0,
                denominator=1.0,
            )
        ]
        return snapshot


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


class UnsupportedShareBasisSEC(FixtureCompleteMetricsSEC):
    async def get_fundamentals(self, ticker, as_of=None):
        result = await super().get_fundamentals(ticker, as_of=as_of)
        shares = result.latest["shares_outstanding"]
        assert shares.provenance is not None
        unsupported = replace(
            shares,
            provenance=replace(shares.provenance, concept="DepositarySharesOutstanding"),
        )
        result.latest["shares_outstanding"] = unsupported
        result.history["shares_outstanding"][0] = unsupported
        return result


class FixtureShareClassSEC(FixtureCompleteMetricsSEC):
    async def get_fundamentals(self, ticker, as_of=None):
        result = await super().get_fundamentals("METR", as_of=as_of)
        return replace(result, ticker=ticker)


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
            "retrieved_at": None,
            "currency": None,
            "cache_age_seconds": None,
            "freshness": "unknown",
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
    assert body["consensus_status"] == "incomplete"
    assert body["consensus_verdict"] == "incomplete"
    assert body["agreement_score"] is None
    assert body["partial_indication"] is None
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
        "retrieved_at": "2026-10-14T12:00:01+00:00",
        "currency": "USD",
        "cache_age_seconds": 0.0,
        "freshness": "fresh",
        "reason": None,
    }
    assert status["price_dependent_analysis"] == {
        "status": "complete",
        "reason": None,
    }
    assert FixtureAAPLSEC.calls == ["AAPL"]
    assert WorkingYahoo.calls == ["AAPL"]


def test_sec_metrics_remain_canonical_when_market_data_is_available(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"ticker": "METR", "agents": ["buffett", "munger"]},
        )

    assert response.status_code == 200
    body = response.json()
    gross_margin = body["metric_evidence"]["gross_margin"]
    assert gross_margin["value"] == 0.5
    assert gross_margin["source"] == "sec"
    assert gross_margin["formula"] == "gross_profit_divided_by_revenue"
    assert {item["kind"] for item in gross_margin["inputs"]} == {"sec_fact"}
    assert body["metric_evidence"]["roe"]["source"] == "sec"
    assert body["fundamentals"]["revenue"]["value"] == 140.0
    assert body["fundamentals"]["revenue"]["provenance"]["accession"] == "filing-2024"
    assert body["data_status"]["fundamentals"]["source"] == "sec"
    assert "normalized_ratios" not in body["data_status"]


def test_nvda_public_evidence_uses_only_reported_compatible_debt_facts(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureNVDASEC)
    monkeypatch.setattr(main, "YahooChartClient", UnavailableYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "NVDA", "agents": ["buffett"]}
        )

    assert response.status_code == 200
    body = response.json()
    evidence = body["metric_evidence"]["debt_to_equity"]
    assert evidence["available"] is True
    assert evidence["formula"] == (
        "long_term_debt_components_plus_optional_short_term_borrowings_"
        "divided_by_equity"
    )
    assert evidence["formula_version"] == "sec_companyfacts_debt_to_equity_v3"
    assert [item["role"] for item in evidence["inputs"]] == [
        "debt.current.long_term_current",
        "debt.current.long_term_noncurrent",
        "capital.end.equity",
    ]
    assert [item["value"] for item in evidence["inputs"]] == [
        999_000_000.0,
        7_469_000_000.0,
        157_293_000_000.0,
    ]
    independently_recomputed = (999_000_000 + 7_469_000_000) / 157_293_000_000
    assert evidence["value"] == pytest.approx(independently_recomputed, rel=1e-12)
    assert body["fundamentals"]["short_term_borrowings"]["value"] is None
    assert body["fundamentals"]["short_term_borrowings"][
        "availability_reason"
    ] == "issuer_does_not_report_a_standard_entity_wide_annual_fact"


@pytest.mark.parametrize("ticker", ["MSFT", "COST"])
def test_representative_unreported_short_term_debt_preserves_five_agents(
    monkeypatch, ticker
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        main,
        "SECFundamentalsClient",
        FixtureRepresentativeUnreportedShortTermSEC,
    )
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={
                "ticker": ticker,
                "agents": ["buffett", "lynch", "graham", "munger", "dalio"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["completed_analyses"] == 5
    assert all(result["status"] == "complete" for result in body["agent_results"])
    evidence = body["metric_evidence"]["debt_to_equity"]
    assert [item["role"] for item in evidence["inputs"]] == [
        "debt.current.long_term_current",
        "debt.current.long_term_noncurrent",
        "capital.end.equity",
    ]
    assert evidence["value"] == pytest.approx((5 + 22) / 70)


def _reduced_issuer_response(monkeypatch, sec_client, ticker):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", sec_client)
    monkeypatch.setattr(main, "YahooChartClient", UnavailableYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        return client.post(
            "/api/analyze", json={"ticker": ticker, "agents": ["buffett"]}
        )


@pytest.mark.parametrize(
    ("sec_client", "ticker", "formula", "expected_roles", "expected_values"),
    [
        (
            FixtureTSLASEC,
            "TSLA",
            "debt_current_plus_long_term_debt_divided_by_equity",
            (
                "debt.current.debt_current",
                "debt.current.long_term_debt",
                "capital.end.equity",
            ),
            (1_569_000_000.0, 6_584_000_000.0, 82_137_000_000.0),
        ),
        (
            FixtureNFLXSEC,
            "NFLX",
            "short_term_borrowings_plus_long_term_debt_noncurrent_divided_by_equity",
            (
                "debt.current.short_term_borrowings",
                "debt.current.long_term_noncurrent",
                "capital.end.equity",
            ),
            (998_865_000.0, 13_463_971_000.0, 26_615_488_000.0),
        ),
    ],
    ids=["tsla-debt-current-plus-long-term-debt", "nflx-short-term-plus-noncurrent"],
)
def test_public_debt_evidence_selects_one_reviewed_funded_debt_branch(
    monkeypatch, sec_client, ticker, formula, expected_roles, expected_values
):
    response = _reduced_issuer_response(monkeypatch, sec_client, ticker)

    assert response.status_code == 200
    body = response.json()
    evidence = body["metric_evidence"]["debt_to_equity"]
    assert evidence["available"] is True
    assert evidence["formula"] == formula
    assert evidence["formula_version"] == "sec_companyfacts_debt_to_equity_v3"
    assert tuple(item["role"] for item in evidence["inputs"]) == expected_roles
    assert tuple(item["value"] for item in evidence["inputs"]) == expected_values
    assert evidence["value"] == pytest.approx(
        sum(expected_values[:-1]) / expected_values[-1], rel=1e-12
    )


@pytest.mark.parametrize(
    ("sec_client", "ticker", "omitted_role"),
    [
        (FixtureTSLASEC, "TSLA", "debt.current.long_term_debt"),
        (FixtureNFLXSEC, "NFLX", "debt.current.short_term_borrowings"),
    ],
)
def test_public_response_rejects_omitted_compatible_branch_component(
    monkeypatch, sec_client, ticker, omitted_role
):
    response = _reduced_issuer_response(monkeypatch, sec_client, ticker)
    assert response.status_code == 200
    body = response.json()
    evidence = body["metric_evidence"]["debt_to_equity"]
    evidence["inputs"] = [
        item for item in evidence["inputs"] if item["role"] != omitted_role
    ]
    evidence["value"] = sum(
        item["value"] for item in evidence["inputs"][:-1]
    ) / evidence["inputs"][-1]["value"]
    body["agent_results"][0]["required_metric_values"]["debt_to_equity"] = evidence[
        "value"
    ]

    with pytest.raises(ValidationError, match="exact input roles"):
        main.AnalysisResponse.model_validate(body)


def test_wmt_prefers_canonical_total_revenues_over_narrower_contract_revenue(
    monkeypatch,
):
    response = _reduced_issuer_response(monkeypatch, FixtureWMTSEC, "WMT")

    assert response.status_code == 200
    body = response.json()
    revenue = body["fundamentals"]["revenue"]
    assert revenue["value"] == 713_163_000_000.0
    assert revenue["provenance"]["concept"] == "Revenues"
    evidence = body["metric_evidence"]["profit_margin"]
    revenue_input = next(
        item for item in evidence["inputs"] if item["role"] == "income.current.revenue"
    )
    assert revenue_input["value"] == 713_163_000_000.0
    assert revenue_input["provenance"]["concept"] == "Revenues"
    debt = body["metric_evidence"]["debt_to_equity"]
    assert debt["formula"] == (
        "long_term_debt_components_plus_optional_short_term_borrowings_"
        "divided_by_equity"
    )
    assert [item["role"] for item in debt["inputs"]] == [
        "debt.current.long_term_current",
        "debt.current.long_term_noncurrent",
        "debt.current.short_term_borrowings",
        "capital.end.equity",
    ]
    assert [item["value"] for item in debt["inputs"]] == [
        3_542_000_000.0,
        34_624_000_000.0,
        6_596_000_000.0,
        99_617_000_000.0,
    ]
    assert all(
        item["provenance"]["concept"] != "LongTermDebt"
        for item in debt["inputs"]
    )


def test_wmt_selected_revenues_concept_rejects_same_context_conflict(monkeypatch):
    response = _reduced_issuer_response(
        monkeypatch, FixtureWMTConflictingRevenuesSEC, "WMT"
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "fundamentals_invalid_response"


def test_conflicting_short_term_debt_is_not_treated_as_legitimately_unreported(
    monkeypatch,
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureConflictingShortTermSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "CONFLICT", "agents": ["buffett"]}
        ).json()

    evidence = body["metric_evidence"]["debt_to_equity"]
    assert evidence["available"] is False
    assert evidence["inputs"] == []


def test_public_response_closes_unavailable_debt_evidence_contract(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureConflictingShortTermSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "CONFLICT", "agents": ["buffett"]}
        ).json()

    canonical = body["metric_evidence"]["debt_to_equity"]
    mutations = {}

    wrong_formula = copy.deepcopy(body)
    wrong_formula["metric_evidence"]["debt_to_equity"]["formula"] = "fabricated"
    mutations["formula"] = wrong_formula

    for field, value in {
        "source": "yahoo",
        "formula_version": "fabricated",
        "formula_owner": "yahoo",
        "availability_reason": "fabricated",
        "as_of": "2024-12-31",
        "filed_at": "2025-02-01",
        "observation_count": 1,
    }.items():
        mutated = copy.deepcopy(body)
        mutated["metric_evidence"]["debt_to_equity"][field] = value
        mutations[field] = mutated

    fabricated_input = copy.deepcopy(body)
    input_fact = copy.deepcopy(
        fabricated_input["metric_evidence"]["roe"]["inputs"][0]
    )
    input_fact["role"] = "debt.current.fabricated"
    debt = fabricated_input["metric_evidence"]["debt_to_equity"]
    debt["inputs"] = [input_fact]
    debt["as_of"] = input_fact["provenance"]["end"]
    debt["filed_at"] = input_fact["provenance"]["filed"]
    mutations["input"] = fabricated_input

    accepted = []
    for name, mutated in mutations.items():
        try:
            main.AnalysisResponse.model_validate(mutated)
        except ValidationError:
            pass
        else:
            accepted.append(name)

    assert canonical == {
        "value": None,
        "available": False,
        "source": "sec",
        "as_of": None,
        "filed_at": None,
        "formula": "reviewed_funded_debt_branch_divided_by_equity",
        "formula_version": "sec_companyfacts_debt_to_equity_v3",
        "formula_owner": "goatlens",
        "inputs": [],
        "observation_count": None,
        "availability_reason": "required_inputs_unavailable",
    }
    assert accepted == []


def test_sec_metric_dates_distinguish_period_end_from_filing_date(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["buffett"]}
        ).json()

    roe = body["metric_evidence"]["roe"]
    assert roe["as_of"] == "2024-12-31"
    assert roe["filed_at"] == "2025-02-01"


@pytest.mark.parametrize("drift", ["missing", "unavailable", "mismatched"])
def test_response_model_rejects_complete_agent_required_evidence_drift(
    monkeypatch, drift
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["buffett", "munger"]}
        ).json()

    assert body["completed_analyses"] == 2
    if drift == "missing":
        del body["metric_evidence"]["roe"]
    elif drift == "unavailable":
        body["metric_evidence"]["roe"].update(
            value=None,
            available=False,
            availability_reason="required_inputs_unavailable",
        )
    else:
        body["metric_evidence"]["roe"]["value"] += 0.01

    with pytest.raises(
        ValidationError,
        match="exact shipped registry|required metric evidence|formula evidence",
    ):
        main.AnalysisResponse.model_validate(body)


def test_completed_munger_binds_available_optional_roic(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["munger"]}
        ).json()

    result = body["agent_results"][0]
    assert result["status"] == "complete"
    assert result["required_metric_values"] == {
        metric: body["metric_evidence"][metric]["value"]
        for metric in (
            "gross_margin",
            "operating_margin",
            "roic",
            "revenue_growth_5y",
            "capex_to_revenue",
            "debt_to_equity",
            "margin_trend",
        )
    }


@pytest.mark.parametrize("mutation", ["rogue", "missing", "optional_erased"])
def test_public_response_requires_exact_metric_evidence_registry(monkeypatch, mutation):
    body = _complete_metric_response(monkeypatch, ["munger"])
    if mutation == "rogue":
        body["metric_evidence"]["rogue_metric"] = copy.deepcopy(
            body["metric_evidence"]["current_price"]
        )
    elif mutation == "missing":
        del body["metric_evidence"]["current_price"]
    else:
        del body["metric_evidence"]["roic"]
        del body["agent_results"][0]["required_metric_values"]["roic"]

    with pytest.raises(ValidationError, match="exact shipped registry"):
        main.AnalysisResponse.model_validate(body)


def test_completed_dalio_binds_available_optional_debt_metrics(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["dalio"]}
        ).json()

    consumed = body["agent_results"][0]["required_metric_values"]
    assert consumed["interest_coverage"] == pytest.approx(7.0)
    assert consumed["debt_growth_3y"] == pytest.approx(
        body["metric_evidence"]["debt_growth_3y"]["value"]
    )


def test_every_completed_strategy_binds_exact_available_consumed_values(monkeypatch):
    body = _complete_metric_response(
        monkeypatch, ["buffett", "lynch", "graham", "munger", "dalio"]
    )
    expected = {
        "Warren Buffett": {
            "roe", "profit_margin", "debt_to_equity", "free_cash_flow"
        },
        "Peter Lynch": {
            "pe_ratio", "earnings_growth", "debt_to_equity",
            "cash_and_equivalents",
        },
        "Benjamin Graham": {
            "pe_ratio", "pb_ratio", "current_ratio", "current_price",
            "current_assets", "total_liabilities", "shares_outstanding", "eps",
            "book_value_per_share", "dividend_yield",
            "consecutive_dividend_years",
        },
        "Charlie Munger": {
            "gross_margin", "operating_margin", "roic", "revenue_growth_5y",
            "capex_to_revenue", "debt_to_equity", "margin_trend",
        },
        "Ray Dalio": {
            "beta", "volatility_252d", "debt_to_equity",
            "correlation_sp500", "realized_total_return_252d",
            "interest_coverage", "debt_growth_3y",
        },
    }

    for result in body["agent_results"]:
        consumed = result["required_metric_values"]
        assert set(consumed) == expected[result["agent"]]
        assert consumed == {
            metric: body["metric_evidence"][metric]["value"]
            for metric in expected[result["agent"]]
        }


def _complete_metric_response(monkeypatch, agents):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": agents}
        )
    assert response.status_code == 200
    return response.json()


def test_public_response_rejects_tampered_price_to_sales_value(monkeypatch):
    body = _complete_metric_response(monkeypatch, ["buffett"])
    body["metric_evidence"]["ps_ratio"]["value"] = 123.0

    with pytest.raises(ValidationError, match="formula evidence"):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_tampered_formula_version(monkeypatch):
    body = _complete_metric_response(monkeypatch, ["buffett"])
    body["metric_evidence"]["roe"]["formula_version"] = "attacker_v1"

    with pytest.raises(ValidationError, match="contract is inconsistent"):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_omitted_compatible_short_term_debt(monkeypatch):
    body = _complete_metric_response(monkeypatch, ["buffett"])
    evidence = body["metric_evidence"]["debt_to_equity"]
    del evidence["inputs"][2]
    evidence["value"] = (
        evidence["inputs"][0]["value"] + evidence["inputs"][1]["value"]
    ) / evidence["inputs"][2]["value"]
    body["agent_results"][0]["required_metric_values"]["debt_to_equity"] = (
        evidence["value"]
    )

    with pytest.raises(ValidationError, match="exact input roles"):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_available_debt_with_nonpositive_equity(monkeypatch):
    body = _complete_metric_response(monkeypatch, ["buffett"])
    evidence = body["metric_evidence"]["debt_to_equity"]
    equity_input = evidence["inputs"][-1]
    equity_input["value"] = -70.0
    negative_equity_fact = copy.deepcopy(body["fundamentals"]["equity"])
    negative_equity_fact["value"] = -70.0
    body["fundamentals_history"]["equity"].append(negative_equity_fact)
    evidence["value"] = sum(
        item["value"] for item in evidence["inputs"][:-1]
    ) / equity_input["value"]
    body["agent_results"][0]["required_metric_values"]["debt_to_equity"] = evidence[
        "value"
    ]

    with pytest.raises(ValidationError, match="positive equity"):
        main.AnalysisResponse.model_validate(body)


@pytest.mark.parametrize(
    "mutation",
    ["price_timestamp", "sec_as_of", "filed_at", "observation_count"],
)
def test_public_response_binds_evidence_metadata_to_inputs_and_status(
    monkeypatch, mutation
):
    body = _complete_metric_response(monkeypatch, ["buffett", "dalio"])
    if mutation == "price_timestamp":
        for evidence in body["metric_evidence"].values():
            for item in evidence["inputs"]:
                if item["kind"] == "market_observation":
                    item["observed_at"] = "2026-10-14T11:59:59+00:00"
    elif mutation == "sec_as_of":
        body["metric_evidence"]["roe"]["as_of"] = "2020-01-01"
    elif mutation == "filed_at":
        body["metric_evidence"]["roe"]["filed_at"] = "2020-01-01"
    else:
        body["metric_evidence"]["beta"]["observation_count"] -= 1

    with pytest.raises(ValidationError, match="metadata|market status"):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_same_role_bound_to_wrong_public_fact(monkeypatch):
    body = _complete_metric_response(monkeypatch, ["buffett"])
    roe = body["metric_evidence"]["roe"]
    revenue = body["fundamentals"]["revenue"]
    roe["inputs"][0]["value"] = revenue["value"]
    roe["inputs"][0]["provenance"] = revenue["provenance"]
    roe["value"] = revenue["value"] / roe["inputs"][1]["value"]
    body["agent_results"][0]["required_metric_values"]["roe"] = roe["value"]

    with pytest.raises(
        ValidationError, match="bind to (?:current|public) fundamentals"
    ):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_tampered_price_to_sales_leaf_role_and_value(
    monkeypatch,
):
    body = _complete_metric_response(monkeypatch, ["buffett"])
    ps_ratio = body["metric_evidence"]["ps_ratio"]
    ps_ratio["inputs"][0]["role"] = "capital.end.equity"
    ps_ratio["inputs"][0]["value"] = 20.0
    ps_ratio["value"] = (
        ps_ratio["inputs"][2]["value"]
        * ps_ratio["inputs"][0]["value"]
        / ps_ratio["inputs"][1]["value"]
    )

    with pytest.raises(ValidationError, match="exact input roles"):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_tampered_roe_input_role_and_value(monkeypatch):
    body = _complete_metric_response(monkeypatch, ["buffett"])
    roe = body["metric_evidence"]["roe"]
    roe["inputs"][0]["role"] = "income.current.revenue"
    roe["inputs"][0]["value"] = 999.0
    roe["value"] = 999.0 / roe["inputs"][1]["value"]
    body["agent_results"][0]["required_metric_values"]["roe"] = roe["value"]

    with pytest.raises(ValidationError, match="formula evidence"):
        main.AnalysisResponse.model_validate(body)


@pytest.mark.parametrize(
    "mutation",
    [
        "historical_current_role",
        "reversed_growth",
        "mixed_period_roe",
        "mixed_period_start",
    ],
)
def test_public_response_enforces_sec_role_chronology_and_filing_context(
    monkeypatch, mutation
):
    agents = ["lynch"] if mutation == "historical_current_role" else ["munger", "buffett"]
    body = _complete_metric_response(monkeypatch, agents)
    if mutation == "historical_current_role":
        evidence = body["metric_evidence"]["cash_and_equivalents"]
        historical = body["fundamentals_history"]["cash_and_equivalents"][-1]
        evidence["inputs"][0]["value"] = historical["value"]
        evidence["inputs"][0]["provenance"] = historical["provenance"]
        evidence["value"] = historical["value"]
        evidence["as_of"] = historical["provenance"]["end"]
        evidence["filed_at"] = historical["provenance"]["filed"]
        body["agent_results"][0]["required_metric_values"][
            "cash_and_equivalents"
        ] = evidence["value"]
    elif mutation == "reversed_growth":
        evidence = body["metric_evidence"]["revenue_growth_5y"]
        begin = copy.deepcopy(evidence["inputs"][0])
        current = copy.deepcopy(evidence["inputs"][1])
        evidence["inputs"][0].update(value=current["value"], provenance=current["provenance"])
        evidence["inputs"][1].update(value=begin["value"], provenance=begin["provenance"])
        evidence["value"] = (begin["value"] / current["value"]) ** (1 / 5) - 1
        evidence["as_of"] = current["provenance"]["end"]
        evidence["filed_at"] = current["provenance"]["filed"]
        munger = next(
            result for result in body["agent_results"]
            if result["agent"] == "Charlie Munger"
        )
        munger["required_metric_values"]["revenue_growth_5y"] = evidence["value"]
    elif mutation == "mixed_period_roe":
        evidence = body["metric_evidence"]["roe"]
        historical = body["fundamentals_history"]["equity"][-1]
        evidence["inputs"][1].update(
            value=historical["value"], provenance=historical["provenance"]
        )
        evidence["value"] = evidence["inputs"][0]["value"] / historical["value"]
        evidence["as_of"] = max(
            item["provenance"]["end"] for item in evidence["inputs"]
        )
        evidence["filed_at"] = max(
            item["provenance"]["filed"] for item in evidence["inputs"]
        )
        buffett = next(
            result for result in body["agent_results"]
            if result["agent"] == "Warren Buffett"
        )
        buffett["required_metric_values"]["roe"] = evidence["value"]
    else:
        body["fundamentals"]["capex_payments"]["provenance"]["start"] = (
            "2024-02-01"
        )
        for evidence in body["metric_evidence"].values():
            for item in evidence["inputs"]:
                if item.get("role") == "cashflow.current.capex":
                    item["provenance"]["start"] = "2024-02-01"

    with pytest.raises(ValidationError, match="current fundamentals|chronology|filing context"):
        main.AnalysisResponse.model_validate(body)


@pytest.mark.parametrize("mutation", ["missing", "mismatch", "rogue"])
def test_completed_result_rejects_consumed_metric_key_or_value_drift(
    monkeypatch, mutation
):
    body = _complete_metric_response(monkeypatch, ["munger"])
    consumed = body["agent_results"][0]["required_metric_values"]
    if mutation == "missing":
        del consumed["roic"]
    elif mutation == "mismatch":
        consumed["roic"] += 0.01
    else:
        consumed["rogue_metric"] = 1.0

    with pytest.raises(ValidationError, match="consumed metric evidence|required metric evidence"):
        main.AnalysisResponse.model_validate(body)


def test_completed_lynch_binds_available_context_and_excludes_unavailable_optional(
    monkeypatch,
):
    body = _complete_metric_response(monkeypatch, ["lynch"])
    consumed = body["agent_results"][0]["required_metric_values"]

    assert set(consumed) == {
        "pe_ratio", "earnings_growth", "debt_to_equity", "cash_and_equivalents"
    }
    assert "institutional_ownership" not in consumed


@pytest.mark.parametrize(
    "mutation",
    ["unknown_identity", "missing_row", "wrong_order", "wrong_score", "wrong_verdict"],
)
def test_public_response_rejects_agent_identity_or_comparison_projection_drift(
    monkeypatch, mutation
):
    body = _complete_metric_response(monkeypatch, ["buffett", "munger"])
    if mutation == "unknown_identity":
        body["agent_results"][0]["agent"] = "Unknown Completed Agent"
        body["comparison_table"]["agents"][0]["name"] = "Unknown Completed Agent"
    elif mutation == "missing_row":
        body["comparison_table"]["agents"].pop()
    elif mutation == "wrong_order":
        body["comparison_table"]["agents"].reverse()
    elif mutation == "wrong_score":
        body["comparison_table"]["agents"][0]["score"] += 1
    else:
        body["comparison_table"]["agents"][0]["verdict"] = "sell"

    with pytest.raises(ValidationError, match="known identity|exactly project"):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_unknown_incomplete_agent_identity(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", NonUSDYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["lynch", "graham"]}
        ).json()

    assert all(result["status"] == "incomplete" for result in body["agent_results"])
    body["agent_results"][0]["agent"] = "Unknown Investor"
    body["comparison_table"]["agents"][0]["name"] = "Unknown Investor"

    with pytest.raises(ValidationError, match="known identity"):
        main.AnalysisResponse.model_validate(body)


def test_completed_lynch_turnaround_binds_branch_evidence_not_cagr(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAMZNSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "AMZN", "agents": ["lynch"]}
        )
    assert response.status_code == 200
    result = response.json()["agent_results"][0]
    assert result["status"] == "complete"
    assert "earnings_turnaround" in result["required_metric_values"]
    assert "earnings_growth" not in result["required_metric_values"]


def test_non_usd_quote_fails_price_sec_metrics_closed_at_public_seam(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", NonUSDYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["lynch", "graham"]}
        )

    assert response.status_code == 200
    body = response.json()
    for metric in ("pe_ratio", "pb_ratio", "ps_ratio", "market_cap", "dividend_yield"):
        assert body["metric_evidence"][metric]["available"] is False
        assert body["metric_evidence"][metric]["availability_reason"] == (
            "non_usd_quote_currency"
        )
    assert all(result["status"] == "incomplete" for result in body["agent_results"])
    assert body["consensus_status"] == "incomplete"


def test_unproven_share_class_fails_only_sec_price_metrics_closed(monkeypatch):
    WorkingYahoo.calls = []
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureShareClassSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={
                "ticker": "BRK.B",
                "agents": ["buffett", "lynch", "graham", "munger", "dalio"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "BRK.B"
    assert WorkingYahoo.calls == ["BRK.B"]
    reason = "unproven_share_class_price_sec_compatibility"
    for metric in ("pe_ratio", "pb_ratio", "ps_ratio", "market_cap", "dividend_yield"):
        assert body["metric_evidence"][metric]["available"] is False
        assert body["metric_evidence"][metric]["availability_reason"] == reason

    for metric in (
        "current_price",
        "beta",
        "volatility_252d",
        "correlation_sp500",
        "realized_total_return_252d",
    ):
        assert body["metric_evidence"][metric]["available"] is True

    statuses = {result["agent"]: result["status"] for result in body["agent_results"]}
    assert statuses == {
        "Warren Buffett": "complete",
        "Peter Lynch": "incomplete",
        "Benjamin Graham": "incomplete",
        "Charlie Munger": "complete",
        "Ray Dalio": "complete",
    }


def test_unsupported_sec_share_basis_fails_price_sec_metrics_closed(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", UnsupportedShareBasisSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["lynch", "graham"]}
        )

    assert response.status_code == 200
    body = response.json()
    for metric in ("pe_ratio", "pb_ratio", "ps_ratio", "market_cap", "dividend_yield"):
        assert body["metric_evidence"][metric]["available"] is False
        assert body["metric_evidence"][metric]["availability_reason"] == (
            "unsupported_filing_or_security_basis"
        )
    assert body["completed_analyses"] == 0


def test_post_filing_split_fails_affected_price_sec_metrics_closed(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", PostFilingSplitYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["lynch", "graham"]}
        )

    assert response.status_code == 200
    body = response.json()
    for metric in ("pe_ratio", "pb_ratio", "ps_ratio", "market_cap", "dividend_yield"):
        assert body["metric_evidence"][metric]["available"] is False
        assert body["metric_evidence"][metric]["availability_reason"] == (
            "post_filing_split_unadjusted_sec_basis"
        )
    assert body["completed_analyses"] == 0


def test_amzn_public_analysis_completes_all_strategy_branches_without_fake_zeroes(
    monkeypatch,
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAMZNSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
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
    assert body["metric_evidence"]["gross_margin"]["source"] == "sec"
    assert body["metric_evidence"]["dividend_yield"]["value"] is None
    assert body["metric_evidence"]["earnings_growth"]["value"] is None
    assert body["metric_evidence"]["earnings_growth"]["available"] is False
    lynch = next(result for result in body["agent_results"] if result["agent"] == "Peter Lynch")
    assert any("CAGR and PEG are not applicable" in item for item in lynch["insights"])


def test_amzn_sec_formulas_keep_all_five_agents_complete(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureAMZNSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
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
    assert "normalized_ratios" not in body["data_status"]
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


def test_public_market_status_and_openapi_expose_yahoo_retrieval_metadata(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["buffett"]}
        ).json()
        schema = client.get("/openapi.json").json()

    assert body["data_status"]["market_data"] == {
        "available": True,
        "source": "yahoo",
        "as_of": "2026-10-14T12:00:00+00:00",
        "retrieved_at": "2026-10-14T12:00:01+00:00",
        "currency": "USD",
        "cache_age_seconds": 0.0,
        "freshness": "fresh",
        "reason": None,
    }
    market_schema = schema["components"]["schemas"]["MarketAvailabilityStatus"]
    for field in ("retrieved_at", "currency", "cache_age_seconds", "freshness"):
        assert field in market_schema["properties"]
        assert market_schema["properties"][field]["description"]


def test_browser_ticker_and_formula_copy_match_runtime_contracts():
    html = (Path(__file__).parents[2] / "frontend" / "index.html").read_text()

    assert 'id="tickerInput" placeholder="AAPL" maxlength="10"' in html
    assert 'pattern="[A-Za-z0-9]+([.\\-][A-Za-z0-9]+)?"' in html
    assert "(?=.{1,10}$)[A-Z0-9]+(?:[.-][A-Z0-9]+)?" in html
    assert "Free Cash Flow" in html
    assert "Positive</td><td>+15" in html
    assert "Negative</td><td>-20" in html
    assert "owner earnings" not in html.lower()
    assert "Deleveraging</td><td>-5" not in html
    assert "Aggressive deleveraging</td><td>-15" not in html


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
    ps_inputs = body["metric_evidence"]["ps_ratio"]["inputs"]
    assert {
        item["role"] for item in ps_inputs if item["kind"] == "sec_fact"
    } == {"shares.period_end", "income.current.revenue"}
    assert next(
        item for item in ps_inputs if item["role"] == "income.current.revenue"
    )["provenance"]["concept"] == "Revenues"
    assert {item["kind"] for item in ps_inputs} == {
        "sec_fact",
        "market_observation",
    }
    volatility_input = body["metric_evidence"]["volatility_252d"]["inputs"][0]
    assert {key: value for key, value in volatility_input.items() if key != "series_id"} == {
        "kind": "market_series",
        "role": "asset.adjusted_close_return_intervals",
        "source": "yahoo",
        "symbol": "METR",
        "start": "2025-10-27",
        "end": "2026-10-14",
        "observation_count": 252,
    }
    assert volatility_input["series_id"].startswith("sha256:")
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
        "filed_at": None,
        "formula": "provider_regular_market_price",
        "formula_version": "yahoo_chart_v1",
        "formula_owner": "yahoo",
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


@pytest.mark.parametrize(
    ("yahoo_client", "expected_asset_values"),
    [
        (ZeroBenchmarkVarianceYahoo, None),
        (ZeroAssetVarianceYahoo, (0.0, 0.0)),
        (SparseAlignedHistoryYahoo, None),
    ],
    ids=[
        "zero-benchmark-variance",
        "zero-asset-variance",
        "fewer-than-126-aligned-intervals",
    ],
)
def test_degraded_paired_history_preserves_independent_public_evidence(
    monkeypatch,
    yahoo_client,
    expected_asset_values,
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", yahoo_client)
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
    main.AnalysisResponse.model_validate(body)
    evidence = body["metric_evidence"]
    assert evidence["roe"]["available"] is True
    assert evidence["current_price"]["value"] == 200.0
    assert evidence["volatility_252d"]["available"] is True
    assert evidence["realized_total_return_252d"]["available"] is True
    if expected_asset_values is not None:
        assert evidence["volatility_252d"]["value"] == expected_asset_values[0]
        assert evidence["realized_total_return_252d"]["value"] == expected_asset_values[1]
    for metric in ("beta", "correlation_sp500"):
        assert evidence[metric]["available"] is False
        assert evidence[metric]["value"] is None
        assert evidence[metric]["availability_reason"] == "required_inputs_unavailable"
        assert evidence[metric]["inputs"] == []
        assert evidence[metric]["observation_count"] is None

    referenced_series_ids = {
        item["series_id"]
        for item in evidence.values()
        if item["available"]
        for item in item["inputs"]
        if item["kind"] == "market_series"
    }
    published_series_ids = [item["series_id"] for item in body["snapshot_series"]]
    assert set(published_series_ids) == referenced_series_ids
    assert len(published_series_ids) == len(set(published_series_ids)) == 1
    assert (
        evidence["volatility_252d"]["inputs"][0]["series_id"]
        == evidence["realized_total_return_252d"]["inputs"][0]["series_id"]
    )

    agents = {item["agent"]: item for item in body["agent_results"]}
    assert {
        name: agents[name]["status"]
        for name in ("Warren Buffett", "Peter Lynch", "Benjamin Graham", "Charlie Munger")
    } == {
        "Warren Buffett": "complete",
        "Peter Lynch": "complete",
        "Benjamin Graham": "complete",
        "Charlie Munger": "complete",
    }
    assert agents["Ray Dalio"]["status"] == "incomplete"
    assert agents["Ray Dalio"]["unavailable_metrics"] == [
        "beta",
        "correlation_sp500",
    ]


def test_public_snapshot_series_recompute_all_dalio_market_metrics(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["dalio"]}
        ).json()

    series = {item["series_id"]: item for item in body["snapshot_series"]}
    evidence = body["metric_evidence"]
    for metric, version in {
        "beta": "goatlens_beta_v1",
        "volatility_252d": "goatlens_volatility_252d_v1",
        "correlation_sp500": "goatlens_correlation_sp500_v1",
        "realized_total_return_252d": "goatlens_realized_total_return_252d_v1",
    }.items():
        assert evidence[metric]["formula_owner"] == "goatlens"
        assert evidence[metric]["formula_version"] == version
        assert all(item["series_id"] in series for item in evidence[metric]["inputs"])

    asset_logs_id = evidence["volatility_252d"]["inputs"][0]["series_id"]
    assert asset_logs_id == evidence["realized_total_return_252d"]["inputs"][0][
        "series_id"
    ]
    assert evidence["beta"]["inputs"][0]["series_id"] == evidence[
        "correlation_sp500"
    ]["inputs"][0]["series_id"]
    assert evidence["beta"]["inputs"][1]["series_id"] == evidence[
        "correlation_sp500"
    ]["inputs"][1]["series_id"]
    assert len(series) == len(body["snapshot_series"]) == 3
    asset_logs = [item["value"] for item in series[asset_logs_id]["intervals"]]
    assert statistics.stdev(asset_logs) * math.sqrt(252) == pytest.approx(
        evidence["volatility_252d"]["value"]
    )
    assert math.exp(sum(asset_logs)) - 1 == pytest.approx(
        evidence["realized_total_return_252d"]["value"]
    )

    beta_inputs = evidence["beta"]["inputs"]
    asset_returns = [item["value"] for item in series[beta_inputs[0]["series_id"]]["intervals"]]
    benchmark_returns = [item["value"] for item in series[beta_inputs[1]["series_id"]]["intervals"]]
    asset_mean = statistics.mean(asset_returns)
    benchmark_mean = statistics.mean(benchmark_returns)
    covariance = sum(
        (asset - asset_mean) * (benchmark - benchmark_mean)
        for asset, benchmark in zip(asset_returns, benchmark_returns)
    ) / (len(asset_returns) - 1)
    assert covariance / statistics.variance(benchmark_returns) == pytest.approx(
        evidence["beta"]["value"]
    )
    correlation = covariance / (
        statistics.stdev(asset_returns) * statistics.stdev(benchmark_returns)
    )
    assert correlation == pytest.approx(evidence["correlation_sp500"]["value"])


def test_public_response_rejects_falsified_dalio_value_even_when_agent_value_matches(
    monkeypatch,
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["dalio"]}
        ).json()

    body["metric_evidence"]["beta"]["value"] = 99.0
    body["agent_results"][0]["required_metric_values"]["beta"] = 99.0

    with pytest.raises(ValidationError, match="recomputed value"):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_beta_without_benchmark_input(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["dalio"]}
        ).json()

    body["metric_evidence"]["beta"]["inputs"] = body["metric_evidence"]["beta"][
        "inputs"
    ][:1]

    with pytest.raises(ValidationError, match="exact series inputs"):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_duplicated_beta_asset_and_benchmark_series(
    monkeypatch,
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["dalio"]}
        ).json()

    inputs = body["metric_evidence"]["beta"]["inputs"]
    inputs[1]["series_id"] = inputs[0]["series_id"]
    body["metric_evidence"]["beta"]["value"] = 1.0
    body["agent_results"][0]["required_metric_values"]["beta"] = 1.0

    with pytest.raises(ValidationError, match="distinct series identities"):
        main.AnalysisResponse.model_validate(body)


@pytest.mark.parametrize(
    ("metric", "mutation"),
    [
        ("beta", "wrong_role"),
        ("correlation_sp500", "wrong_benchmark_symbol"),
        ("volatility_252d", "extra_input"),
        ("realized_total_return_252d", "wrong_asset_symbol"),
    ],
)
def test_public_response_rejects_malformed_dalio_series_contracts(
    monkeypatch, metric, mutation
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["dalio"]}
        ).json()

    inputs = body["metric_evidence"][metric]["inputs"]
    if mutation == "wrong_role":
        inputs[0]["role"] = "benchmark.sp500.adjusted_close_return_intervals"
    elif mutation == "wrong_benchmark_symbol":
        inputs[1]["symbol"] = "METR"
    elif mutation == "extra_input":
        inputs.append(copy.deepcopy(inputs[0]))
    else:
        inputs[0]["symbol"] = "^GSPC"

    with pytest.raises(ValidationError, match="exact series inputs"):
        main.AnalysisResponse.model_validate(body)


@pytest.mark.parametrize(
    "metric",
    ["correlation_sp500", "volatility_252d", "realized_total_return_252d"],
)
def test_public_response_rejects_other_falsified_dalio_market_values(
    monkeypatch, metric
):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["dalio"]}
        ).json()

    body["metric_evidence"][metric]["value"] = 99.0
    body["agent_results"][0]["required_metric_values"][metric] = 99.0

    with pytest.raises(ValidationError, match="recomputed value"):
        main.AnalysisResponse.model_validate(body)


@pytest.mark.parametrize("mutation", ["non_finite", "too_short_for_sample"])
def test_public_response_rejects_invalid_volatility_series(monkeypatch, mutation):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["dalio"]}
        ).json()

    volatility_input = body["metric_evidence"]["volatility_252d"]["inputs"][0]
    old_id = volatility_input["series_id"]
    series = next(item for item in body["snapshot_series"] if item["series_id"] == old_id)
    if mutation == "non_finite":
        series["intervals"][0]["value"] = float("nan")
    else:
        series["intervals"] = series["intervals"][:1]
    canonical = json.dumps(
        {
            "source": series["source"],
            "symbol": series["symbol"],
            "return_kind": series["return_kind"],
            "intervals": series["intervals"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    new_id = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    series["series_id"] = new_id
    for evidence in body["metric_evidence"].values():
        references_changed_series = False
        for typed_input in evidence["inputs"]:
            if typed_input.get("series_id") != old_id:
                continue
            references_changed_series = True
            typed_input.update(
                series_id=new_id,
                observation_count=len(series["intervals"]),
                start=series["intervals"][0]["start"],
                end=series["intervals"][-1]["end"],
            )
        if references_changed_series:
            evidence["observation_count"] = len(series["intervals"])
            evidence["as_of"] = series["intervals"][-1]["end"]

    with pytest.raises(ValidationError, match="finite and non-degenerate"):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_inconsistent_snapshot_series_references(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", FixtureCompleteMetricsSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        body = client.post(
            "/api/analyze", json={"ticker": "METR", "agents": ["dalio"]}
        ).json()

    mutations = []

    wrong_digest = copy.deepcopy(body)
    wrong_digest["snapshot_series"][0]["series_id"] = "sha256:" + "0" * 64
    mutations.append(wrong_digest)

    wrong_count = copy.deepcopy(body)
    wrong_count["metric_evidence"]["volatility_252d"]["inputs"][0][
        "observation_count"
    ] += 1
    mutations.append(wrong_count)

    wrong_window = copy.deepcopy(body)
    wrong_window["metric_evidence"]["volatility_252d"]["inputs"][0][
        "start"
    ] = "1900-01-01"
    mutations.append(wrong_window)

    wrong_kind = copy.deepcopy(body)
    volatility_series_id = wrong_kind["metric_evidence"]["volatility_252d"][
        "inputs"
    ][0]["series_id"]
    next(
        item
        for item in wrong_kind["snapshot_series"]
        if item["series_id"] == volatility_series_id
    )["return_kind"] = "simple"
    mutations.append(wrong_kind)

    for mutated in mutations:
        with pytest.raises(ValidationError):
            main.AnalysisResponse.model_validate(mutated)


def _redigest_snapshot_series(series):
    canonical = json.dumps(
        {
            "source": series["source"],
            "symbol": series["symbol"],
            "return_kind": series["return_kind"],
            "intervals": series["intervals"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    series["series_id"] = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _replace_snapshot_series_id(body, old_id, series):
    for evidence in body["metric_evidence"].values():
        changed = False
        for typed_input in evidence["inputs"]:
            if typed_input.get("series_id") != old_id:
                continue
            changed = True
            typed_input.update(
                series_id=series["series_id"],
                start=series["intervals"][0]["start"],
                end=series["intervals"][-1]["end"],
            )
        if changed:
            evidence["as_of"] = series["intervals"][-1]["end"]


def test_public_response_rejects_digested_unreferenced_snapshot_series(monkeypatch):
    body = _complete_metric_response(monkeypatch, ["dalio"])
    rogue = copy.deepcopy(body["snapshot_series"][0])
    rogue["symbol"] = "ROGUE"
    _redigest_snapshot_series(rogue)
    body["snapshot_series"].append(rogue)

    with pytest.raises(ValidationError, match="exactly match.*referenced"):
        main.AnalysisResponse.model_validate(body)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("invalid_date", "ISO dates"),
        ("reversed", "start before end"),
        ("duplicate", "strictly ascending.*distinct"),
        ("non_monotonic", "strictly ascending.*distinct"),
        ("overlapping", "non-overlapping"),
    ],
)
def test_public_response_rejects_falsified_return_interval_chronology(
    monkeypatch, mutation, expected_error
):
    body = _complete_metric_response(monkeypatch, ["dalio"])
    old_id = body["metric_evidence"]["volatility_252d"]["inputs"][0]["series_id"]
    series = next(
        item for item in body["snapshot_series"] if item["series_id"] == old_id
    )
    intervals = series["intervals"]
    if mutation == "invalid_date":
        intervals[0]["start"] = "not-a-date"
    elif mutation == "reversed":
        intervals[0]["start"], intervals[0]["end"] = (
            intervals[0]["end"],
            intervals[0]["start"],
        )
    elif mutation == "duplicate":
        intervals[1] = copy.deepcopy(intervals[0])
    elif mutation == "non_monotonic":
        intervals[0], intervals[1] = intervals[1], intervals[0]
    else:
        intervals[1]["start"] = intervals[0]["start"]
    _redigest_snapshot_series(series)
    _replace_snapshot_series_id(body, old_id, series)

    with pytest.raises(ValidationError, match=expected_error):
        main.AnalysisResponse.model_validate(body)


def test_public_response_rejects_misaligned_asset_and_benchmark_windows(monkeypatch):
    body = _complete_metric_response(monkeypatch, ["dalio"])
    benchmark_input = body["metric_evidence"]["beta"]["inputs"][1]
    old_id = benchmark_input["series_id"]
    series = next(
        item for item in body["snapshot_series"] if item["series_id"] == old_id
    )
    for interval in series["intervals"]:
        interval["start"] = (
            datetime.fromisoformat(interval["start"]) + timedelta(days=1)
        ).date().isoformat()
        interval["end"] = (
            datetime.fromisoformat(interval["end"]) + timedelta(days=1)
        ).date().isoformat()
    _redigest_snapshot_series(series)
    _replace_snapshot_series_id(body, old_id, series)

    with pytest.raises(ValidationError, match="aligned interval windows"):
        main.AnalysisResponse.model_validate(body)


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


def test_nonpositive_equity_publishes_closed_unavailable_debt_evidence(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", NegativeValuationSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "AAPL", "agents": ["buffett"]}
        )

    assert response.status_code == 200
    evidence = response.json()["metric_evidence"]["debt_to_equity"]
    assert evidence["value"] is None
    assert evidence["available"] is False
    assert evidence["inputs"] == []
    assert evidence["as_of"] is None
    assert evidence["filed_at"] is None
    assert evidence["observation_count"] is None
    assert evidence["availability_reason"] == "nonpositive_equity_not_meaningful"


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


def test_negative_equity_preserves_fact_but_fails_roe_and_leverage_closed(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", NegativeValuationSEC)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", json={"ticker": "AAPL", "agents": ["buffett", "munger"]}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["fundamentals"]["equity"]["value"] == -1.0
    for metric in ("roe", "debt_to_equity"):
        assert body["metric_evidence"][metric]["value"] is None
        assert body["metric_evidence"][metric]["available"] is False
        assert body["metric_evidence"][metric]["availability_reason"] == (
            "nonpositive_equity_not_meaningful"
        )
    assert body["completed_analyses"] == 0
    assert body["consensus_status"] == "incomplete"
    assert all(result["status"] == "incomplete" for result in body["agent_results"])


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


@pytest.mark.parametrize(
    ("ticker", "sec_client"),
    [("AAPL", FixtureAAPLSEC), ("AMZN", FixtureAMZNSEC)],
)
def test_legitimate_five_agent_responses_validate(monkeypatch, ticker, sec_client):
    monkeypatch.setenv("SEC_USER_AGENT", "GOATlens tests tests@example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "SECFundamentalsClient", sec_client)
    monkeypatch.setattr(main, "YahooChartClient", WorkingYahoo)
    monkeypatch.setattr(main, "NewsClient", NoNews)
    monkeypatch.setattr(main, "RAGRetriever", NoRAG)

    with TestClient(main.app) as client:
        response = client.post("/api/analyze", json={"ticker": ticker})

    assert response.status_code == 200
    body = response.json()
    assert len(body["agent_results"]) == 5
    assert set(body["metric_evidence"]) == set(main._METRIC_EVIDENCE_CONTRACTS)
    assert body["comparison_table"]["agents"] == [
        {
            "name": result["agent"],
            "style": result["style"],
            "verdict": result["verdict"],
            "score": result["score"],
            "key_metric": result["insights"][0] if result["insights"] else "",
        }
        for result in body["agent_results"]
    ]
    main.AnalysisResponse.model_validate(body)


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
    assert consensus["completed_analyses"] == 2
    assert consensus["requested_analyses"] == expected_requested
    if include_incomplete:
        assert consensus["verdict"] == "incomplete"
        assert consensus["agreement_score"] is None
        assert consensus["consensus_points"] == []
        assert consensus["partial_indication"] == {
            "verdict": "buy",
            "directional_support": pytest.approx(1.0),
            "supporting_analyses": 2,
            "completed_analyses": 2,
            "requested_analyses": 3,
            "excluded_agents": [
                {
                    "agent": "Unavailable Agent",
                    "unavailable_metrics": ["current_price"],
                }
            ],
            "consensus_points": ["Shared durable advantage"],
            "divergence_points": [],
        }
    else:
        assert consensus["verdict"] == "buy"
        assert consensus["agreement_score"] == pytest.approx(1.0)
        assert consensus["consensus_points"] == ["Shared durable advantage"]
        assert consensus["partial_indication"] is None


@pytest.mark.asyncio
async def test_agent_exception_is_preserved_and_single_vote_has_no_consensus(monkeypatch):
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
    assert state["consensus"]["status"] == "incomplete"
    assert state["consensus"]["verdict"] == "incomplete"
    assert state["consensus"]["agreement_score"] is None
    assert state["consensus"]["consensus_points"] == []
    assert state["consensus"]["partial_indication"] is None
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
    assert response["partial_indication"]["anyOf"][0]["$ref"].endswith(
        "/PartialIndicationResponse"
    )
    partial_indication = schemas["PartialIndicationResponse"]
    assert partial_indication["properties"]["completed_analyses"]["minimum"] == 2
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
