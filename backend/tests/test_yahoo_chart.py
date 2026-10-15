"""Contracts for the narrow, crumb-less Yahoo chart adapter."""

import asyncio
from datetime import datetime, timedelta, timezone
import math
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import main
from data_sources import yahoo_chart
from data_sources.yahoo_chart import (
    YahooChartClient,
    YahooChartMalformedResponse,
    YahooChartTemporarilyUnavailable,
    YahooPricePoint,
)


def _chart_payload(symbol, returns, *, price=200.0):
    end = datetime(2025, 12, 31, 21, tzinfo=timezone.utc)
    start = end - timedelta(days=len(returns))
    timestamps = [int((start + timedelta(days=index)).timestamp()) for index in range(len(returns) + 1)]
    adjusted = [100.0]
    for change in returns:
        adjusted.append(adjusted[-1] * (1 + change))
    return {
        "chart": {
            "result": [{
                "meta": {
                    "symbol": symbol,
                    "currency": "USD",
                    "exchangeTimezoneName": "America/New_York",
                    "regularMarketPrice": price,
                    "regularMarketTime": timestamps[-1],
                },
                "timestamp": timestamps,
                "indicators": {
                    "quote": [{
                        "open": [value - 1 for value in adjusted],
                        "high": [value + 1 for value in adjusted],
                        "low": [value - 2 for value in adjusted],
                        "close": adjusted,
                        "volume": [1_000_000 + index for index in range(len(adjusted))],
                    }],
                    "adjclose": [{"adjclose": adjusted}],
                },
            }],
            "error": None,
        }
    }


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        return None


@pytest.fixture(autouse=True)
def clear_snapshot_cache():
    clear = getattr(yahoo_chart, "clear_chart_cache", None)
    if clear is not None:
        clear()
    yield
    if clear is not None:
        clear()


@pytest.mark.asyncio
async def test_snapshot_uses_two_chart_calls_and_independent_expected_metrics():
    benchmark_returns = [0.001 if index % 2 == 0 else -0.001 for index in range(252)]
    asset_returns = [0.002 if index % 2 == 0 else -0.002 for index in range(252)]
    session = FakeSession([
        FakeResponse(200, _chart_payload("AAPL", asset_returns, price=342.96)),
        FakeResponse(200, _chart_payload("^GSPC", benchmark_returns, price=6800.0)),
    ])
    client = YahooChartClient(session=session, now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

    snapshot = await client.get_snapshot("AAPL")

    assert len(session.calls) == 2
    assert all("/v8/finance/chart/" in url for url, _ in session.calls)
    assert all(call[1]["params"]["range"] == "2y" for call in session.calls)
    assert snapshot.current_price == 342.96
    assert snapshot.history[-1].open == pytest.approx(snapshot.history[-1].close - 1)
    assert snapshot.history[-1].high == pytest.approx(snapshot.history[-1].close + 1)
    assert snapshot.history[-1].low == pytest.approx(snapshot.history[-1].close - 2)
    assert snapshot.history[-1].volume == 1_000_252
    assert snapshot.volatility_252d == pytest.approx(0.03181224033249497)
    assert snapshot.beta_sp500 == pytest.approx(2.0)
    assert snapshot.correlation_sp500 == pytest.approx(1.0)
    assert snapshot.realized_total_return_252d == pytest.approx(-0.0005038740208309633)
    assert snapshot.asset_return_observations == 252
    assert snapshot.aligned_return_observations == 252
    assert snapshot.metric_session_start == "2025-04-23"
    assert snapshot.metric_session_end == "2025-12-31"
    assert snapshot.aligned_metric_session_start == "2025-04-23"
    assert snapshot.aligned_metric_session_end == "2025-12-31"
    assert snapshot.errors == []


@pytest.mark.asyncio
async def test_snapshot_cache_reuses_complete_two_call_operation_across_clients():
    returns = [0.001 for _ in range(252)]
    first_session = FakeSession([
        FakeResponse(200, _chart_payload("AAPL", returns, price=300.0)),
        FakeResponse(200, _chart_payload("^GSPC", returns, price=6800.0)),
    ])
    second_session = FakeSession([])
    yahoo_chart.clear_chart_cache()

    def fixture_now():
        return datetime(2026, 1, 1, tzinfo=timezone.utc)

    async with YahooChartClient(session=first_session, now=fixture_now) as first:
        first_snapshot = await first.get_snapshot("AAPL")
    async with YahooChartClient(session=second_session, now=fixture_now) as second:
        second_snapshot = await second.get_snapshot(" aapl ")

    assert len(first_session.calls) == 2
    assert second_session.calls == []
    assert second_snapshot == first_snapshot


@pytest.mark.asyncio
async def test_cached_snapshot_is_rejected_after_observations_cross_stale_boundary():
    returns = [0.001 for _ in range(252)]
    session = FakeSession([
        FakeResponse(200, _chart_payload("AAPL", returns, price=300.0)),
        FakeResponse(200, _chart_payload("^GSPC", returns, price=6800.0)),
    ])
    clock = [datetime(2026, 1, 7, 20, 59, tzinfo=timezone.utc)]
    client = YahooChartClient(session=session, now=lambda: clock[0])

    await client.get_snapshot("AAPL")
    clock[0] = datetime(2026, 1, 7, 21, 1, tzinfo=timezone.utc)

    with pytest.raises(YahooChartMalformedResponse):
        await client.get_snapshot("AAPL")
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_concurrent_snapshots_coalesce_the_complete_operation():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []
    sentinel = SimpleNamespace(symbol="AAPL")

    class BlockingClient(YahooChartClient):
        async def _get_snapshot_uncached(self, symbol):
            calls.append(symbol)
            started.set()
            await release.wait()
            return sentinel

    yahoo_chart.clear_chart_cache()
    first_client = BlockingClient(session=FakeSession([]))
    second_client = BlockingClient(session=FakeSession([]))
    first = asyncio.create_task(first_client.get_snapshot("AAPL"))
    await started.wait()
    second = asyncio.create_task(second_client.get_snapshot("aapl"))
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == [sentinel, sentinel]
    assert calls == ["AAPL"]


@pytest.mark.asyncio
async def test_distinct_symbol_inflight_registry_is_bounded():
    release = asyncio.Event()
    calls = []

    class BlockingClient(YahooChartClient):
        async def _get_snapshot_uncached(self, symbol):
            calls.append(symbol)
            await release.wait()
            return SimpleNamespace(symbol=symbol)

    client = BlockingClient(session=FakeSession([]))
    tasks = [
        asyncio.create_task(client.get_snapshot(f"S{index}"))
        for index in range(yahoo_chart._MAX_INFLIGHT_SNAPSHOTS)
    ]
    while len(calls) < yahoo_chart._MAX_INFLIGHT_SNAPSHOTS:
        await asyncio.sleep(0)

    with pytest.raises(YahooChartTemporarilyUnavailable):
        await client.get_snapshot("OVERFLOW")

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_snapshot_cache_is_bounded_and_evicts_least_recent_symbol():
    calls = []

    class CountingClient(YahooChartClient):
        async def _get_snapshot_uncached(self, symbol):
            calls.append(symbol)
            return SimpleNamespace(symbol=symbol)

    client = CountingClient(session=FakeSession([]))
    for index in range(129):
        await client.get_snapshot(f"S{index}")
    await client.get_snapshot("S0")

    assert len(calls) == 130
    assert calls[-1] == "S0"


@pytest.mark.asyncio
async def test_clearing_cache_during_inflight_request_cannot_restore_old_result():
    old_release = asyncio.Event()
    new_release = asyncio.Event()
    calls = []
    timestamp = int(datetime(2025, 12, 31, 21, tzinfo=timezone.utc).timestamp())
    old_result = SimpleNamespace(
        version="old",
        current_price_as_of="2025-12-31T21:00:00+00:00",
        history=[SimpleNamespace(timestamp=timestamp)],
    )
    new_result = SimpleNamespace(
        version="new",
        current_price_as_of="2025-12-31T21:00:00+00:00",
        history=[SimpleNamespace(timestamp=timestamp)],
    )

    class VersionedClient(YahooChartClient):
        async def _get_snapshot_uncached(self, symbol):
            calls.append(symbol)
            if len(calls) == 1:
                await old_release.wait()
                return old_result
            await new_release.wait()
            return new_result

    client = VersionedClient(
        session=FakeSession([]),
        now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    old_task = asyncio.create_task(client.get_snapshot("AAPL"))
    await asyncio.sleep(0)
    yahoo_chart.clear_chart_cache()
    new_task = asyncio.create_task(client.get_snapshot("AAPL"))
    await asyncio.sleep(0)
    new_release.set()
    assert await new_task is new_result
    old_release.set()
    assert await old_task is old_result

    assert await client.get_snapshot("AAPL") is new_result
    assert calls == ["AAPL", "AAPL"]


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_close_session_before_shared_work_finishes():
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class CancellationClient(YahooChartClient):
        async def _get_snapshot_uncached(self, symbol):
            started.set()
            await release.wait()
            return SimpleNamespace(symbol=symbol)

        async def close(self):
            closed.set()

    client = CancellationClient(session=FakeSession([]))

    async def request_scope():
        async with client:
            return await client.get_snapshot("AAPL")

    waiter = asyncio.create_task(request_scope())
    await started.wait()
    waiter.cancel()
    await asyncio.sleep(0)
    assert not closed.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert closed.is_set()


@pytest.mark.asyncio
async def test_benchmark_failure_preserves_independent_asset_metrics():
    asset_returns = [0.001 for _ in range(252)]
    session = FakeSession([
        FakeResponse(200, _chart_payload("AAPL", asset_returns, price=300.0)),
        FakeResponse(503, {"chart": {"result": None, "error": {"code": "Unavailable"}}}),
    ])
    client = YahooChartClient(session=session, now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

    snapshot = await client.get_snapshot("AAPL")

    assert snapshot.status == "partial"
    assert snapshot.current_price == 300.0
    assert snapshot.realized_total_return_252d is not None
    assert snapshot.beta_sp500 is None
    assert snapshot.correlation_sp500 is None
    assert snapshot.errors == ["benchmark_temporarily_unavailable"]


@pytest.mark.asyncio
async def test_malformed_parallel_arrays_fail_closed():
    payload = _chart_payload("AAPL", [0.01 for _ in range(252)])
    payload["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"].pop()
    session = FakeSession([FakeResponse(200, payload)])
    client = YahooChartClient(session=session)

    with pytest.raises(YahooChartMalformedResponse):
        await client.get_snapshot("AAPL")


def test_non_finite_values_are_not_accepted():
    payload = _chart_payload("AAPL", [0.01 for _ in range(252)])
    payload["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][10] = math.inf
    session = FakeSession([FakeResponse(200, payload)])
    client = YahooChartClient(session=session)

    with pytest.raises(YahooChartMalformedResponse):
        client._fetch_chart("AAPL")


@pytest.mark.parametrize("failure", ["symbol", "stale", "future", "order", "types"])
def test_untrusted_chart_identity_time_order_and_types_fail_closed(failure):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload = _chart_payload("AAPL", [0.01 for _ in range(252)])
    result = payload["chart"]["result"][0]
    if failure == "symbol":
        result["meta"]["symbol"] = "MSFT"
    elif failure == "stale":
        result["meta"]["regularMarketTime"] = 1
    elif failure == "future":
        result["meta"]["regularMarketTime"] = int(
            (now + timedelta(days=2)).timestamp()
        )
    elif failure == "order":
        result["timestamp"][10], result["timestamp"][11] = (
            result["timestamp"][11], result["timestamp"][10]
        )
    else:
        result["indicators"]["quote"][0]["open"][10] = "123.0"
        result["indicators"]["quote"][0]["volume"][10] = True

    client = YahooChartClient(
        session=FakeSession([FakeResponse(200, payload)]), now=lambda: now
    )

    with pytest.raises(YahooChartMalformedResponse):
        client._fetch_chart("AAPL")


def test_price_history_endpoint_uses_narrow_chart_snapshot(monkeypatch):
    class WorkingChartClient:
        calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def get_snapshot(self, ticker):
            self.calls.append(ticker)
            return SimpleNamespace(
                history=[
                    YahooPricePoint(
                        session_date="2026-10-13",
                        timestamp=1_760_396_400,
                        open=198.0,
                        high=202.0,
                        low=197.0,
                        close=200.0,
                        adjusted_close=200.0,
                        volume=1_000_000,
                    )
                ]
            )

    monkeypatch.setattr(main, "YahooChartClient", WorkingChartClient)

    with TestClient(main.app) as client:
        response = client.get("/api/price-history/AAPL?period=2y")

    assert response.status_code == 200
    assert response.json() == {
        "ticker": "AAPL",
        "period": "2y",
        "prices": [
            {
                "date": "2026-10-13",
                "open": 198.0,
                "high": 202.0,
                "low": 197.0,
                "close": 200.0,
                "volume": 1_000_000,
            }
        ],
        "earnings_markers": [],
        "next_earnings_date": None,
    }
    assert WorkingChartClient.calls == ["AAPL"]


def test_price_history_endpoint_applies_requested_window(monkeypatch):
    class WorkingChartClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def get_snapshot(self, ticker):
            def point(session_date, timestamp):
                return YahooPricePoint(
                    session_date=session_date,
                    timestamp=timestamp,
                    open=198.0,
                    high=202.0,
                    low=197.0,
                    close=200.0,
                    adjusted_close=200.0,
                    volume=1_000_000,
                )

            return SimpleNamespace(
                history=[
                    point("2026-08-01", 1_754_006_400),
                    point("2026-10-01", 1_759_276_800),
                    point("2026-10-13", 1_760_313_600),
                ]
            )

    monkeypatch.setattr(main, "YahooChartClient", WorkingChartClient)

    with TestClient(main.app) as client:
        response = client.get("/api/price-history/AAPL?period=1mo")
        invalid = client.get("/api/price-history/AAPL?period=max")

    assert response.status_code == 200
    assert [point["date"] for point in response.json()["prices"]] == [
        "2026-10-01",
        "2026-10-13",
    ]
    assert invalid.status_code == 422
