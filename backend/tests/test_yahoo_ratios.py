"""Contracts for narrow Yahoo normalized-ratio enrichment."""

import asyncio
from datetime import datetime, timezone
import math
import threading
import time

import pytest

from data_sources.yahoo_ratios import (
    YahooRatioMalformedResponse,
    YahooRatioTemporarilyUnavailable,
    YahooRatiosClient,
    clear_ratio_cache,
)


NOW = datetime(2026, 10, 15, 18, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def empty_ratio_cache():
    clear_ratio_cache()
    yield
    clear_ratio_cache()


class FakeTicker:
    def __init__(self, info):
        self._info = info

    def get_info(self):
        return self._info


def client_for(info):
    return YahooRatiosClient(
        ticker_factory=lambda symbol: FakeTicker(info),
        now=lambda: NOW,
    )


@pytest.mark.asyncio
async def test_snapshot_exposes_only_allowlisted_normalized_ratios():
    snapshot = await client_for(
        {
            "symbol": "AMZN",
            "trailingPE": 20.75,
            "priceToBook": 4.95,
            "currentRatio": 1.033,
            "trailingAnnualDividendYield": 0.0,
            "returnOnEquity": 0.30558,
            "profitMargins": 0.1744,
            "grossMargins": 0.5077,
            "operatingMargins": 0.13689,
            "debtToEquity": 45.617,
            "totalRevenue": 999999999,
            "earningsGrowth": 2.423,
        }
    ).get_snapshot("AMZN")

    assert snapshot.symbol == "AMZN"
    assert snapshot.fetched_at == "2026-10-15T18:00:00+00:00"
    assert snapshot.values == pytest.approx(
        {
            "pe_ratio": 20.75,
            "pb_ratio": 4.95,
            "current_ratio": 1.033,
            "dividend_yield": 0.0,
            "roe": 0.30558,
            "profit_margin": 0.1744,
            "gross_margin": 0.5077,
            "operating_margin": 0.13689,
            "debt_to_equity": 0.45617,
        }
    )
    assert snapshot.provider_fields == {
        "pe_ratio": "trailingPE",
        "pb_ratio": "priceToBook",
        "current_ratio": "currentRatio",
        "dividend_yield": "trailingAnnualDividendYield",
        "roe": "returnOnEquity",
        "profit_margin": "profitMargins",
        "gross_margin": "grossMargins",
        "operating_margin": "operatingMargins",
        "debt_to_equity": "debtToEquity",
    }
    assert "earnings_growth" not in snapshot.values
    assert "total_revenue" not in snapshot.values


@pytest.mark.asyncio
async def test_missing_fields_stay_missing_while_explicit_zero_is_preserved():
    snapshot = await client_for(
        {"symbol": "AMZN", "trailingAnnualDividendYield": 0}
    ).get_snapshot("AMZN")

    assert snapshot.values == {"dividend_yield": 0.0}
    assert "gross_margin" not in snapshot.values


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [True, "0.4", math.nan, math.inf, -math.inf])
async def test_invalid_ratio_values_fail_closed(invalid):
    snapshot = await client_for(
        {"symbol": "AMZN", "grossMargins": invalid}
    ).get_snapshot("AMZN")

    assert "gross_margin" not in snapshot.values


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_field", "value", "metric"),
    [
        ("trailingPE", 0.0, "pe_ratio"),
        ("trailingPE", -5.0, "pe_ratio"),
        ("priceToBook", 0.0, "pb_ratio"),
        ("priceToBook", -2.0, "pb_ratio"),
        ("currentRatio", -1.0, "current_ratio"),
        ("trailingAnnualDividendYield", -0.01, "dividend_yield"),
        ("debtToEquity", -50.0, "debt_to_equity"),
    ],
)
async def test_financially_invalid_ratio_domains_are_omitted(
    provider_field, value, metric
):
    snapshot = await client_for(
        {"symbol": "AMZN", provider_field: value, "profitMargins": -0.2}
    ).get_snapshot("AMZN")

    assert metric not in snapshot.values
    assert snapshot.values["profit_margin"] == -0.2


@pytest.mark.asyncio
async def test_valid_zero_and_negative_profitability_ratios_are_preserved():
    snapshot = await client_for(
        {
            "symbol": "AMZN",
            "currentRatio": 0.0,
            "trailingAnnualDividendYield": 0.0,
            "debtToEquity": 0.0,
            "returnOnEquity": -0.1,
            "profitMargins": -0.2,
            "grossMargins": -0.3,
            "operatingMargins": -0.4,
        }
    ).get_snapshot("AMZN")

    assert snapshot.values == {
        "current_ratio": 0.0,
        "dividend_yield": 0.0,
        "roe": -0.1,
        "profit_margin": -0.2,
        "gross_margin": -0.3,
        "operating_margin": -0.4,
        "debt_to_equity": 0.0,
    }


@pytest.mark.asyncio
async def test_repeated_snapshot_uses_bounded_process_cache():
    calls = []

    def factory(symbol):
        calls.append(symbol)
        return FakeTicker({"symbol": symbol, "grossMargins": 0.5})

    client = YahooRatiosClient(ticker_factory=factory, now=lambda: NOW)
    first = await client.get_snapshot("AMZN")
    second = await client.get_snapshot("AMZN")

    assert first is second
    assert calls == ["AMZN"]


@pytest.mark.asyncio
async def test_concurrent_snapshots_coalesce_one_provider_call():
    calls = []

    class SlowTicker(FakeTicker):
        def get_info(self):
            time.sleep(0.02)
            return super().get_info()

    def factory(symbol):
        calls.append(symbol)
        return SlowTicker({"symbol": symbol, "grossMargins": 0.5})

    client = YahooRatiosClient(ticker_factory=factory, now=lambda: NOW)
    first, second = await asyncio.gather(
        client.get_snapshot("AMZN"),
        client.get_snapshot("AMZN"),
    )

    assert first is second
    assert calls == ["AMZN"]


@pytest.mark.asyncio
async def test_hung_provider_is_bounded_by_fetch_deadline():
    class HungTicker(FakeTicker):
        def get_info(self):
            time.sleep(0.05)
            return super().get_info()

    client = YahooRatiosClient(
        ticker_factory=lambda symbol: HungTicker({"symbol": symbol}),
        now=lambda: NOW,
        fetch_timeout_seconds=0.005,
    )

    with pytest.raises(YahooRatioTemporarilyUnavailable):
        await client.get_snapshot("AMZN")


@pytest.mark.asyncio
async def test_timed_out_provider_work_remains_coalesced_until_thread_finishes():
    release = threading.Event()
    calls = []

    class BlockedTicker(FakeTicker):
        def get_info(self):
            release.wait(timeout=1)
            return super().get_info()

    def factory(symbol):
        calls.append(symbol)
        return BlockedTicker({"symbol": symbol, "grossMargins": 0.5})

    client = YahooRatiosClient(
        ticker_factory=factory,
        now=lambda: NOW,
        fetch_timeout_seconds=0.005,
    )

    with pytest.raises(YahooRatioTemporarilyUnavailable):
        await client.get_snapshot("AMZN")
    with pytest.raises(YahooRatioTemporarilyUnavailable):
        await client.get_snapshot("AMZN")
    assert calls == ["AMZN"]

    release.set()
    await asyncio.sleep(0.02)
    assert (await client.get_snapshot("AMZN")).values == {"gross_margin": 0.5}
    assert calls == ["AMZN"]


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_fetch():
    class SlowTicker(FakeTicker):
        def get_info(self):
            time.sleep(0.02)
            return super().get_info()

    client = YahooRatiosClient(
        ticker_factory=lambda symbol: SlowTicker(
            {"symbol": symbol, "grossMargins": 0.5}
        ),
        now=lambda: NOW,
    )
    first = asyncio.create_task(client.get_snapshot("AMZN"))
    second = asyncio.create_task(client.get_snapshot("AMZN"))
    await asyncio.sleep(0)
    first.cancel()

    with pytest.raises(asyncio.CancelledError):
        await first
    assert (await second).values == {"gross_margin": 0.5}


@pytest.mark.asyncio
async def test_symbol_identity_mismatch_is_rejected():
    with pytest.raises(YahooRatioMalformedResponse):
        await client_for({"symbol": "AAPL", "grossMargins": 0.5}).get_snapshot("AMZN")
