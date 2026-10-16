"""Narrow Yahoo chart adapter for timestamped prices and risk analytics."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import math
import re
import statistics
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from curl_cffi import requests as curl_requests


class YahooChartError(Exception):
    """Base class for sanitized Yahoo chart failures."""

    code = "provider_failure"
    retryable = True


class YahooChartInvalidSymbol(YahooChartError):
    code = "invalid_symbol"
    retryable = False


class YahooChartNotFound(YahooChartError):
    code = "ticker_not_found"
    retryable = False


class YahooChartTemporarilyUnavailable(YahooChartError):
    code = "provider_temporarily_unavailable"
    retryable = True


class YahooChartMalformedResponse(YahooChartError):
    code = "malformed_response"
    retryable = True


@dataclass(frozen=True)
class YahooPricePoint:
    session_date: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int


@dataclass(frozen=True)
class YahooSplitEvent:
    effective_date: str
    timestamp: int
    numerator: float
    denominator: float


@dataclass(frozen=True)
class YahooChartSnapshot:
    symbol: str
    currency: str
    fetched_at: str
    current_price: float
    current_price_as_of: str
    history: List[YahooPricePoint]
    volatility_252d: Optional[float]
    beta_sp500: Optional[float]
    correlation_sp500: Optional[float]
    realized_total_return_252d: Optional[float]
    asset_return_observations: int
    aligned_return_observations: int
    metric_session_start: Optional[str]
    metric_session_end: Optional[str]
    aligned_metric_session_start: Optional[str]
    aligned_metric_session_end: Optional[str]
    cache_age_seconds: float = 0.0
    benchmark_history: List[YahooPricePoint] = field(default_factory=list)
    split_events: List[YahooSplitEvent] = field(default_factory=list)
    status: str = "complete"
    errors: List[str] = field(default_factory=list)


_CACHE_TTL_SECONDS = 300.0
_MAX_CACHE_ENTRIES = 128
_MAX_INFLIGHT_SNAPSHOTS = 32
_SNAPSHOT_DEADLINE_SECONDS = 18.0
_SNAPSHOT_CACHE: OrderedDict[str, Tuple[float, YahooChartSnapshot]] = OrderedDict()
_INFLIGHT_SNAPSHOTS: Dict[Tuple[int, str], asyncio.Task] = {}
_INFLIGHT_ASSETS: Dict[Tuple[int, str], Dict[str, Any]] = {}
_SNAPSHOT_CACHE_LOCK = threading.Lock()
_CACHE_GENERATION = 0


def clear_chart_cache() -> None:
    """Clear process-local snapshots for deterministic tests and maintenance."""
    global _CACHE_GENERATION
    with _SNAPSHOT_CACHE_LOCK:
        _CACHE_GENERATION += 1
        _SNAPSHOT_CACHE.clear()
        _INFLIGHT_SNAPSHOTS.clear()
        _INFLIGHT_ASSETS.clear()


class YahooChartClient:
    """Fetch at most one asset chart and one S&P 500 chart per snapshot."""

    _URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    _SYMBOL = re.compile(r"^[A-Z0-9.^=-]{1,20}$")
    _MAX_MARKET_AGE = timedelta(days=7)
    _MAX_FUTURE_SKEW = timedelta(minutes=5)

    def __init__(
        self,
        session: Optional[Any] = None,
        now: Optional[Callable[[], datetime]] = None,
    ):
        self._session = session or curl_requests.Session()
        self._owns_session = session is None
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._snapshot_tasks = set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def close(self):
        if self._owns_session:
            pending = [task for task in self._snapshot_tasks if not task.done()]
            if pending:
                session = self._session
                self._owns_session = False

                async def close_after_provider_work():
                    await asyncio.gather(*pending, return_exceptions=True)
                    await asyncio.to_thread(session.close)

                asyncio.create_task(close_after_provider_work())
            else:
                await asyncio.to_thread(self._session.close)

    def _normalized_symbol(self, symbol: str) -> str:
        normalized = symbol.strip().upper()
        if not self._SYMBOL.fullmatch(normalized):
            raise YahooChartInvalidSymbol("The market symbol is invalid.")
        return normalized

    def _validate_observation_times(
        self, quote_time: datetime, latest_session: datetime
    ) -> None:
        try:
            now = self._now()
            if now.tzinfo is None:
                raise ValueError
            now_utc = now.astimezone(timezone.utc)
            for observed_at in (quote_time, latest_session):
                age = now_utc - observed_at
                if age > self._MAX_MARKET_AGE or age < -self._MAX_FUTURE_SKEW:
                    raise ValueError
        except (OSError, OverflowError, TypeError, ValueError):
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            ) from None

    def _fetch_chart(self, symbol: str) -> Dict[str, Any]:
        symbol = self._normalized_symbol(symbol)
        try:
            response = self._session.get(
                self._URL.format(symbol=quote(symbol, safe="")),
                params={
                    "range": "2y",
                    "interval": "1d",
                    "events": "div,splits",
                    "includePrePost": "false",
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
        except Exception:
            raise YahooChartTemporarilyUnavailable(
                "Market data is temporarily unavailable."
            ) from None

        status = getattr(response, "status_code", None)
        if status == 404:
            raise YahooChartNotFound("The market symbol was not found.")
        if status in {401, 403, 408, 425, 429} or (
            isinstance(status, int) and status >= 500
        ):
            raise YahooChartTemporarilyUnavailable(
                "Market data is temporarily unavailable."
            )
        if status != 200:
            raise YahooChartError("Market data request failed.")

        try:
            payload = response.json()
            chart = payload["chart"]
            if chart.get("error"):
                code = str(chart["error"].get("code", ""))
                if code.lower() in {"not found", "not_found"}:
                    raise YahooChartNotFound("The market symbol was not found.")
                raise YahooChartTemporarilyUnavailable(
                    "Market data is temporarily unavailable."
                )
            result = chart["result"][0]
            meta = result["meta"]
            timestamps = result["timestamp"]
            quote_data = result["indicators"]["quote"][0]
            opens = quote_data["open"]
            highs = quote_data["high"]
            lows = quote_data["low"]
            closes = quote_data["close"]
            volumes = quote_data["volume"]
            adjusted = result["indicators"]["adjclose"][0]["adjclose"]
            raw_splits = result.get("events", {}).get("splits", {})
        except YahooChartError:
            raise
        except (IndexError, KeyError, TypeError, ValueError):
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            ) from None

        if not timestamps or not (
            len(timestamps)
            == len(opens)
            == len(highs)
            == len(lows)
            == len(closes)
            == len(volumes)
            == len(adjusted)
        ):
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            )
        if str(meta.get("symbol") or "").strip().upper() != symbol:
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            )
        if any(type(timestamp) is not int for timestamp in timestamps) or any(
            later <= earlier for earlier, later in zip(timestamps, timestamps[1:])
        ):
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            )

        if not isinstance(raw_splits, dict):
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            )
        split_events = []
        try:
            for event in raw_splits.values():
                timestamp = event["date"]
                numerator = event["numerator"]
                denominator = event["denominator"]
                if (
                    type(timestamp) is not int
                    or type(numerator) not in (int, float)
                    or type(denominator) not in (int, float)
                ):
                    raise ValueError
                numerator = float(numerator)
                denominator = float(denominator)
                if (
                    not math.isfinite(numerator)
                    or not math.isfinite(denominator)
                    or numerator <= 0
                    or denominator <= 0
                ):
                    raise ValueError
                split_events.append(
                    YahooSplitEvent(
                        effective_date=datetime.fromtimestamp(
                            timestamp, timezone.utc
                        ).date().isoformat(),
                        timestamp=timestamp,
                        numerator=numerator,
                        denominator=denominator,
                    )
                )
        except (KeyError, OSError, OverflowError, TypeError, ValueError):
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            ) from None
        split_events.sort(key=lambda item: item.timestamp)
        if len({item.timestamp for item in split_events}) != len(split_events):
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            )

        points: List[YahooPricePoint] = []
        try:
            for timestamp, open_value, high, low, close, adjusted_close, volume in zip(
                timestamps, opens, highs, lows, closes, adjusted, volumes
            ):
                if (
                    type(open_value) not in (int, float)
                    or type(high) not in (int, float)
                    or type(low) not in (int, float)
                    or type(close) not in (int, float)
                    or type(adjusted_close) not in (int, float)
                    or type(volume) is not int
                ):
                    raise ValueError
                open_value = float(open_value)
                high = float(high)
                low = float(low)
                close = float(close)
                adjusted_close = float(adjusted_close)
                volume = int(volume)
                if (
                    not math.isfinite(open_value)
                    or not math.isfinite(high)
                    or not math.isfinite(low)
                    or not math.isfinite(close)
                    or not math.isfinite(adjusted_close)
                    or open_value <= 0
                    or high <= 0
                    or low <= 0
                    or close <= 0
                    or adjusted_close <= 0
                    or high < low
                    or volume < 0
                ):
                    raise ValueError
                session_date = datetime.fromtimestamp(
                    timestamp, timezone.utc
                ).date().isoformat()
                points.append(
                    YahooPricePoint(
                        session_date=session_date,
                        timestamp=timestamp,
                        open=open_value,
                        high=high,
                        low=low,
                        close=close,
                        adjusted_close=adjusted_close,
                        volume=volume,
                    )
                )
            if (
                type(meta["regularMarketPrice"]) not in (int, float)
                or type(meta["regularMarketTime"]) is not int
            ):
                raise ValueError
            current_price = float(meta["regularMarketPrice"])
            quote_timestamp = meta["regularMarketTime"]
            if not math.isfinite(current_price) or current_price <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            ) from None

        if len({point.session_date for point in points}) != len(points):
            raise YahooChartMalformedResponse(
                "Market data returned duplicate sessions."
            )

        try:
            quote_time = datetime.fromtimestamp(quote_timestamp, timezone.utc)
            latest_session = datetime.fromtimestamp(points[-1].timestamp, timezone.utc)
        except (OSError, OverflowError, TypeError, ValueError):
            raise YahooChartMalformedResponse(
                "Market data returned an invalid response."
            ) from None
        self._validate_observation_times(quote_time, latest_session)

        return {
            "symbol": symbol,
            "currency": str(meta.get("currency") or ""),
            "current_price": current_price,
            "current_price_as_of": datetime.fromtimestamp(
                quote_timestamp, timezone.utc
            ).isoformat(),
            "points": points,
            "split_events": split_events,
        }

    @staticmethod
    def _returns(points: List[YahooPricePoint], *, count: int = 252):
        if len(points) < count + 1:
            return [], [], []
        window = points[-(count + 1):]
        log_returns = []
        simple_returns = []
        intervals = []
        for previous, current in zip(window, window[1:]):
            log_returns.append(
                math.log(current.adjusted_close / previous.adjusted_close)
            )
            simple_returns.append(
                current.adjusted_close / previous.adjusted_close - 1
            )
            intervals.append((previous.session_date, current.session_date))
        return log_returns, simple_returns, intervals

    @staticmethod
    def _aligned_market_metrics(
        asset_points: List[YahooPricePoint],
        benchmark_points: List[YahooPricePoint],
    ) -> Tuple[
        Optional[float], Optional[float], int, Optional[str], Optional[str]
    ]:
        _, asset_returns, asset_intervals = YahooChartClient._returns(asset_points)
        _, market_returns, market_intervals = YahooChartClient._returns(
            benchmark_points
        )
        asset = dict(zip(asset_intervals, asset_returns))
        market = dict(zip(market_intervals, market_returns))
        shared = sorted(set(asset) & set(market))
        if len(shared) < 126:
            return None, None, len(shared), None, None
        shared = shared[-252:]
        session_start = shared[0][0]
        session_end = shared[-1][1]
        asset_values = [asset[key] for key in shared]
        market_values = [market[key] for key in shared]
        market_variance = statistics.variance(market_values)
        asset_std = statistics.stdev(asset_values)
        market_std = statistics.stdev(market_values)
        if market_variance == 0 or asset_std == 0 or market_std == 0:
            return None, None, len(shared), session_start, session_end
        asset_mean = statistics.mean(asset_values)
        market_mean = statistics.mean(market_values)
        covariance = sum(
            (asset_value - asset_mean) * (market_value - market_mean)
            for asset_value, market_value in zip(asset_values, market_values)
        ) / (len(shared) - 1)
        beta = covariance / market_variance
        correlation = covariance / (asset_std * market_std)
        if not math.isfinite(beta) or not math.isfinite(correlation):
            return None, None, len(shared), session_start, session_end
        return beta, correlation, len(shared), session_start, session_end

    async def get_snapshot(self, symbol: str) -> YahooChartSnapshot:
        symbol = self._normalized_symbol(symbol)
        now = time.monotonic()
        with _SNAPSHOT_CACHE_LOCK:
            cached = _SNAPSHOT_CACHE.get(symbol)
            if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
                try:
                    self._validate_observation_times(
                        datetime.fromisoformat(cached[1].current_price_as_of),
                        datetime.fromtimestamp(
                            cached[1].history[-1].timestamp, timezone.utc
                        ),
                    )
                except (IndexError, ValueError, YahooChartMalformedResponse):
                    _SNAPSHOT_CACHE.pop(symbol, None)
                    raise YahooChartMalformedResponse(
                        "Market data returned an invalid response."
                    ) from None
                _SNAPSHOT_CACHE.move_to_end(symbol)
                if isinstance(cached[1], YahooChartSnapshot):
                    return replace(
                        cached[1], cache_age_seconds=max(0.0, now - cached[0])
                    )
                return cached[1]
            loop = asyncio.get_running_loop()
            inflight_key = (id(loop), symbol)
            task = _INFLIGHT_SNAPSHOTS.get(inflight_key)
            if task is None:
                if len(_INFLIGHT_SNAPSHOTS) >= _MAX_INFLIGHT_SNAPSHOTS:
                    raise YahooChartTemporarilyUnavailable(
                        "Market data is temporarily unavailable."
                    )
                generation = _CACHE_GENERATION
                task = loop.create_task(self._get_snapshot_uncached(symbol))
                self._snapshot_tasks.add(task)
                _INFLIGHT_SNAPSHOTS[inflight_key] = task

                def complete(done_task):
                    self._snapshot_tasks.discard(done_task)
                    try:
                        result = done_task.result()
                    except (BaseException, asyncio.CancelledError):
                        result = None
                    with _SNAPSHOT_CACHE_LOCK:
                        is_current = (
                            _INFLIGHT_SNAPSHOTS.get(inflight_key) is done_task
                            and _CACHE_GENERATION == generation
                        )
                        if is_current:
                            _INFLIGHT_SNAPSHOTS.pop(inflight_key, None)
                            _INFLIGHT_ASSETS.pop(inflight_key, None)
                        if is_current and result is not None:
                            _SNAPSHOT_CACHE[symbol] = (time.monotonic(), result)
                            _SNAPSHOT_CACHE.move_to_end(symbol)
                            while len(_SNAPSHOT_CACHE) > _MAX_CACHE_ENTRIES:
                                _SNAPSHOT_CACHE.popitem(last=False)

                task.add_done_callback(complete)

        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=_SNAPSHOT_DEADLINE_SECONDS,
            )
        except asyncio.TimeoutError:
            with _SNAPSHOT_CACHE_LOCK:
                asset = _INFLIGHT_ASSETS.get(inflight_key)
            if asset is not None:
                return self._build_snapshot(
                    asset, None, ["benchmark_deadline_exceeded"]
                )
            raise YahooChartTemporarilyUnavailable(
                "Market data is temporarily unavailable."
            ) from None
        except asyncio.CancelledError:
            try:
                await asyncio.shield(task)
            except BaseException:
                pass
            raise

    async def _get_snapshot_uncached(self, symbol: str) -> YahooChartSnapshot:
        started = time.monotonic()
        asset = await asyncio.to_thread(self._fetch_chart, symbol)
        inflight_key = (id(asyncio.get_running_loop()), symbol)
        with _SNAPSHOT_CACHE_LOCK:
            if _INFLIGHT_SNAPSHOTS.get(inflight_key) is asyncio.current_task():
                _INFLIGHT_ASSETS[inflight_key] = asset

        errors: List[str] = []
        remaining = _SNAPSHOT_DEADLINE_SECONDS - (time.monotonic() - started)
        scheduling_reserve = min(0.05, _SNAPSHOT_DEADLINE_SECONDS * 0.2)
        if remaining <= scheduling_reserve:
            benchmark = None
            errors.append("benchmark_deadline_exceeded")
        else:
            benchmark_task = asyncio.create_task(
                asyncio.to_thread(self._fetch_chart, "^GSPC")
            )
            self._snapshot_tasks.add(benchmark_task)

            def benchmark_complete(done_task):
                self._snapshot_tasks.discard(done_task)
                if not done_task.cancelled():
                    done_task.exception()

            benchmark_task.add_done_callback(benchmark_complete)
            done, _ = await asyncio.wait(
                {benchmark_task}, timeout=remaining - scheduling_reserve
            )
            if not done:
                benchmark = None
                errors.append("benchmark_deadline_exceeded")
            else:
                try:
                    benchmark = benchmark_task.result()
                except YahooChartError:
                    benchmark = None
                    errors.append("benchmark_temporarily_unavailable")
        return self._build_snapshot(asset, benchmark, errors)

    def _build_snapshot(self, asset, benchmark, errors) -> YahooChartSnapshot:
        asset_logs, _, _ = self._returns(asset["points"])
        volatility = (
            statistics.stdev(asset_logs) * math.sqrt(252)
            if len(asset_logs) == 252
            else None
        )
        realized_return = (
            asset["points"][-1].adjusted_close
            / asset["points"][-253].adjusted_close
            - 1
            if len(asset_logs) == 252
            else None
        )

        if benchmark is None:
            beta = None
            correlation = None
            aligned_count = 0
            aligned_start = None
            aligned_end = None
        else:
            beta, correlation, aligned_count, aligned_start, aligned_end = (
                self._aligned_market_metrics(
                asset["points"], benchmark["points"]
            )
            )
            if beta is None or correlation is None:
                errors.append("benchmark_metrics_unavailable")

        return YahooChartSnapshot(
            symbol=asset["symbol"],
            currency=asset["currency"],
            fetched_at=self._now().astimezone(timezone.utc).isoformat(),
            current_price=asset["current_price"],
            current_price_as_of=asset["current_price_as_of"],
            history=asset["points"],
            volatility_252d=volatility,
            beta_sp500=beta,
            correlation_sp500=correlation,
            realized_total_return_252d=realized_return,
            asset_return_observations=len(asset_logs),
            aligned_return_observations=aligned_count,
            metric_session_start=(
                asset["points"][-253].session_date if len(asset_logs) == 252 else None
            ),
            metric_session_end=(
                asset["points"][-1].session_date if asset["points"] else None
            ),
            aligned_metric_session_start=aligned_start,
            aligned_metric_session_end=aligned_end,
            benchmark_history=(benchmark["points"] if benchmark is not None else []),
            split_events=asset["split_events"],
            status="partial" if errors else "complete",
            errors=errors,
        )
