"""Narrow Yahoo adapter for provider-normalized financial ratios."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import yfinance as yf


class YahooRatioError(Exception):
    """Base class for sanitized Yahoo ratio failures."""

    code = "provider_failure"
    retryable = True


class YahooRatioInvalidSymbol(YahooRatioError):
    code = "invalid_symbol"
    retryable = False


class YahooRatioTemporarilyUnavailable(YahooRatioError):
    code = "provider_temporarily_unavailable"
    retryable = True


class YahooRatioMalformedResponse(YahooRatioError):
    code = "malformed_response"
    retryable = True


@dataclass(frozen=True)
class YahooRatioSnapshot:
    symbol: str
    fetched_at: str
    values: Dict[str, float]
    provider_fields: Dict[str, str]


_CACHE_TTL_SECONDS = 300.0
_FETCH_TIMEOUT_SECONDS = 8.0
_MAX_CACHE_ENTRIES = 128
_MAX_INFLIGHT_SNAPSHOTS = 32
_RATIO_CACHE: OrderedDict[str, Tuple[float, YahooRatioSnapshot]] = OrderedDict()
_INFLIGHT_SNAPSHOTS: Dict[Tuple[int, str], asyncio.Task] = {}
_RATIO_CACHE_LOCK = threading.Lock()
_CACHE_GENERATION = 0


def clear_ratio_cache() -> None:
    """Clear process-local ratio snapshots for deterministic tests."""
    global _CACHE_GENERATION
    with _RATIO_CACHE_LOCK:
        _CACHE_GENERATION += 1
        _RATIO_CACHE.clear()
        _INFLIGHT_SNAPSHOTS.clear()


_FIELD_MAP = {
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


class YahooRatiosClient:
    """Fetch only the normalized Yahoo ratios consumed by GOATlens agents."""

    _SYMBOL = re.compile(r"^[A-Z0-9.^=-]{1,20}$")

    def __init__(
        self,
        ticker_factory: Optional[Callable[[str], Any]] = None,
        now: Optional[Callable[[], datetime]] = None,
        fetch_timeout_seconds: float = _FETCH_TIMEOUT_SECONDS,
    ):
        self._ticker_factory = ticker_factory or yf.Ticker
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._fetch_timeout_seconds = fetch_timeout_seconds

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    @classmethod
    def _normalized_symbol(cls, symbol: str) -> str:
        normalized = symbol.strip().upper()
        if not cls._SYMBOL.fullmatch(normalized):
            raise YahooRatioInvalidSymbol("The ratio symbol is invalid.")
        return normalized

    @staticmethod
    def _finite_number(value: Any) -> Optional[float]:
        if type(value) not in (int, float):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _fetch_info(self, symbol: str) -> Dict[str, Any]:
        try:
            info = self._ticker_factory(symbol).get_info()
        except Exception:
            raise YahooRatioTemporarilyUnavailable(
                "Normalized ratios are temporarily unavailable."
            ) from None
        if not isinstance(info, dict):
            raise YahooRatioMalformedResponse(
                "Normalized ratios returned an invalid response."
            )
        return info

    async def get_snapshot(self, symbol: str) -> YahooRatioSnapshot:
        symbol = self._normalized_symbol(symbol)
        now = time.monotonic()
        with _RATIO_CACHE_LOCK:
            cached = _RATIO_CACHE.get(symbol)
            if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
                _RATIO_CACHE.move_to_end(symbol)
                return cached[1]

            loop = asyncio.get_running_loop()
            inflight_key = (id(loop), symbol)
            task = _INFLIGHT_SNAPSHOTS.get(inflight_key)
            if task is None:
                if len(_INFLIGHT_SNAPSHOTS) >= _MAX_INFLIGHT_SNAPSHOTS:
                    raise YahooRatioTemporarilyUnavailable(
                        "Normalized ratios are temporarily unavailable."
                    )
                generation = _CACHE_GENERATION
                task = loop.create_task(self._get_snapshot_uncached(symbol))
                _INFLIGHT_SNAPSHOTS[inflight_key] = task

                def complete(done_task):
                    try:
                        result = done_task.result()
                    except BaseException:
                        result = None
                    with _RATIO_CACHE_LOCK:
                        is_current = (
                            _INFLIGHT_SNAPSHOTS.get(inflight_key) is done_task
                            and _CACHE_GENERATION == generation
                        )
                        if is_current:
                            _INFLIGHT_SNAPSHOTS.pop(inflight_key, None)
                        if is_current and result is not None:
                            _RATIO_CACHE[symbol] = (time.monotonic(), result)
                            _RATIO_CACHE.move_to_end(symbol)
                            while len(_RATIO_CACHE) > _MAX_CACHE_ENTRIES:
                                _RATIO_CACHE.popitem(last=False)

                task.add_done_callback(complete)

        return await asyncio.shield(task)

    async def _get_snapshot_uncached(self, symbol: str) -> YahooRatioSnapshot:
        try:
            info = await asyncio.wait_for(
                asyncio.to_thread(self._fetch_info, symbol),
                timeout=self._fetch_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise YahooRatioTemporarilyUnavailable(
                "Normalized ratios are temporarily unavailable."
            ) from None
        returned_symbol = str(info.get("symbol") or "").strip().upper()
        if returned_symbol != symbol:
            raise YahooRatioMalformedResponse(
                "Normalized ratios returned an invalid response."
            )

        values: Dict[str, float] = {}
        provider_fields: Dict[str, str] = {}
        for metric, provider_field in _FIELD_MAP.items():
            value = self._finite_number(info.get(provider_field))
            if value is None:
                continue
            if metric == "debt_to_equity":
                value /= 100.0
            values[metric] = value
            provider_fields[metric] = provider_field

        fetched_at = self._now()
        if fetched_at.tzinfo is None:
            raise YahooRatioMalformedResponse(
                "Normalized ratios returned an invalid response."
            )
        return YahooRatioSnapshot(
            symbol=symbol,
            fetched_at=fetched_at.astimezone(timezone.utc).isoformat(),
            values=values,
            provider_fields=provider_fields,
        )
