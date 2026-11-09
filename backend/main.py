"""GOATlens web server.

Serves the page and the nightly snapshots. It makes no calls to Yahoo or the SEC
while a user waits: all data is prepared by the nightly GitHub Action
(lens/build.py) and published to the repo's `data` branch.

Snapshots are read from SNAPSHOT_DIR when set (local development and tests),
otherwise from SNAPSHOT_BASE_URL (the `data` branch on GitHub).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from lens.investors import INVESTORS, consensus

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR")
SNAPSHOT_BASE_URL = os.getenv(
    "SNAPSHOT_BASE_URL",
    "https://raw.githubusercontent.com/pinnyeow/ai-investment-goatlens/data",
).rstrip("/")
CACHE_SECONDS = 30 * 60
TICKER_PATTERN = re.compile(r"^(?=.{1,10}$)[A-Z0-9]+(?:[.-][A-Z0-9]+)?$")
PERIOD_DAYS = {"1mo": 31, "3m": 92, "3mo": 92, "6m": 183, "6mo": 183, "1y": 366, "2y": 731, "5y": 1830}

app = FastAPI(title="GOATlens", version="1.0.0",
              description="How five legendary investors would look at a stock. Educational, not advice.")

_cache: Dict[str, tuple[float, Any]] = {}
_http: Optional[httpx.AsyncClient] = None


async def load(path: str) -> Optional[Any]:
    """Read a JSON file from the snapshot store, cached for 30 minutes. None if absent."""
    hit = _cache.get(path)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    if SNAPSHOT_DIR:
        file = Path(SNAPSHOT_DIR) / path
        data = json.loads(file.read_text()) if file.exists() else None
    else:
        global _http
        _http = _http or httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        try:
            response = await _http.get(f"{SNAPSHOT_BASE_URL}/{path}")
        except httpx.HTTPError:
            if hit:  # serve the older copy rather than failing
                return hit[1]
            raise HTTPException(503, "GOATlens data is temporarily unreachable. Please try again shortly.")
        if response.status_code == 404:
            data = None
        elif response.status_code != 200:
            if hit:
                return hit[1]
            raise HTTPException(503, "GOATlens data is temporarily unreachable. Please try again shortly.")
        else:
            data = response.json()
    if len(_cache) > 400:
        _cache.clear()
    _cache[path] = (time.monotonic(), data)
    return data


def normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper().replace(".", "-")
    if not TICKER_PATTERN.fullmatch(ticker):
        raise HTTPException(422, "Ticker must be 1-10 letters or digits with one optional . or -")
    return ticker


async def snapshot_for(ticker: str) -> Dict[str, Any]:
    snapshot = await load(f"snapshots/{ticker}.json")
    if snapshot is None:
        index = await load("index.json") or []
        suggestions = [item["ticker"] for item in index if item["ticker"].startswith(ticker[:2])][:6]
        hint = f" Try: {', '.join(suggestions)}." if suggestions else ""
        raise HTTPException(404, (
            f"{ticker} isn't covered yet. GOATlens covers {len(index)} companies "
            f"(the S&P 500 plus popular names), refreshed nightly.{hint}"
        ))
    return snapshot


# ── Pages and health ────────────────────────────────────────────────────────

def _inline_script_hashes() -> str:
    """CSP hashes for the page's inline scripts, computed from the file itself so
    editing the page can never silently break it."""
    html = FRONTEND.read_text()
    scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
    return " ".join(
        f"'sha256-{base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()}'" for body in scripts
    )


_CSP = (
    f"default-src 'self'; script-src 'self' {_inline_script_hashes()} https://unpkg.com https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
    "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path == "/":
        response.headers["Content-Security-Policy"] = _CSP
    return response


@app.get("/")
async def root():
    return FileResponse(FRONTEND, headers={"Cache-Control": "no-cache"})


@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/ping", methods=["GET", "HEAD"])
async def health():
    return {"status": "healthy", "service": "goatlens",
            "release": os.getenv("RENDER_GIT_COMMIT", "local")}


@app.get("/api/coverage")
async def coverage():
    report = await load("coverage.json")
    if report is None:
        raise HTTPException(503, "Coverage report not published yet.")
    return report


@app.get("/api/tickers")
async def tickers():
    return await load("index.json") or []


@app.get("/api/agents")
async def agents():
    return {"agents": [
        {"id": investor_id, "name": name, "style": style,
         "criteria": [criterion.label for criterion in criteria]}
        for investor_id, (name, style, criteria) in INVESTORS.items()
    ]}


# ── Analysis ────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str
    agents: List[str] = Field(default_factory=lambda: list(INVESTORS))
    time_period: Optional[str] = None  # accepted for older clients; analysis uses the latest annual data

    @field_validator("agents")
    @classmethod
    def known_agents(cls, value: List[str]) -> List[str]:
        chosen = [a for a in dict.fromkeys(value) if a in INVESTORS]
        if not chosen:
            raise ValueError("Choose at least one of: " + ", ".join(INVESTORS))
        return chosen


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    snapshot = await snapshot_for(normalize_ticker(request.ticker))
    results = [r for r in snapshot["agent_results"] if r["id"] in request.agents]
    body = {k: v for k, v in snapshot.items() if k not in ("prices", "agent_results")}
    body.update(consensus(results))
    body["agent_results"] = results
    body["completed_analyses"] = sum(r["status"] == "complete" for r in results)
    body["requested_analyses"] = len(results)
    return body


# ── Charts and earnings ─────────────────────────────────────────────────────

def _price_points(rows: List[list], period: str) -> List[Dict[str, Any]]:
    if not rows:
        return []
    last = rows[-1][0]
    if period == "ytd":
        start = f"{last[:4]}-01-01"
    else:
        start = str(date.fromisoformat(last) - timedelta(days=PERIOD_DAYS.get(period, 366)))
    return [
        {"date": d, "open": o, "high": h, "low": lo, "close": c, "adjusted_close": a, "volume": v}
        for d, o, h, lo, c, a, v in rows if d >= start
    ]


def _reaction(rows: List[list], day: str, sessions: int) -> Optional[float]:
    dates = [r[0] for r in rows]
    before = [i for i, d in enumerate(dates) if d < day]
    if not before or before[-1] + sessions >= len(rows):
        return None
    base = rows[before[-1]][4]
    return round((rows[before[-1] + sessions][4] / base - 1) * 100, 2) if base else None


@app.get("/api/price-history/{ticker}")
async def price_history(ticker: str, period: str = "1y"):
    snapshot = await snapshot_for(normalize_ticker(ticker))
    rows = snapshot.get("prices", [])
    prices = _price_points(rows, period)
    first = prices[0]["date"] if prices else "9999"
    markers = [
        {**{k: q.get(k) for k in ("quarter", "date", "eps_actual", "eps_estimate", "surprise_pct", "beat_miss")},
         "price_reaction_1d": _reaction(rows, q["date"], 1),
         "price_reaction_5d": _reaction(rows, q["date"], 5)}
        for q in snapshot.get("earnings_history", []) if q["date"] >= first
    ]
    return {"ticker": snapshot["ticker"], "period": period, "prices": prices,
            "earnings_markers": markers, "next_earnings_date": snapshot.get("next_earnings_date")}


@app.get("/api/earnings/{ticker}")
async def earnings(ticker: str):
    snapshot = await snapshot_for(normalize_ticker(ticker))
    return {"ticker": snapshot["ticker"], "earnings_history": snapshot.get("earnings_history", []),
            "earnings_streak": snapshot.get("earnings_streak"), "reaction_insights": [],
            "forward_guidance": None, "next_earnings_date": snapshot.get("next_earnings_date")}
