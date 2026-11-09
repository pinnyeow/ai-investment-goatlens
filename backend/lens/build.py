"""Nightly build: fetch every covered company, score the five investors, write JSON.

    python -m lens.build --out ../snapshots                 # whole universe
    python -m lens.build --out /tmp/snap --tickers AAPL,KO  # a few tickers

If a ticker fails tonight, its previous snapshot (if any) is kept and marked
stale, so a bad night at Yahoo never takes a company off the site.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from lens.investors import INVESTORS, consensus, evaluate_investor, moat_trend

UNIVERSE_FILE = Path(__file__).with_name("universe.txt")
SCHEMA_VERSION = 1


def load_universe() -> List[str]:
    tickers = []
    for line in UNIVERSE_FILE.read_text().splitlines():
        line = line.split("#", 1)[0].strip().upper()
        if line and line not in tickers:
            tickers.append(line)
    return tickers


def earnings_streak(history: List[Dict]) -> Dict:
    known = [q for q in history if q["beat_miss"] in ("beat", "miss", "inline")]
    counts = Counter(q["beat_miss"] for q in known)
    streak_type, streak = "unknown", 0
    if known:
        streak_type = known[0]["beat_miss"]
        for quarter in known:
            if quarter["beat_miss"] != streak_type:
                break
            streak += 1
    summary = (
        f"{streak_type.capitalize()} {streak} quarter{'s' if streak != 1 else ''} in a row"
        if known else "No earnings history"
    )
    return {
        "streak_type": streak_type, "streak_count": streak, "beats": counts["beat"],
        "misses": counts["miss"], "inline": counts["inline"], "total": len(known), "summary": summary,
    }


def analyze(company: Dict) -> Dict:
    """Turn fetched company data into the snapshot the website serves."""
    metrics = company["metrics"]
    not_applicable = set(company.get("not_applicable", []))
    results = [evaluate_investor(investor_id, metrics, not_applicable) for investor_id in INVESTORS]
    complete = sum(r["status"] == "complete" for r in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": company["ticker"],
        "company_name": company["company_name"],
        "sector": company["sector"],
        "industry": company["industry"],
        "currency": company["currency"],
        "is_financial": company["is_financial"],
        "not_applicable": sorted(not_applicable),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stale": False,
        "data_status": {
            "source": "Yahoo Finance",
            "price_as_of": company["price_as_of"],
            "fiscal_year_end": company["fiscal_year_end"],
            "statement_years": company["statement_years"],
        },
        "metrics": metrics,
        "agent_results": results,
        "completed_analyses": complete,
        "requested_analyses": len(results),
        "moat_trend": moat_trend(metrics),
        **consensus(results),
        "earnings_history": company["earnings_history"],
        "earnings_streak": earnings_streak(company["earnings_history"]),
        "next_earnings_date": company["next_earnings_date"],
        "prices": company["prices"],
    }


def build(tickers: List[str], out: Path, pause: float) -> Dict:
    from lens.data import fetch_company, fetch_market_returns

    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)
    market = fetch_market_returns()
    fresh, stale, missing = [], [], []
    index = []
    for i, ticker in enumerate(tickers, 1):
        path = out / "snapshots" / f"{ticker}.json"
        try:
            snapshot = analyze(fetch_company(ticker, market))
            path.write_text(json.dumps(snapshot, separators=(",", ":")))
            fresh.append(ticker)
        except Exception as exc:  # keep yesterday's snapshot if there is one
            print(f"[{i}/{len(tickers)}] {ticker}: {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
            if not path.exists():
                missing.append(ticker)
                continue
            snapshot = json.loads(path.read_text())
            snapshot["stale"] = True
            path.write_text(json.dumps(snapshot, separators=(",", ":")))
            stale.append(ticker)
        index.append({
            "ticker": snapshot["ticker"],
            "name": snapshot["company_name"],
            "sector": snapshot["sector"],
            "complete": snapshot["completed_analyses"],
        })
        if i % 50 == 0:
            print(f"[{i}/{len(tickers)}] {len(fresh)} fresh, {len(stale)} stale, {len(missing)} missing")
        time.sleep(pause)

    covered = len(index)
    full = sum(1 for item in index if item["complete"] == len(INVESTORS))
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe": len(tickers),
        "covered": covered,
        "fresh": len(fresh),
        "stale": stale,
        "missing": missing,
        "all_five_investors": full,
        "all_five_rate": round(full / covered, 3) if covered else 0,
        "by_completed_count": dict(Counter(item["complete"] for item in index)),
    }
    (out / "index.json").write_text(json.dumps(sorted(index, key=lambda x: x["ticker"]), separators=(",", ":")))
    (out / "coverage.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tickers", help="comma-separated; defaults to universe.txt")
    parser.add_argument("--pause", type=float, default=0.3, help="seconds between tickers")
    parser.add_argument("--min-fresh-rate", type=float, default=0.8,
                        help="exit non-zero (so nothing is published) if fewer succeed tonight")
    args = parser.parse_args()
    tickers = args.tickers.upper().split(",") if args.tickers else load_universe()
    report = build(tickers, args.out, args.pause)
    print(json.dumps(report, indent=2))
    if report["fresh"] < args.min_fresh_rate * report["universe"]:
        print("Too many failures tonight; not publishing.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
