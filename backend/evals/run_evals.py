"""
CLI Runner for GOATlens Evaluation Framework.

Usage:
    python -m backend.evals.run_evals               # all layers, 1 default ticker
    python -m backend.evals.run_evals --layer 1      # data pipeline only
    python -m backend.evals.run_evals --layer 2      # agent quality only
    python -m backend.evals.run_evals --layer 3      # consensus only
    python -m backend.evals.run_evals --ticker AAPL   # specific ticker
    python -m backend.evals.run_evals --all-golden    # run all 5 golden tickers

HOW IT WORKS:
1. Calls the GOATlens API (same as the frontend does)
2. Feeds the response into each eval layer
3. Prints a summary table showing pass/fail for each check
4. Optionally saves results to a JSON file

The runner talks to your running server (default: http://localhost:8000).
Make sure the server is up before running evals.
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path so imports work when run as module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import httpx
except ImportError:
    httpx = None

from backend.evals.data_pipeline_evals import run_data_pipeline_evals, LayerResults
from backend.evals.agent_evals import run_agent_evals
from backend.evals.consensus_evals import run_consensus_evals
from backend.evals.golden_dataset import GOLDEN_TICKERS


# ─── Colours for terminal output ───

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


def _colour(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


# ─── API Client ───


async def fetch_analysis(ticker: str, base_url: str) -> Optional[Dict[str, Any]]:
    """Fetch full analysis from the GOATlens API."""
    if httpx is None:
        print(_colour("ERROR: httpx not installed. Run: pip install httpx", RED))
        return None

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            print(f"  Fetching analysis for {ticker}...", end=" ", flush=True)
            resp = await client.post(
                f"{base_url}/api/analyze",
                json={"ticker": ticker},
            )
            resp.raise_for_status()
            print(_colour("✓", GREEN))
            return resp.json()
        except httpx.HTTPStatusError as e:
            print(_colour(f"✗ HTTP {e.response.status_code}", RED))
            return None
        except httpx.ConnectError:
            print(_colour("✗ Cannot connect to server", RED))
            print(f"  Make sure the server is running at {base_url}")
            return None
        except Exception as e:
            print(_colour(f"✗ {e}", RED))
            return None


async def fetch_earnings(ticker: str, base_url: str) -> Optional[Dict[str, Any]]:
    """Fetch earnings data from the GOATlens API."""
    if httpx is None:
        return None

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            print(f"  Fetching earnings for {ticker}...", end=" ", flush=True)
            resp = await client.get(f"{base_url}/api/earnings/{ticker}")
            resp.raise_for_status()
            print(_colour("✓", GREEN))
            return resp.json()
        except Exception as e:
            print(_colour(f"✗ {e}", RED))
            return None


async def fetch_price_history(ticker: str, base_url: str) -> Optional[Dict[str, Any]]:
    """Fetch price history from the GOATlens API."""
    if httpx is None:
        return None

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            print(f"  Fetching price history for {ticker}...", end=" ", flush=True)
            resp = await client.get(f"{base_url}/api/price-history/{ticker}")
            resp.raise_for_status()
            print(_colour("✓", GREEN))
            return resp.json()
        except Exception as e:
            print(_colour(f"✗ {e}", RED))
            return None


# ─── LLM Client Setup ───


def get_llm_client():
    """Get LLM client for LLM-as-judge evals (optional)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from backend.llm.client import LLMClient
        return LLMClient(model_name="gpt-4o-mini")
    except Exception:
        return None


# ─── Result Printing ───


def print_layer_results(layer: LayerResults):
    """Print a formatted table of eval results."""
    print()
    print(f"  {BOLD}{layer.layer}{RESET}")
    print(f"  {'─' * 60}")

    for result in layer.results:
        icon = _colour("✓", GREEN) if result.passed else (
            _colour("⚠", YELLOW) if result.severity == "warning" else _colour("✗", RED)
        )
        name = result.name
        # Truncate long names
        if len(name) > 45:
            name = name[:42] + "..."

        line = f"  {icon} {name}"
        if result.details and not result.passed:
            detail = result.details[:80]
            line += f"\n    {DIM}{detail}{RESET}"

        print(line)

    # Summary line
    total = len(layer.results)
    passed = layer.pass_count
    failed = layer.fail_count
    rate = layer.pass_rate * 100

    colour = GREEN if rate == 100 else (YELLOW if rate >= 80 else RED)
    print(f"\n  {_colour(f'{passed}/{total} passed ({rate:.0f}%)', colour)}")


def print_summary(all_layers: List[LayerResults]):
    """Print final summary across all layers."""
    print()
    print(f"{'═' * 64}")
    print(f"  {BOLD}EVAL SUMMARY{RESET}")
    print(f"{'═' * 64}")

    total_passed = sum(l.pass_count for l in all_layers)
    total_failed = sum(l.fail_count for l in all_layers)
    total = total_passed + total_failed
    rate = total_passed / total * 100 if total else 0

    for layer in all_layers:
        lrate = layer.pass_rate * 100
        colour = GREEN if lrate == 100 else (YELLOW if lrate >= 80 else RED)
        print(f"  {layer.layer}: {_colour(f'{layer.pass_count}/{len(layer.results)} ({lrate:.0f}%)', colour)}")

    print(f"\n  {BOLD}Total: {total_passed}/{total} ({rate:.0f}%){RESET}")

    if total_failed > 0:
        errors = [
            r for l in all_layers for r in l.results
            if not r.passed and r.severity == "error"
        ]
        warnings = [
            r for l in all_layers for r in l.results
            if not r.passed and r.severity == "warning"
        ]
        if errors:
            print(f"  {_colour(f'{len(errors)} error(s)', RED)}, {_colour(f'{len(warnings)} warning(s)', YELLOW)}")

    print()


def save_results_json(all_layers: List[LayerResults], ticker: str, output_dir: str):
    """Save eval results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{ticker}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    data = {
        "ticker": ticker,
        "timestamp": timestamp,
        "layers": [],
    }
    for layer in all_layers:
        layer_data = {
            "name": layer.layer,
            "pass_count": layer.pass_count,
            "fail_count": layer.fail_count,
            "pass_rate": layer.pass_rate,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "details": r.details,
                    "severity": r.severity,
                }
                for r in layer.results
            ],
        }
        data["layers"].append(layer_data)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Results saved to: {filepath}")


# ─── Main ───


async def run_evals_for_ticker(
    ticker: str,
    layers: List[int],
    base_url: str,
    llm_client=None,
) -> List[LayerResults]:
    """Run selected eval layers for a single ticker."""
    all_layers = []

    print(f"\n{'━' * 64}")
    print(f"  {BOLD}Evaluating: {ticker}{RESET}")
    print(f"{'━' * 64}")

    # Fetch data from API
    analysis = await fetch_analysis(ticker, base_url)
    if analysis is None:
        print(_colour("  Skipping — could not fetch analysis data", RED))
        return all_layers

    # Also fetch earnings + price data if Layer 1 is requested
    earnings_data = None
    price_data = None
    if 1 in layers:
        earnings_resp = await fetch_earnings(ticker, base_url)
        price_resp = await fetch_price_history(ticker, base_url)
        earnings_data = (earnings_resp.get("earnings_history") or earnings_resp.get("earnings", [])) if earnings_resp else []
        price_data = price_resp.get("prices", []) if price_resp else []

    # ── Layer 1: Data Pipeline ──
    if 1 in layers:
        # Extract metrics from first agent result (they all get the same data)
        agent_results = analysis.get("agent_results", [])
        first_metrics = agent_results[0].get("metrics", {}) if agent_results else {}

        layer1 = run_data_pipeline_evals(
            earnings=earnings_data or [],
            prices=price_data or [],
            metrics=first_metrics,
            full_response=analysis,
        )
        print_layer_results(layer1)
        all_layers.append(layer1)

    # ── Layer 2: Agent Quality ──
    if 2 in layers:
        agent_results = analysis.get("agent_results", [])
        layer2 = await run_agent_evals(
            agent_outputs=agent_results,
            ticker=ticker,
            llm_client=llm_client,
        )
        print_layer_results(layer2)
        all_layers.append(layer2)

    # ── Layer 3: Consensus Quality ──
    if 3 in layers:
        # The API flattens consensus into top-level keys.
        # We rebuild the nested dict that the consensus eval expects.
        consensus = analysis.get("consensus", {})
        if not consensus:
            # Rebuild from flat API response
            consensus = {
                "verdict": analysis.get("consensus_verdict", ""),
                "agreement_score": analysis.get("agreement_score"),
                "consensus_points": analysis.get("consensus_points", []),
                "divergence_points": analysis.get("divergence_points", []),
            }
        agent_results = analysis.get("agent_results", [])
        layer3 = await run_consensus_evals(
            consensus=consensus,
            agent_outputs=agent_results,
            ticker=ticker,
            llm_client=llm_client,
        )
        print_layer_results(layer3)
        all_layers.append(layer3)

    return all_layers


async def main():
    parser = argparse.ArgumentParser(
        description="GOATlens Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backend.evals.run_evals                  # default: AAPL, all layers
  python -m backend.evals.run_evals --ticker NVDA     # specific ticker
  python -m backend.evals.run_evals --layer 1         # data checks only
  python -m backend.evals.run_evals --all-golden      # all 5 golden tickers
  python -m backend.evals.run_evals --save             # save results to JSON
        """,
    )
    parser.add_argument(
        "--ticker", "-t",
        default="AAPL",
        help="Ticker symbol to evaluate (default: AAPL)",
    )
    parser.add_argument(
        "--layer", "-l",
        type=int,
        choices=[1, 2, 3],
        help="Run only this eval layer (1=data, 2=agents, 3=consensus)",
    )
    parser.add_argument(
        "--all-golden",
        action="store_true",
        help="Run evals for all 5 golden dataset tickers",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the GOATlens API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file in backend/evals/results/",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM-as-judge evals (code-based only)",
    )

    args = parser.parse_args()

    # Determine which layers to run
    layers = [args.layer] if args.layer else [1, 2, 3]

    # Determine which tickers to evaluate
    tickers = list(GOLDEN_TICKERS.keys()) if args.all_golden else [args.ticker.upper()]

    # Setup LLM client
    llm_client = None
    if not args.no_llm:
        llm_client = get_llm_client()
        if llm_client:
            print(_colour("  LLM-as-judge: enabled (gpt-4o-mini)", GREEN))
        else:
            print(_colour("  LLM-as-judge: disabled (no OPENAI_API_KEY)", YELLOW))
    else:
        print(_colour("  LLM-as-judge: disabled (--no-llm flag)", DIM))

    print(f"  Layers: {layers}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Server: {args.url}")

    # Run evals
    all_results = []
    for ticker in tickers:
        results = await run_evals_for_ticker(
            ticker=ticker,
            layers=layers,
            base_url=args.url,
            llm_client=llm_client,
        )
        all_results.extend(results)

    # Summary
    if all_results:
        print_summary(all_results)

        if args.save:
            output_dir = os.path.join(os.path.dirname(__file__), "results")
            ticker_label = "golden" if args.all_golden else tickers[0]
            save_results_json(all_results, ticker_label, output_dir)
    else:
        print(_colour("\n  No results — check that the server is running.", RED))


if __name__ == "__main__":
    asyncio.run(main())
