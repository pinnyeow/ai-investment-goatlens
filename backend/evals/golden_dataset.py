"""
Golden Dataset for GOATlens Evaluation.

Contains known-good test cases with expected score ranges per agent.
These ranges are established from baseline behavior and verified manually.

WHY RANGES NOT EXACT SCORES:
Financial data changes daily. Apple's P/E today differs from next week.
But Buffett should *always* score Apple positively (strong moat, high ROE).
The range captures the invariant: "this should always be roughly positive."

HOW TO UPDATE:
1. Run evals against a ticker
2. If the score is reasonable, add it here with a ±20 buffer
3. Document why you expect that range (e.g., "AAPL has high ROE = Buffett likes it")
"""

from typing import Dict, Tuple, List


# Each ticker maps to:
#   - expected score range (min, max) per agent
#   - a brief rationale explaining why we expect that range
GOLDEN_TICKERS: Dict[str, Dict[str, Dict]] = {
    "AAPL": {
        "buffett": {
            "score_range": (-10, 80),
            "rationale": "High ROE, strong brand moat, consistent margins. Buffett literally owns it.",
        },
        "lynch": {
            "score_range": (-30, 40),
            "rationale": "Mega-cap with high institutional ownership — not a hidden gem. PEG often above 1.",
        },
        "graham": {
            "score_range": (-60, 10),
            "rationale": "High P/E and P/B — violates Graham's deep value thresholds. May have dividend record.",
        },
        "munger": {
            "score_range": (-10, 70),
            "rationale": "Exceptional business quality, pricing power, strong management. Munger's sweet spot.",
        },
        "dalio": {
            "score_range": (-30, 50),
            "rationale": "Low beta, strong balance sheet, but concentrated in tech (not diversified).",
        },
    },
    "MSFT": {
        "buffett": {
            "score_range": (-10, 80),
            "rationale": "Wide moat (enterprise software), high ROE, strong margins.",
        },
        "lynch": {
            "score_range": (-30, 40),
            "rationale": "Large cap, high institutional ownership. Steady but not a ten-bagger.",
        },
        "graham": {
            "score_range": (-60, 10),
            "rationale": "Typically trades at premium P/E and P/B. Not a deep value stock.",
        },
        "munger": {
            "score_range": (0, 75),
            "rationale": "Best-in-class business quality, network effects, pricing power.",
        },
        "dalio": {
            "score_range": (-20, 50),
            "rationale": "Strong balance sheet, cloud diversification, but tech-concentrated.",
        },
    },
    "JPM": {
        "buffett": {
            "score_range": (-20, 50),
            "rationale": "Decent ROE for a bank, but financial sector has inherent leverage risks.",
        },
        "lynch": {
            "score_range": (-40, 30),
            "rationale": "Banks are cyclical. Lynch categorizes as slow grower or cyclical.",
        },
        "graham": {
            "score_range": (-10, 60),
            "rationale": "Often trades at low P/E and P/B — fits Graham's value criteria. Dividend payer.",
        },
        "munger": {
            "score_range": (-30, 40),
            "rationale": "Quality management but financial complexity. Munger distrusts banks somewhat.",
        },
        "dalio": {
            "score_range": (-40, 40),
            "rationale": "Debt cycle exposure is high. Dalio watches bank leverage closely.",
        },
    },
    "NVDA": {
        "buffett": {
            "score_range": (-40, 40),
            "rationale": "High margins but tech/semiconductor is hard to predict long-term. Volatile.",
        },
        "lynch": {
            "score_range": (-20, 60),
            "rationale": "High earnings growth. If PEG < 1, Lynch loves it. If PEG > 2, penalized.",
        },
        "graham": {
            "score_range": (-80, -10),
            "rationale": "Extreme P/E and P/B. Graham would never touch this at current valuations.",
        },
        "munger": {
            "score_range": (-20, 60),
            "rationale": "Incredible business quality and competitive position in AI chips.",
        },
        "dalio": {
            "score_range": (-50, 30),
            "rationale": "High beta, concentrated bet. Dalio prefers all-weather diversification.",
        },
    },
    "KO": {
        "buffett": {
            "score_range": (10, 80),
            "rationale": "Buffett's iconic holding. Strong brand moat, consistent earnings, low debt.",
        },
        "lynch": {
            "score_range": (-40, 20),
            "rationale": "Slow grower. Lynch prefers faster earnings growth. Low ten-bagger potential.",
        },
        "graham": {
            "score_range": (-30, 40),
            "rationale": "Moderate P/E, strong dividend record. Mixed on P/B.",
        },
        "munger": {
            "score_range": (0, 60),
            "rationale": "Simple business, incredible brand, global distribution. Munger approves.",
        },
        "dalio": {
            "score_range": (-20, 40),
            "rationale": "Defensive stock, low beta. Good for all-weather, but limited growth.",
        },
    },
}


# Philosophy keywords: words/phrases that SHOULD appear in an agent's insights
# if they're staying in character. Used by the philosophy adherence eval.
PHILOSOPHY_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "buffett": {
        "expected_terms": [
            "moat", "ROE", "margin", "debt", "owner earnings", "competitive advantage",
            "quality", "long-term", "pricing power", "management",
        ],
        "forbidden_terms": [
            "momentum", "technical", "short-term", "day trade", "moving average",
            "RSI", "MACD", "swing trade",
        ],
        "philosophy": "Value investing with quality focus. Buy wonderful companies at fair prices.",
        "principles": [
            "Durable competitive advantage (moat)",
            "Consistent earnings power",
            "ROE above 15%",
            "Conservative debt levels",
            "Quality management with integrity",
        ],
    },
    "lynch": {
        "expected_terms": [
            "PEG", "growth", "ten-bagger", "earnings", "institutional",
            "category", "stalwart", "fast grower", "hidden",
        ],
        "forbidden_terms": [
            "moat", "owner earnings", "margin of safety", "debt cycle",
            "risk parity", "all-weather",
        ],
        "philosophy": "Growth at a Reasonable Price (GARP). Find overlooked growth stories.",
        "principles": [
            "PEG ratio below 1 is ideal",
            "Earnings growth momentum",
            "Low institutional ownership (contrarian signal)",
            "Business simplicity and understandability",
            "Hidden assets and overlooked opportunities",
        ],
    },
    "graham": {
        "expected_terms": [
            "P/E", "P/B", "margin of safety", "intrinsic value", "current ratio",
            "dividend", "balance sheet", "defensive", "net-net",
        ],
        "forbidden_terms": [
            "momentum", "ten-bagger", "fast grower", "pricing power",
            "mental model", "debt cycle",
        ],
        "philosophy": "Deep value investing. Buy below intrinsic value with a margin of safety.",
        "principles": [
            "P/E ratio below 15",
            "P/B ratio below 1.5",
            "Current ratio above 2",
            "Consistent dividend record",
            "Margin of safety below intrinsic value",
        ],
    },
    "munger": {
        "expected_terms": [
            "quality", "mental model", "management", "pricing power",
            "red flag", "inversion", "structural advantage", "incentive",
        ],
        "forbidden_terms": [
            "PEG", "ten-bagger", "net-net", "current ratio",
            "debt cycle", "risk parity",
        ],
        "philosophy": "Quality-first investing. Use mental models from multiple disciplines.",
        "principles": [
            "Business quality over cheap valuation",
            "Mental models and multidisciplinary thinking",
            "Management quality and aligned incentives",
            "Avoiding mistakes (inversion)",
            "Long-term structural advantages",
        ],
    },
    "dalio": {
        "expected_terms": [
            "risk", "debt", "cycle", "diversification", "beta",
            "macro", "all-weather", "stress test", "volatility",
        ],
        "forbidden_terms": [
            "moat", "ten-bagger", "PEG", "net-net",
            "margin of safety", "owner earnings",
        ],
        "philosophy": "Macro-aware, risk-parity investing. Build all-weather portfolios.",
        "principles": [
            "Debt cycle positioning",
            "Risk-adjusted returns",
            "Diversification contribution",
            "Stress-testing under scenarios",
            "Low correlation to broad market",
        ],
    },
}


# Verdict thresholds — the mapping every agent uses
# Used to verify score-to-verdict consistency
VERDICT_THRESHOLDS = {
    "strong_buy": (60, 100),
    "buy": (20, 59.99),
    "hold": (-20, 19.99),
    "sell": (-60, -20.01),
    "strong_sell": (-100, -60.01),
}


# Required keys in every agent output as returned by the API.
# NOTE: Agents internally include "ticker" and "metrics", but the Pydantic
# response model (AgentResult) strips them. We check what the *user* sees.
REQUIRED_OUTPUT_KEYS = [
    "agent", "style", "score", "verdict", "insights", "concerns",
]
