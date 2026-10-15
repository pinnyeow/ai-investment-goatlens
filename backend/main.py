"""
GOATlens - Multi-Agent Investment Analysis Framework

A research tool that analyzes companies using the mental models of 
legendary investors (the GOATs).

Features:
- Time Period Analysis: Analyze company over YTD, 3M, 6M, 1Y, or 5Y
- Moat Detection: Identify competitive advantage strength
- Multi-Agent Debate: Parallel analysis from different investor perspectives
- Consensus/Divergence: Surface where GOATs agree vs. disagree
"""

import os
import asyncio
import math
import re
import threading
import time
from typing import Dict, Any, List, Literal, NoReturn, Optional
from contextlib import asynccontextmanager

# Load environment variables from .env file
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the backend directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- Arize AX Auto-Instrumentation (MUST be before LangGraph imports) ---
_arize_tracing_enabled = False
_arize_tracer_provider = None

space_id = os.getenv("ARIZE_SPACE_ID")
api_key = os.getenv("ARIZE_API_KEY")

if space_id and api_key:
    try:
        from arize.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        
        print("[Arize] Initializing tracing...")
        
        _arize_tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name="goatlens"
        )
        
        LangChainInstrumentor().instrument(
            tracer_provider=_arize_tracer_provider,
            include_chains=True,
            include_agents=True,
            include_tools=True
        )
        
        _arize_tracing_enabled = True
        print("[Arize] ✓ Tracing enabled successfully for project 'goatlens'")
        print("[Arize] ✓ LangChain instrumentor active (chains, agents, tools)")
    except ImportError:
        print("[Arize] Tracing unavailable: required packages are not installed.")
    except Exception:
        print("[Arize] Tracing setup failed.")
else:
    missing = []
    if not space_id:
        missing.append("ARIZE_SPACE_ID")
    if not api_key:
        missing.append("ARIZE_API_KEY")
    print(f"[Arize] ⚠ Tracing disabled: Missing environment variables: {', '.join(missing)}")
    print("[Arize]   Set ARIZE_SPACE_ID and ARIZE_API_KEY in your .env file or environment to enable tracing")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda, RunnableConfig
from typing_extensions import TypedDict

# Local imports
from agents import BuffettAgent, LynchAgent, GrahamAgent, MungerAgent, DalioAgent
from data_sources import YahooFinanceClient, YahooFinanceError
from data_sources.sec import (
    SECConfigurationError,
    SECError,
    SECFundamentalsClient,
    SECInvalidResponseError,
    SECIssuerNotFoundError,
    SECRateLimitError,
    SECUnavailableError,
    normalize_sec_fundamentals,
)
from data_sources import NewsClient
from temporal import TemporalAnalyzer
from strategies import calculate_consensus, calculate_consensus_with_llm, StrategyResult, Verdict
from llm import get_llm_client
from rag import RAGRetriever


# ============================================================================
# Pydantic Models
# ============================================================================

_TICKER_PATTERN = re.compile(r"(?=.{1,10}\Z)[A-Z0-9]+(?:[.-][A-Z0-9]+)?\Z")


def _normalize_ticker(value: str) -> str:
    """Normalize and validate a common-stock ticker symbol."""
    ticker = value.strip().upper()
    if not _TICKER_PATTERN.fullmatch(ticker):
        raise ValueError("Ticker must be 1-10 letters/digits with one optional . or -")
    return ticker


def _normalize_path_ticker(value: str) -> str:
    """Apply ticker validation to path parameters with an HTTP 422 response."""
    try:
        return _normalize_ticker(value)
    except (AttributeError, ValueError):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")


class AnalysisRequest(BaseModel):
    """Request to analyze a company."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    time_period: Literal["ytd", "3m", "6m", "1y", "5y"] = Field(
        default="1y",
        description="Time period for analysis: ytd, 3m, 6m, 1y, 5y",
    )
    agents: Optional[List[str]] = Field(
        default=None,
        description="Specific agents to run (default: all)",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return _normalize_ticker(value)

    @field_validator("agents")
    @classmethod
    def validate_agents(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        supported = {"buffett", "lynch", "graham", "munger", "dalio"}
        if not value or any(agent not in supported for agent in value):
            raise ValueError("Agents must be one or more supported agent identifiers")
        return list(dict.fromkeys(value))


class AgentResult(BaseModel):
    """Result from a single GOAT agent."""
    agent: str = Field(min_length=1)
    style: str = Field(min_length=1)
    verdict: Optional[
        Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    ] = None
    score: Optional[float] = Field(default=None, ge=-100, le=100)
    insights: List[str]
    concerns: List[str]
    status: Literal["complete", "incomplete"] = "complete"
    unavailable_metrics: List[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def validate_score_type(cls, value):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError("Agent score must be numeric")
        return value

    @model_validator(mode="after")
    def validate_completion_contract(self):
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("Agent score must be finite")
        if self.status == "complete":
            score = self.score
            verdict = self.verdict
            if score is None or verdict is None:
                raise ValueError("Complete agent results require score and verdict")
            expected_verdict = (
                "strong_buy" if score >= 60
                else "buy" if score >= 20
                else "hold" if score >= -20
                else "sell" if score >= -60
                else "strong_sell"
            )
            if verdict != expected_verdict:
                raise ValueError("Agent score and verdict are inconsistent")
            if self.unavailable_metrics:
                raise ValueError("Complete agent results cannot list unavailable metrics")
        if self.status == "incomplete" and (
            self.score is not None or self.verdict is not None
        ):
            raise ValueError("Incomplete agent results cannot claim score or verdict")
        if self.status == "incomplete" and not self.unavailable_metrics:
            raise ValueError("Incomplete agent results must identify unavailable metrics")
        return self


class EarningsQuarter(BaseModel):
    """Earnings data for a single quarter."""
    quarter: str = Field(description="Quarter label, e.g. Q4 2025")
    date: str = Field(description="Earnings report date YYYY-MM-DD")
    eps_actual: Optional[float] = Field(default=None, description="Reported EPS")
    eps_estimate: Optional[float] = Field(default=None, description="Consensus EPS estimate")
    eps_surprise: Optional[float] = Field(default=None, description="EPS surprise (actual - estimate)")
    surprise_pct: Optional[float] = Field(default=None, description="Surprise as percentage")
    beat_miss: str = Field(default="unknown", description="beat, miss, or inline")


class EarningsStreak(BaseModel):
    """Summary of earnings beat/miss track record."""
    streak_type: str = Field(description="Current streak: beat, miss, inline, or unknown")
    streak_count: int = Field(description="Number of consecutive quarters in current streak")
    beats: int = Field(description="Total beats in history")
    misses: int = Field(description="Total misses in history")
    inline: int = Field(description="Total inline results")
    total: int = Field(description="Total quarters with data")
    summary: str = Field(description="Human-readable summary")


class PricePoint(BaseModel):
    """Single day of price data."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class EarningsMarker(BaseModel):
    """Earnings event marker for overlaying on a price chart."""
    quarter: str
    date: str
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    surprise_pct: Optional[float] = None
    beat_miss: str = "unknown"
    price_reaction_1d: Optional[float] = None
    price_reaction_5d: Optional[float] = None


class PriceHistoryResponse(BaseModel):
    """Price history with earnings overlays for charting."""
    ticker: str
    period: str
    prices: List[PricePoint]
    earnings_markers: List[EarningsMarker]
    next_earnings_date: Optional[str] = None


class EarningsReactionInsight(BaseModel):
    """Per-quarter reaction insight explaining why the stock moved."""
    quarter: str = ""
    date: str = ""
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_beat_miss: str = "unknown"
    eps_surprise_pct: Optional[float] = None
    price_reaction_1d: Optional[float] = None
    price_reaction_5d: Optional[float] = None
    pre_earnings_runup: Optional[float] = None
    analyst_upgrades: int = 0
    analyst_downgrades: int = 0
    analyst_maintains: int = 0
    revenue_growth_yoy: Optional[float] = None
    guidance_signal: str = "unknown"
    insight: str = ""
    # Optional quarterly filing fields (None when unavailable)
    revenue_actual: Optional[float] = None
    revenue_yoy_pct: Optional[float] = None
    capex: Optional[float] = None
    capex_prev_quarter: Optional[float] = None
    capex_qoq_pct: Optional[float] = None
    capex_pct_revenue: Optional[float] = None
    fcf: Optional[float] = None


class EstimateRange(BaseModel):
    """Analyst estimate range for a single period (current Q or next Q)."""
    period: str = ""
    eps_low: Optional[float] = None
    eps_avg: Optional[float] = None
    eps_high: Optional[float] = None
    eps_year_ago: Optional[float] = None
    eps_growth: Optional[float] = None
    eps_num_analysts: int = 0
    rev_low: Optional[float] = None
    rev_avg: Optional[float] = None
    rev_high: Optional[float] = None
    rev_year_ago: Optional[float] = None
    rev_growth: Optional[float] = None
    rev_num_analysts: int = 0


class PriceTargetRange(BaseModel):
    """Analyst price target range."""
    current: Optional[float] = None
    low: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    high: Optional[float] = None


class ForwardGuidanceSummary(BaseModel):
    """Forward-looking guidance summary from analyst estimates."""
    current_q_eps_growth: Optional[float] = None
    next_q_eps_growth: Optional[float] = None
    current_q_rev_growth: Optional[float] = None
    next_q_rev_growth: Optional[float] = None
    growth_trend: Literal["accelerating", "decelerating", "stable", "unknown"] = (
        "unknown"
    )
    analyst_price_target_mean: Optional[float] = None
    analyst_price_target_upside_pct: Optional[float] = None
    num_analysts: Optional[int] = None
    # New: full estimate ranges and confidence
    estimate_ranges: List[EstimateRange] = []
    consensus_confidence: str = "unknown"
    price_targets: Optional[PriceTargetRange] = None
    summary: str = ""


class EarningsInsightsResponse(BaseModel):
    ticker: str
    earnings_history: List[EarningsQuarter]
    earnings_streak: Optional[EarningsStreak]
    reaction_insights: List[EarningsReactionInsight]
    forward_guidance: Optional[ForwardGuidanceSummary]
    next_earnings_date: Optional[str]


class FactProvenanceResponse(BaseModel):
    namespace: str
    concept: str
    unit: str
    start: Optional[str]
    end: str
    accession: str
    form: str
    filed: str
    fiscal_year: Optional[int]
    derived: bool
    inputs: List["FactProvenanceResponse"] = Field(default_factory=list)


class FundamentalFactResponse(BaseModel):
    value: Optional[float]
    availability_reason: Optional[str]
    provenance: Optional[FactProvenanceResponse]


class FundamentalsAvailabilityStatus(BaseModel):
    available: bool
    source: Literal["sec"]
    as_of: Optional[str]


class MarketAvailabilityStatus(BaseModel):
    available: bool
    source: Literal["yahoo"]
    as_of: Optional[str]
    reason: Optional[str] = None


class AnalysisCompletenessStatus(BaseModel):
    status: Literal["complete", "incomplete"]
    reason: Optional[str]


class AnalysisDataStatus(BaseModel):
    fundamentals: FundamentalsAvailabilityStatus
    market_data: MarketAvailabilityStatus
    price_dependent_analysis: AnalysisCompletenessStatus


class ComparisonAgentResponse(BaseModel):
    name: str
    style: str
    verdict: Optional[
        Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    ]
    score: Optional[float] = Field(default=None, ge=-100, le=100)
    key_metric: str


class ComparisonTableResponse(BaseModel):
    agents: List[ComparisonAgentResponse]


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    ticker: str
    company_name: str
    sector: str
    data_status: AnalysisDataStatus
    fundamentals: Dict[str, FundamentalFactResponse]
    fundamentals_history: Dict[str, List[FundamentalFactResponse]]
    
    # Consensus
    consensus_status: Literal["complete", "partial", "incomplete"]
    consensus_verdict: Literal[
        "strong_buy", "buy", "hold", "sell", "strong_sell", "incomplete"
    ]
    agreement_score: Optional[float] = Field(default=None, ge=0, le=1)
    completed_analyses: int = Field(ge=0)
    requested_analyses: int = Field(ge=1)
    consensus_points: List[str]
    divergence_points: List[str]
    
    # Individual agents
    agent_results: List[AgentResult]
    
    # Temporal analysis
    moat_trend: str
    
    # GOAT comparison table
    comparison_table: ComparisonTableResponse

    @model_validator(mode="after")
    def validate_analysis_counts(self):
        if self.completed_analyses > self.requested_analyses:
            raise ValueError("Completed analyses cannot exceed requested analyses")
        return self
    
    # Earnings history
    earnings_history: List[EarningsQuarter] = Field(default_factory=list)
    earnings_streak: Optional[EarningsStreak] = None
    
    # Next earnings date
    next_earnings_date: Optional[str] = None


# ============================================================================
# LangGraph State & Workflow
# ============================================================================

class GOATState(TypedDict):
    """State passed through the LangGraph workflow."""
    ticker: str
    time_period: str
    selected_agents: List[str]
    
    # Data
    raw_data: Optional[Dict[str, Any]]
    normalized_data: Optional[Dict[str, Any]]
    data_source: Optional[str]
    data_status: Optional[Dict[str, Any]]
    sec_fundamentals: Optional[Dict[str, Any]]
    sec_fundamentals_history: Optional[Dict[str, List[Dict[str, Any]]]]
    
    # Earnings data (actual vs. consensus)
    earnings_data: Optional[List[Dict[str, Any]]]
    earnings_streak: Optional[Dict[str, Any]]
    
    # Next earnings date (fetched once in fetch_data_node, shared across nodes)
    next_earnings_date: Optional[str]
    
    # News sentiment (optional tool - agents decide when to use it)
    news_sentiment: Optional[Dict[str, Any]]
    
    # RAG context (earnings transcripts, analyst reports)
    rag_context: Optional[List[Dict[str, Any]]]
    
    # Temporal analysis
    temporal_results: Optional[Dict[str, Any]]
    
    # Agent results
    agent_results: List[Dict[str, Any]]
    
    # Final synthesis
    consensus: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    
    # Error tracking
    error: Optional[str]
    error_code: Optional[str]
    market_data_only: bool


_TRANSIENT_YAHOO_ERROR_MARKERS = (
    "too many requests",
    "rate limit",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection",
    "service unavailable",
)


def _classify_yahoo_error(error: YahooFinanceError, ticker: str) -> str:
    """Classify provider failures without using user-controlled ticker text."""
    structured_code = getattr(error, "code", None)
    if structured_code in {
        "ticker_not_found",
        "market_data_temporarily_unavailable",
        "market_data_failure",
    }:
        return structured_code

    error_text = re.sub(
        rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])",
        "",
        str(error),
        flags=re.IGNORECASE,
    ).lower()
    if "no data found for ticker" in error_text:
        return "ticker_not_found"
    if any(marker in error_text for marker in _TRANSIENT_YAHOO_ERROR_MARKERS):
        return "market_data_temporarily_unavailable"
    return "market_data_failure"


def _serialize_sec_fundamentals(fundamentals):
    """Expose authoritative values and their complete filing provenance."""

    def provenance_payload(provenance):
        if provenance is None:
            return None
        return {
            "namespace": provenance.namespace,
            "concept": provenance.concept,
            "unit": provenance.unit,
            "start": provenance.start,
            "end": provenance.end,
            "accession": provenance.accession,
            "form": provenance.form,
            "filed": provenance.filed.isoformat(),
            "fiscal_year": provenance.fiscal_year,
            "derived": provenance.derived,
            "inputs": [provenance_payload(item) for item in provenance.inputs],
        }

    def value_payload(item):
        return {
            "value": item.value,
            "availability_reason": item.availability_reason,
            "provenance": provenance_payload(item.provenance),
        }

    return (
        {metric: value_payload(item) for metric, item in fundamentals.latest.items()},
        {
            metric: [value_payload(item) for item in items]
            for metric, items in fundamentals.history.items()
        },
    )


async def fetch_data_node(state: GOATState) -> GOATState:
    """Fetch SEC fundamentals first, then add best-effort Yahoo market data."""
    ticker = state["ticker"]
    market_data_only = state.get("market_data_only", False)

    if market_data_only:
        try:
            async with YahooFinanceClient() as client:
                raw_data = await client.get_company_data(ticker, years=1)
            state["raw_data"] = raw_data
            state["data_source"] = "yahoo"
            return state
        except YahooFinanceError as yahoo_error:
            error_code = _classify_yahoo_error(yahoo_error, ticker)
            state["error_code"] = error_code
            state["error"] = (
                f"No market data found for ticker {ticker}."
                if error_code == "ticker_not_found"
                else "Market data is temporarily unavailable."
                if error_code == "market_data_temporarily_unavailable"
                else "Market data request failed."
            )
            return state

    user_agent = os.getenv("SEC_USER_AGENT", "")
    try:
        async with SECFundamentalsClient(user_agent=user_agent) as sec_client:
            fundamentals = await sec_client.get_fundamentals(ticker)
    except SECConfigurationError:
        state["error"] = "SEC fundamental data is not configured."
        state["error_code"] = "fundamentals_configuration_error"
        return state
    except SECIssuerNotFoundError:
        state["error"] = f"No SEC issuer found for ticker {ticker}."
        state["error_code"] = "ticker_not_found"
        return state
    except SECInvalidResponseError:
        state["error"] = "SEC returned invalid fundamental data."
        state["error_code"] = "fundamentals_invalid_response"
        return state
    except SECUnavailableError:
        state["error"] = "SEC fundamental data is temporarily unavailable."
        state["error_code"] = "fundamentals_temporarily_unavailable"
        return state
    except SECError:
        state["error"] = "Fundamental data request failed."
        state["error_code"] = "fundamentals_failure"
        return state
    except Exception:
        state["error"] = "Fundamental data request failed."
        state["error_code"] = "fundamentals_failure"
        return state

    normalized = normalize_sec_fundamentals(fundamentals)
    (
        state["sec_fundamentals"],
        state["sec_fundamentals_history"],
    ) = _serialize_sec_fundamentals(fundamentals)
    filing_dates = [
        value.provenance.end
        for value in fundamentals.latest.values()
        if value.provenance is not None
    ]
    fundamentals_as_of = max(filing_dates) if filing_dates else None
    state["data_status"] = {
        "fundamentals": {
            "available": True,
            "source": "sec",
            "as_of": fundamentals_as_of,
        },
        "market_data": {
            "available": False,
            "source": "yahoo",
            "as_of": None,
            "reason": "market_data_unavailable",
        },
        "price_dependent_analysis": {
            "status": "incomplete",
            "reason": "current_market_price_unavailable",
        },
    }
    state["raw_data"] = {}
    state["earnings_data"] = []
    state["earnings_streak"] = None
    state["next_earnings_date"] = None
    state["data_source"] = "sec"

    try:
        async with YahooFinanceClient() as client:
            raw_task = client.get_company_data(ticker, years=1)
            next_earnings_task = client.get_next_earnings_date(ticker)
            raw_data, next_earnings = await asyncio.gather(
                raw_task, next_earnings_task
            )
            earnings = client.normalize_earnings(raw_data)
            streak = client.get_earnings_streak(earnings)

        info = raw_data.get("info", {})

        def market_number(key):
            value = info.get(key)
            if not isinstance(value, (int, float)):
                return None
            value = float(value)
            return value if value == value and value not in {float("inf"), float("-inf")} else None

        current_price = market_number("regularMarketPrice")
        if current_price is None:
            current_price = market_number("currentPrice")
        normalized.update(
            {
                "current_price": current_price,
                "market_cap": market_number("marketCap"),
                "beta": market_number("beta"),
                "dividend_yield": market_number("dividendYield"),
                "insider_ownership": market_number("heldPercentInsiders"),
                "institutional_ownership": market_number("heldPercentInstitutions"),
            }
        )
        if current_price is not None:
            eps = normalized.get("eps")
            book_value = normalized.get("book_value_per_share")
            shares = normalized.get("shares_outstanding")
            revenue = normalized.get("revenue")
            normalized["pe_ratio"] = current_price / eps if eps not in (None, 0) else None
            normalized["pb_ratio"] = (
                current_price / book_value if book_value not in (None, 0) else None
            )
            normalized["market_cap"] = normalized["market_cap"] or (
                current_price * shares if shares not in (None, 0) else None
            )
            normalized["ps_ratio"] = (
                normalized["market_cap"] / revenue
                if normalized["market_cap"] is not None and revenue not in (None, 0)
                else None
            )
        company_name = info.get("longName") or info.get("shortName")
        if company_name:
            normalized["company_name"] = company_name
        if info.get("sector"):
            normalized["sector"] = info["sector"]
        if info.get("industry"):
            normalized["industry"] = info["industry"]
        state["raw_data"] = raw_data
        state["earnings_data"] = earnings
        state["earnings_streak"] = streak
        state["next_earnings_date"] = next_earnings
        state["data_source"] = "sec+yahoo"
        state["data_status"]["market_data"] = {
            "available": current_price is not None,
            "source": "yahoo",
            "as_of": raw_data.get("fetched_at"),
            "reason": None if current_price is not None else "current_market_price_unavailable",
        }
        valuation_inputs = {
            "current_price": current_price,
            "eps": normalized.get("eps"),
            "book_value_per_share": normalized.get("book_value_per_share"),
            "shares_outstanding": normalized.get("shares_outstanding"),
            "revenue": normalized.get("revenue"),
        }
        missing_valuation_inputs = [
            name for name, value in valuation_inputs.items() if value is None
        ]
        valuation_outputs = {
            "pe_ratio": normalized.get("pe_ratio"),
            "pb_ratio": normalized.get("pb_ratio"),
            "ps_ratio": normalized.get("ps_ratio"),
        }
        missing_valuation_outputs = [
            name for name, value in valuation_outputs.items() if value is None
        ]
        state["data_status"]["price_dependent_analysis"] = {
            "status": (
                "incomplete"
                if missing_valuation_inputs or missing_valuation_outputs
                else "complete"
            ),
            "reason": (
                "missing_inputs:" + ",".join(missing_valuation_inputs)
                if missing_valuation_inputs
                else "missing_outputs:" + ",".join(missing_valuation_outputs)
                if missing_valuation_outputs
                else None
            ),
        }
    except YahooFinanceError as yahoo_error:
        error_code = _classify_yahoo_error(yahoo_error, ticker)
        state["data_status"]["market_data"]["reason"] = error_code
    except Exception:
        state["data_status"]["market_data"]["reason"] = "market_data_failure"

    state["normalized_data"] = normalized

    try:
        news_client = NewsClient()
        if await news_client.should_use_tool(ticker, normalized):
            state["news_sentiment"] = await news_client.get_sentiment(
                ticker,
                company_name=normalized["company_name"],
                days=7,
            )
        else:
            state["news_sentiment"] = None
    except Exception:
        state["news_sentiment"] = None

    try:
        rag_retriever = RAGRetriever()
        rag_context = await rag_retriever.retrieve_earnings_context(ticker, quarters=4)
        state["rag_context"] = rag_context if rag_context else None
    except Exception:
        state["rag_context"] = None

    return state


async def temporal_analysis_node(state: GOATState) -> GOATState:
    """
    Node: Calculate moat strength/trend based on time period.
    """
    if state.get("error"):
        return state
    
    analyzer = TemporalAnalyzer()
    time_period = state["time_period"]
    financials = state.get("normalized_data", {})
    
    moat_result = analyzer.calculate_moat_from_period(time_period, financials)
    
    state["temporal_results"] = {
        "moat_trend": moat_result["label"],
        "moat_score": moat_result["score"],
        "moat_observations": moat_result.get("observations", []),
    }
    
    return state


# Agent registry: (key, class) — single source of truth for all 5 GOATs
AGENT_REGISTRY = [
    ("buffett", BuffettAgent),
    ("lynch", LynchAgent),
    ("graham", GrahamAgent),
    ("munger", MungerAgent),
    ("dalio", DalioAgent),
]

_AGENT_REQUIRED_METRICS = {
    "lynch": (
        "pe_ratio",
        "earnings_growth",
        "institutional_ownership",
        "debt_to_equity",
        "cash_and_equivalents",
    ),
    "graham": (
        "pe_ratio",
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
    ),
    "munger": (
        "gross_margin",
        "operating_margin",
        "roic",
        "insider_ownership",
        "revenue_growth_5y",
        "capex_to_revenue",
        "debt_to_equity",
        "margin_trend",
    ),
    "dalio": (
        "beta",
        "volatility_252d",
        "debt_to_equity",
        "interest_coverage",
        "correlation_sp500",
        "revenue_cyclicality",
        "debt_growth_3y",
        "expected_return",
    ),
}


async def run_agents_node(state: GOATState, config: RunnableConfig = None) -> GOATState:
    """
    Node: Run all GOAT agents in parallel.
    
    Each agent is wrapped in a RunnableLambda with a descriptive run_name
    so the LangChain instrumentor creates a named span per agent.
    ChatOpenAI calls inside each agent inherit parent_run_id via config,
    nesting them under the agent span in Arize.
    
    Args:
        config: LangGraph RunnableConfig — threaded to each agent's LLM call
                so spans nest under the main chain trace in Arize.
    """
    if state.get("error"):
        return state
    
    ticker = state["ticker"]
    financials = state["normalized_data"] or {}
    selected = state["selected_agents"]
    earnings_data = state.get("earnings_data") or []
    earnings_streak = state.get("earnings_streak") or {}
    
    # Build agents: single loop handles both LLM and rule-based modes
    has_llm = bool(os.getenv("OPENAI_API_KEY"))
    all_agents = {
        name: cls(
            llm_client=get_llm_client(getattr(cls, "model_preference", "gpt-4o-mini"))
            if has_llm else None
        )
        for name, cls in AGENT_REGISTRY
    }
    
    # Filter to selected agents (or use all)
    agents = {k: v for k, v in all_agents.items() if k in selected} if selected else all_agents
    
    # Wrap each agent in a RunnableLambda so the LangChain instrumentor
    # creates a named span. Config flows through automatically, linking
    # the ChatOpenAI child span via parent_run_id.
    tasks = []
    for name, agent in agents.items():
        unavailable = [
            metric
            for metric in _AGENT_REQUIRED_METRICS.get(name, ())
            if financials.get(metric) is None
        ]
        if unavailable:
            async def _incomplete(
                _, *, _agent=agent, _missing=unavailable, config=None
            ):
                return {
                    "agent": _agent.name,
                    "style": _agent.style,
                    "ticker": ticker,
                    "score": None,
                    "verdict": None,
                    "insights": [
                        "Analysis incomplete because required financial or market data is unavailable."
                    ],
                    "concerns": [],
                    "status": "incomplete",
                    "unavailable_metrics": _missing,
                }

            runnable = RunnableLambda(_incomplete).with_config(
                {"run_name": f"agent.{name}"}
            )
            tasks.append(
                runnable.ainvoke({"ticker": ticker, "agent": name}, config=config)
            )
            continue

        async def _analyze(_, *, _agent=agent, config=None):
            return await _agent.analyze(
                ticker, financials,
                earnings_data=earnings_data,
                earnings_streak=earnings_streak,
                config=config,
            )
        
        runnable = RunnableLambda(_analyze).with_config({"run_name": f"agent.{name}"})
        tasks.append(runnable.ainvoke({"ticker": ticker, "agent": name}, config=config))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Preserve every requested agent. A failed agent is explicit and cannot
    # disappear from the consensus denominator.
    state["agent_results"] = []
    for (name, agent), result in zip(agents.items(), results):
        if isinstance(result, Exception):
            state["agent_results"].append(
                {
                    "agent": agent.name,
                    "style": agent.style,
                    "ticker": ticker,
                    "score": None,
                    "verdict": None,
                    "insights": ["Analysis unavailable because the agent failed."],
                    "concerns": [],
                    "status": "incomplete",
                    "unavailable_metrics": ["agent_execution"],
                }
            )
        else:
            try:
                validated_result = AgentResult.model_validate(result)
                if (
                    validated_result.agent != agent.name
                    or validated_result.style != agent.style
                ):
                    raise ValueError("Agent output identity mismatch")
            except (TypeError, ValueError, ValidationError):
                state["agent_results"].append(
                    {
                        "agent": agent.name,
                        "style": agent.style,
                        "ticker": ticker,
                        "score": None,
                        "verdict": None,
                        "insights": [],
                        "concerns": ["Analysis output was unavailable."],
                        "status": "incomplete",
                        "unavailable_metrics": ["agent_execution_failure"],
                    }
                )
            else:
                state["agent_results"].append(validated_result.model_dump())
    
    return state


async def synthesize_node(state: GOATState, config: RunnableConfig = None) -> GOATState:
    """
    Node: Synthesize results and calculate consensus.
    
    Args:
        config: LangGraph RunnableConfig — threaded to the consensus LLM call
                so its span nests under the main chain trace in Arize.
    """
    if state.get("error"):
        return state
    
    agent_results = state["agent_results"]
    
    if not agent_results:
        state["error"] = "No agent results to synthesize"
        return state
    
    complete_results = [
        result for result in agent_results if result.get("status", "complete") == "complete"
    ]
    completed_analyses = len(complete_results)
    requested_analyses = len(agent_results)

    # Convert only complete analyses to StrategyResult. Incomplete placeholder
    # scores must not become a plausible neutral consensus.
    strategy_results = []
    for result in complete_results:
        strategy_results.append(StrategyResult(
            agent_name=result["agent"],
            verdict=Verdict(result["verdict"]),
            confidence=0.8,  # Placeholder
            key_insights=result.get("insights", []),
            concerns=result.get("concerns", []),
            moat_assessment="",
            score=result.get("score", 0),
        ))
    
    llm = get_llm_client() if os.getenv("OPENAI_API_KEY") else None

    if not strategy_results:
        state["consensus"] = {
            "status": "incomplete",
            "verdict": "incomplete",
            "agreement_score": None,
            "consensus_points": [],
            "divergence_points": [
                "Consensus unavailable because every selected analysis is incomplete."
            ],
            "completed_analyses": completed_analyses,
            "requested_analyses": requested_analyses,
        }
    elif completed_analyses == 1 and requested_analyses > 1:
        state["consensus"] = {
            "status": "partial",
            "verdict": "incomplete",
            "agreement_score": None,
            "consensus_points": [],
            "divergence_points": [
                "Consensus unavailable because only one selected analysis is complete."
            ],
            "completed_analyses": completed_analyses,
            "requested_analyses": requested_analyses,
        }
    elif completed_analyses == requested_analyses == 1:
        state["consensus"] = {
            "status": "complete",
            "verdict": complete_results[0]["verdict"],
            "agreement_score": None,
            "consensus_points": [],
            "divergence_points": [
                "Single-strategy result; multi-agent agreement is unavailable."
            ],
            "completed_analyses": 1,
            "requested_analyses": 1,
        }
    elif llm:
        # Wrap in RunnableLambda so the LangChain instrumentor creates a
        # named "consensus.synthesis" span, with the ChatOpenAI call nested
        # underneath via parent_run_id from config.
        async def _consensus_fn(_, config=None):
            return await calculate_consensus_with_llm(
                strategy_results, llm, state["ticker"], config=config,
            )
        
        consensus = await (
            RunnableLambda(_consensus_fn)
            .with_config({"run_name": "consensus.synthesis"})
            .ainvoke({"ticker": state["ticker"]}, config=config)
        )
        state["consensus"] = {
            "status": "partial" if len(complete_results) < len(agent_results) else "complete",
            "verdict": consensus.consensus_verdict.value,
            "agreement_score": consensus.agreement_score,
            "consensus_points": consensus.consensus_points,
            "divergence_points": consensus.divergence_points,
            "completed_analyses": completed_analyses,
            "requested_analyses": requested_analyses,
        }
    else:
        consensus = calculate_consensus(strategy_results)
        state["consensus"] = {
            "status": "partial" if len(complete_results) < len(agent_results) else "complete",
            "verdict": consensus.consensus_verdict.value,
            "agreement_score": consensus.agreement_score,
            "consensus_points": consensus.consensus_points,
            "divergence_points": consensus.divergence_points,
            "completed_analyses": completed_analyses,
            "requested_analyses": requested_analyses,
        }
    
    # Build comparison table
    comparison_table = {
        "agents": [],
    }
    for result in agent_results:
        comparison_table["agents"].append({
            "name": result["agent"],
            "style": result.get("style", ""),
            "verdict": result["verdict"],
            "score": result.get("score", 0),
            "key_metric": result.get("insights", [""])[0] if result.get("insights") else "",
        })
    
    # Build final report from the normalized authoritative/enrichment contract.
    normalized = state.get("normalized_data") or {}
    
    # next_earnings_date already fetched in fetch_data_node — no extra call needed
    state["final_report"] = {
        "ticker": state["ticker"],
        "company_name": normalized.get("company_name") or state["ticker"],
        "sector": normalized.get("sector") or "Unknown",
        "data_status": state.get("data_status") or {},
        "fundamentals": state.get("sec_fundamentals") or {},
        "fundamentals_history": state.get("sec_fundamentals_history") or {},
        "consensus": state["consensus"],
        "agent_results": agent_results,
        "temporal_results": state.get("temporal_results", {}),
        "comparison_table": comparison_table,
        "earnings_data": state.get("earnings_data", []),
        "earnings_streak": state.get("earnings_streak", {}),
        "next_earnings_date": state.get("next_earnings_date"),
    }
    
    return state


def build_goat_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for GOAT analysis.
    
    Flow:
    1. Fetch SEC fundamentals and optional Yahoo market data
    2. Temporal Analysis
    3. Run GOAT Agents (parallel)
    4. Synthesize Results
    """
    workflow = StateGraph(GOATState)
    
    # Add nodes
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("temporal_analysis", temporal_analysis_node)
    workflow.add_node("run_agents", run_agents_node)
    workflow.add_node("synthesize", synthesize_node)
    
    # Define edges
    workflow.set_entry_point("fetch_data")
    workflow.add_edge("fetch_data", "temporal_analysis")
    workflow.add_edge("temporal_analysis", "run_agents")
    workflow.add_edge("run_agents", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# ============================================================================
# FastAPI Application
# ============================================================================

# Global workflow
goat_workflow = None

# The public canary has a fixed ticker, so share its complete Yahoo result
# instead of letting every monitor request repeat provider work independently.
_MARKET_DATA_HEALTH_TTL = 60
_market_data_health_cache = {}
_market_data_health_inflight = {}
_market_data_health_guard = threading.Lock()
_FUNDAMENTALS_HEALTH_TTL = 60
_fundamentals_health_cache = {}
_fundamentals_health_inflight = {}
_fundamentals_health_guard = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global goat_workflow
    goat_workflow = build_goat_workflow()
    yield


app = FastAPI(
    title="GOATlens",
    description="Multi-Agent Investment Analysis using Legendary Investor Mental Models",
    version="0.1.0",
    lifespan=lifespan,
)

_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'sha256-iNFs4AfCuJmTG2E8xVg+E6XrT83ESwpOwahrfwOnfys=' "
    "https://unpkg.com https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path == "/":
        response.headers["Content-Security-Policy"] = _CONTENT_SECURITY_POLICY
    return response


@app.get("/")
async def root():
    """Serve the frontend (no-cache so changes appear immediately)."""
    return FileResponse(
        "../frontend/index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/health")
@app.get("/ping")  # Alias for keep-alive services
async def health():
    """
    Health check endpoint.

    Also serves as a keep-alive endpoint for external monitoring services
    (UptimeRobot, cron-job.org, etc.) to prevent Render from sleeping.
    HEAD is supported because UptimeRobot defaults to HEAD requests.
    """
    return {
        "status": "healthy",
        "service": "goatlens",
        "release": os.getenv("RENDER_GIT_COMMIT", "local"),
    }


@app.head("/health", include_in_schema=False)
@app.head("/ping", include_in_schema=False)
async def health_head():
    return await health()


async def _fetch_market_data_health():
    """Fetch the fixed-symbol market-data snapshot without optional services."""
    ticker = "AAPL"
    state: GOATState = {
        "ticker": ticker,
        "time_period": "1y",
        "selected_agents": [],
        "raw_data": None,
        "normalized_data": None,
        "data_source": None,
        "data_status": None,
        "sec_fundamentals": None,
        "sec_fundamentals_history": None,
        "earnings_data": None,
        "earnings_streak": None,
        "next_earnings_date": None,
        "news_sentiment": None,
        "rag_context": None,
        "temporal_results": None,
        "agent_results": [],
        "consensus": None,
        "final_report": None,
        "error": None,
        "error_code": None,
        "market_data_only": True,
    }
    return await fetch_data_node(state)


def _cleanup_market_data_health_task(inflight_key, completed_task):
    with _market_data_health_guard:
        if _market_data_health_inflight.get(inflight_key) is completed_task:
            _market_data_health_inflight.pop(inflight_key, None)

    if not completed_task.cancelled():
        completed_task.exception()


async def _market_data_health_snapshot():
    """Cache and coalesce the fixed-symbol Yahoo market-data request."""
    loop = asyncio.get_running_loop()
    provider_key = (YahooFinanceClient,)
    now = time.monotonic()

    with _market_data_health_guard:
        cached = _market_data_health_cache.get(provider_key)
        if cached and now - cached[0] < _MARKET_DATA_HEALTH_TTL:
            return cached[1]

        inflight_key = (loop, provider_key)
        task = _market_data_health_inflight.get(inflight_key)
        if task is None:
            task = loop.create_task(_fetch_market_data_health())
            _market_data_health_inflight[inflight_key] = task
            task.add_done_callback(
                lambda completed: _cleanup_market_data_health_task(
                    inflight_key, completed
                )
            )

    result = await asyncio.shield(task)
    with _market_data_health_guard:
        _market_data_health_cache[provider_key] = (time.monotonic(), result)
    return result


async def _fetch_fundamentals_health():
    async with SECFundamentalsClient(
        user_agent=os.getenv("SEC_USER_AGENT", "")
    ) as sec_client:
        return await sec_client.get_fundamentals("AAPL")


class _FundamentalsHealthUnexpectedError(Exception):
    """Internal sanitized marker for an unexpected canary failure."""


def _raise_fundamentals_health_failure(category: str) -> NoReturn:
    failure_types = {
        "configuration": SECConfigurationError,
        "invalid_response": SECInvalidResponseError,
        "unavailable": SECUnavailableError,
        "failure": SECError,
        "unexpected": _FundamentalsHealthUnexpectedError,
    }
    raise failure_types[category]("SEC fundamentals canary failed.") from None


def _cleanup_fundamentals_health_task(inflight_key, completed_task):
    with _fundamentals_health_guard:
        if _fundamentals_health_inflight.get(inflight_key) is completed_task:
            _fundamentals_health_inflight.pop(inflight_key, None)
    if not completed_task.cancelled():
        completed_task.exception()


async def _fundamentals_health_snapshot():
    """Cache and coalesce the fixed-symbol SEC canary request."""
    loop = asyncio.get_running_loop()
    provider_key = (SECFundamentalsClient, os.getenv("SEC_USER_AGENT", ""))
    now = time.monotonic()
    with _fundamentals_health_guard:
        cached = _fundamentals_health_cache.get(provider_key)
        if cached and now - cached[0] < _FUNDAMENTALS_HEALTH_TTL:
            if cached[1] == "success":
                return cached[2]
            _raise_fundamentals_health_failure(cached[2])
        inflight_key = (loop, provider_key)
        task = _fundamentals_health_inflight.get(inflight_key)
        if task is None:
            task = loop.create_task(_fetch_fundamentals_health())
            _fundamentals_health_inflight[inflight_key] = task
            task.add_done_callback(
                lambda completed: _cleanup_fundamentals_health_task(
                    inflight_key, completed
                )
            )
    failure_category = None
    result = None
    try:
        result = await asyncio.shield(task)
    except SECConfigurationError:
        failure_category = "configuration"
    except SECInvalidResponseError:
        failure_category = "invalid_response"
    except (SECRateLimitError, SECUnavailableError):
        failure_category = "unavailable"
    except SECError:
        failure_category = "failure"
    except Exception:
        failure_category = "unexpected"
    with _fundamentals_health_guard:
        _fundamentals_health_cache[provider_key] = (
            time.monotonic(),
            "failure" if failure_category else "success",
            failure_category or result,
        )
    if failure_category:
        if failure_category == "unexpected":
            print("[Canary] SEC fundamentals health check failed unexpectedly.")
        _raise_fundamentals_health_failure(failure_category)
    return result


@app.get("/api/fundamentals-health")
async def fundamentals_health():
    """Exercise the fixed-symbol authoritative SEC fundamentals path once."""
    try:
        fundamentals = await _fundamentals_health_snapshot()
    except SECConfigurationError:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "fundamentals_configuration_error",
                "message": "SEC fundamental data is not configured.",
                "retryable": False,
            },
        )
    except SECInvalidResponseError:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "fundamentals_invalid_response",
                "message": "SEC returned invalid fundamental data.",
                "retryable": False,
            },
        )
    except SECUnavailableError:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "fundamentals_temporarily_unavailable",
                "message": "SEC fundamental data is temporarily unavailable. Please retry.",
                "retryable": True,
            },
            headers={"Retry-After": "60"},
        )
    except SECError:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "fundamentals_failure",
                "message": "SEC fundamental data request failed.",
                "retryable": False,
            },
        )
    except _FundamentalsHealthUnexpectedError:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "fundamentals_failure",
                "message": "SEC fundamental data request failed.",
                "retryable": False,
            },
        ) from None

    filing_dates = [
        value.provenance.end
        for value in fundamentals.latest.values()
        if value.provenance is not None
    ]
    return {
        "status": "healthy",
        "ticker": "AAPL",
        "source": "sec",
        "as_of": max(filing_dates) if filing_dates else None,
    }


@app.get("/api/market-data-health")
async def market_data_health():
    """Exercise the fixed-symbol market-data path without optional services."""
    ticker = "AAPL"
    try:
        result = await _market_data_health_snapshot()
    except Exception:
        print("[Canary] Market-data health check failed unexpectedly.")
        raise HTTPException(
            status_code=502,
            detail={
                "code": "market_data_failure",
                "message": "Market data request failed.",
                "retryable": False,
            },
        ) from None
    if result.get("error"):
        if result.get("error_code") == "ticker_not_found":
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "ticker_not_found",
                    "message": f"No market data found for ticker {ticker}.",
                    "retryable": False,
                },
            )
        if result.get("error_code") == "market_data_temporarily_unavailable":
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "market_data_temporarily_unavailable",
                    "message": "Market data is temporarily unavailable. Please retry.",
                    "retryable": True,
                },
                headers={"Retry-After": "60"},
            )
        raise HTTPException(
            status_code=502,
            detail={
                "code": "market_data_failure",
                "message": "Market data request failed.",
                "retryable": False,
            },
        )

    return {
        "status": "healthy",
        "ticker": ticker,
        "source": result["data_source"],
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_company(request: AnalysisRequest):
    """
    Run full GOAT analysis on a company.
    
    This endpoint:
    1. Fetches authoritative annual SEC fundamentals and best-effort market data
    2. Performs a current-period moat assessment
    3. Runs the requested GOAT agents in parallel
    4. Synthesizes complete results and labels partial coverage
    
    Returns comprehensive investment thesis.
    """
    global goat_workflow
    
    if not goat_workflow:
        raise HTTPException(status_code=503, detail="Workflow not initialized")
    
    time_period = request.time_period or "1y"
    
    # Initialize state
    initial_state: GOATState = {
        "ticker": request.ticker.upper(),
        "time_period": time_period,
        "selected_agents": request.agents or [],
        "raw_data": None,
        "normalized_data": None,
        "data_source": None,
        "data_status": None,
        "sec_fundamentals": None,
        "sec_fundamentals_history": None,
        "earnings_data": None,
        "earnings_streak": None,
        "next_earnings_date": None,
        "news_sentiment": None,
        "rag_context": None,
        "temporal_results": None,
        "agent_results": [],
        "consensus": None,
        "final_report": None,
        "error": None,
        "error_code": None,
        "market_data_only": False,
    }
    
    # Run workflow
    try:
        final_state = await goat_workflow.ainvoke(initial_state)
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis failed")
    
    # Check for errors
    if final_state.get("error"):
        if final_state.get("error_code") == "ticker_not_found":
            ticker = request.ticker.strip().upper()
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "ticker_not_found",
                    "message": f"No SEC issuer found for ticker {ticker}.",
                    "retryable": False,
                },
            )
        if final_state.get("error_code") == "market_data_temporarily_unavailable":
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "market_data_temporarily_unavailable",
                    "message": "Market data is temporarily unavailable. Please retry.",
                    "retryable": True,
                },
                headers={"Retry-After": "60"},
            )
        if final_state.get("error_code") == "fundamentals_temporarily_unavailable":
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "fundamentals_temporarily_unavailable",
                    "message": "SEC fundamental data is temporarily unavailable. Please retry.",
                    "retryable": True,
                },
                headers={"Retry-After": "60"},
            )
        if final_state.get("error_code") == "fundamentals_configuration_error":
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "fundamentals_configuration_error",
                    "message": "SEC fundamental data is not configured.",
                    "retryable": False,
                },
            )
        if final_state.get("error_code") == "fundamentals_invalid_response":
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "fundamentals_invalid_response",
                    "message": "SEC returned invalid fundamental data.",
                    "retryable": False,
                },
            )
        if final_state.get("error_code") == "fundamentals_failure":
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "fundamentals_failure",
                    "message": "SEC fundamental data request failed.",
                    "retryable": False,
                },
            )
        if final_state.get("error_code") == "market_data_failure":
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "market_data_failure",
                    "message": "Market data request failed.",
                    "retryable": False,
                },
            )
        raise HTTPException(status_code=400, detail="Analysis request failed")
    
    # Build response
    report = final_state["final_report"]
    consensus = final_state["consensus"]
    temporal = final_state.get("temporal_results", {})
    
    # Build earnings response
    earnings_history = [
        EarningsQuarter(**eq) for eq in report.get("earnings_data", [])
    ]
    streak_data = report.get("earnings_streak", {})
    earnings_streak_obj = EarningsStreak(**streak_data) if streak_data and streak_data.get("total", 0) > 0 else None
    
    # Do not persist incomplete or scoreless consensus records as neutral advice.
    # The current consensus contract has no investment-score field, so there is
    # intentionally no MemoryStore write here.
    
    return AnalysisResponse(
        ticker=report["ticker"],
        company_name=report["company_name"],
        sector=report["sector"],
        data_status=report["data_status"],
        fundamentals=report["fundamentals"],
        fundamentals_history=report["fundamentals_history"],
        consensus_status=consensus.get("status", "complete"),
        consensus_verdict=consensus["verdict"],
        agreement_score=consensus["agreement_score"],
        completed_analyses=consensus["completed_analyses"],
        requested_analyses=consensus["requested_analyses"],
        consensus_points=consensus["consensus_points"],
        divergence_points=consensus["divergence_points"],
        agent_results=[
            AgentResult(
                agent=r["agent"],
                style=r.get("style", ""),
                verdict=r["verdict"],
                score=r.get("score", 0),
                insights=r.get("insights", []),
                concerns=r.get("concerns", []),
                status=r.get("status", "complete"),
                unavailable_metrics=r.get("unavailable_metrics", []),
            )
            for r in report["agent_results"]
        ],
        moat_trend=temporal.get("moat_trend", "unknown"),
        comparison_table=report["comparison_table"],
        earnings_history=earnings_history,
        earnings_streak=earnings_streak_obj,
        next_earnings_date=report.get("next_earnings_date"),
    )


@app.get("/api/agents")
async def list_agents():
    """List available GOAT agents."""
    return {
        "agents": [
            {
                "id": "buffett",
                "name": "Warren Buffett",
                "style": "Value Investing with Quality Focus",
                "key_metrics": ["ROE", "Profit Margins", "Debt Levels", "Owner Earnings"],
                "moat_focus": "Durable competitive advantage",
            },
            {
                "id": "lynch",
                "name": "Peter Lynch",
                "style": "Growth at Reasonable Price (GARP)",
                "key_metrics": ["PEG Ratio", "Earnings Growth", "Institutional Ownership"],
                "moat_focus": "Growth at reasonable price",
            },
            {
                "id": "graham",
                "name": "Benjamin Graham",
                "style": "Deep Value / Margin of Safety",
                "key_metrics": ["P/E", "P/B", "Current Ratio", "Dividend History"],
                "moat_focus": "Margin of safety",
            },
            {
                "id": "munger",
                "name": "Charlie Munger",
                "style": "Quality-Focused Mental Models",
                "key_metrics": ["Business Quality", "Management", "Pricing Power"],
                "moat_focus": "Mental models, avoid mistakes",
            },
            {
                "id": "dalio",
                "name": "Ray Dalio",
                "style": "Macro & Risk Parity",
                "key_metrics": ["Macro Positioning", "Debt Cycles", "Diversification"],
                "moat_focus": "Risk-adjusted returns",
            },
        ]
    }


@app.get("/api/earnings/{ticker}", response_model=EarningsInsightsResponse)
async def get_earnings(ticker: str):
    """
    Get earnings history for a ticker (standalone endpoint).
    
    Returns last 8 quarters of EPS actual vs. consensus estimate,
    beat/miss classification, streak summary, price reactions,
    per-quarter reaction insights, forward guidance summary, and the next
    upcoming earnings date. This endpoint uses Yahoo market and analyst data.
    """
    ticker = _normalize_path_ticker(ticker)
    
    try:
        async with YahooFinanceClient() as client:
            # Fetch all Yahoo data in parallel
            raw_task = client.get_company_data(ticker, years=1)
            price_task = client.get_price_history(ticker, period="2y")
            analyst_task = client.get_analyst_reactions(ticker)
            revenue_task = client.get_quarterly_revenue(ticker)
            estimates_task = client.get_forward_estimates(ticker)
            next_earnings_task = client.get_next_earnings_date(ticker)

            raw_data, price_history, analyst_reactions, quarterly_revenue, \
                forward_estimates, next_earnings = await asyncio.gather(
                    raw_task, price_task, analyst_task, revenue_task,
                    estimates_task, next_earnings_task,
                )

            # Normalize earnings & streak
            earnings = client.normalize_earnings(raw_data)
            streak = client.get_earnings_streak(earnings)

            # Enrich with price reactions
            earnings = client.calculate_earnings_reactions(price_history, earnings)

            # Build per-quarter reaction insights from Yahoo data.
            reaction_insights = client.build_earnings_insights(
                earnings, price_history, analyst_reactions, quarterly_revenue,
            )

            # Build forward guidance summary
            normalized = client.normalize_data(raw_data)
            forward_guidance = client.build_forward_guidance_summary(
                forward_estimates, current_price=normalized.current_price,
            )
    except YahooFinanceError:
        raise HTTPException(status_code=400, detail="Earnings data request failed")
    except Exception:
        raise HTTPException(status_code=500, detail="Earnings fetch failed")

    return {
        "ticker": ticker,
        "earnings_history": earnings,
        "earnings_streak": streak,
        "reaction_insights": reaction_insights,
        "forward_guidance": forward_guidance,
        "next_earnings_date": next_earnings,
    }


@app.get("/api/price-history/{ticker}", response_model=PriceHistoryResponse)
async def get_price_history(ticker: str, period: str = "2y"):
    """
    Get historical price data with earnings markers for charting.
    
    Returns daily OHLCV prices and earnings date markers (with
    price reaction calculations) for overlay on a chart.
    
    Query params:
        period: 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd (default: 2y)
    """
    ticker = _normalize_path_ticker(ticker)
    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd"}
    if period not in valid_periods:
        period = "2y"
    
    try:
        async with YahooFinanceClient() as client:
            # Fetch price history and earnings in parallel
            price_task = client.get_price_history(ticker, period=period)
            data_task = client.get_company_data(ticker, years=1)
            next_earnings_task = client.get_next_earnings_date(ticker)
            
            prices, raw_data, next_earnings = await asyncio.gather(
                price_task, data_task, next_earnings_task
            )
            
            # Normalize earnings and calculate reactions
            earnings = client.normalize_earnings(raw_data)
            earnings_with_reactions = client.calculate_earnings_reactions(prices, earnings)
    except YahooFinanceError:
        raise HTTPException(status_code=400, detail="Price data request failed")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch price data")
    
    return PriceHistoryResponse(
        ticker=ticker,
        period=period,
        prices=[PricePoint(**p) for p in prices],
        earnings_markers=[
            EarningsMarker(
                quarter=e.get("quarter", ""),
                date=e.get("date", ""),
                eps_actual=e.get("eps_actual"),
                eps_estimate=e.get("eps_estimate"),
                surprise_pct=e.get("surprise_pct"),
                beat_miss=e.get("beat_miss", "unknown"),
                price_reaction_1d=e.get("price_reaction_1d"),
                price_reaction_5d=e.get("price_reaction_5d"),
            )
            for e in earnings_with_reactions
        ],
        next_earnings_date=next_earnings,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
