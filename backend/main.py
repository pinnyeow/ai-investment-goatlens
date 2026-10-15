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
from datetime import date, timedelta
from typing import Annotated, Dict, Any, List, Literal, NoReturn, Optional, Union
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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda, RunnableConfig
from typing_extensions import NotRequired, TypedDict

# Local imports
from agents import BuffettAgent, LynchAgent, GrahamAgent, MungerAgent, DalioAgent
from agents.incomplete import metric_is_available
from data_sources.yahoo_chart import (
    YahooChartClient,
    YahooChartError,
    YahooChartNotFound,
)
from data_sources.yahoo_ratios import YahooRatioError, YahooRatiosClient
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
from data_sources.news import NewsClient
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


class StrictAPIModel(BaseModel):
    """Fail-closed public boundary: no coercion and no unknown fields."""

    model_config = ConfigDict(strict=True, extra="forbid")


class AnalysisRequest(StrictAPIModel):
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


class AgentResult(StrictAPIModel):
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


class EarningsQuarter(StrictAPIModel):
    """Earnings data for a single quarter."""
    quarter: str = Field(description="Quarter label, e.g. Q4 2025")
    date: str = Field(description="Earnings report date YYYY-MM-DD")
    eps_actual: Optional[float] = Field(default=None, description="Reported EPS")
    eps_estimate: Optional[float] = Field(default=None, description="Consensus EPS estimate")
    eps_surprise: Optional[float] = Field(default=None, description="EPS surprise (actual - estimate)")
    surprise_pct: Optional[float] = Field(default=None, description="Surprise as percentage")
    beat_miss: str = Field(default="unknown", description="beat, miss, or inline")


class EarningsStreak(StrictAPIModel):
    """Summary of earnings beat/miss track record."""
    streak_type: str = Field(description="Current streak: beat, miss, inline, or unknown")
    streak_count: int = Field(description="Number of consecutive quarters in current streak")
    beats: int = Field(description="Total beats in history")
    misses: int = Field(description="Total misses in history")
    inline: int = Field(description="Total inline results")
    total: int = Field(description="Total quarters with data")
    summary: str = Field(description="Human-readable summary")


class PricePoint(StrictAPIModel):
    """Single day of price data."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class EarningsMarker(StrictAPIModel):
    """Earnings event marker for overlaying on a price chart."""
    quarter: str
    date: str
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    surprise_pct: Optional[float] = None
    beat_miss: str = "unknown"
    price_reaction_1d: Optional[float] = None
    price_reaction_5d: Optional[float] = None


class PriceHistoryResponse(StrictAPIModel):
    """Price history with earnings overlays for charting."""
    ticker: str
    period: str
    prices: List[PricePoint]
    earnings_markers: List[EarningsMarker]
    next_earnings_date: Optional[str] = None


class EarningsReactionInsight(StrictAPIModel):
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


class EstimateRange(StrictAPIModel):
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


class PriceTargetRange(StrictAPIModel):
    """Analyst price target range."""
    current: Optional[float] = None
    low: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    high: Optional[float] = None


class ForwardGuidanceSummary(StrictAPIModel):
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


class EarningsInsightsResponse(StrictAPIModel):
    ticker: str
    earnings_history: List[EarningsQuarter]
    earnings_streak: Optional[EarningsStreak]
    reaction_insights: List[EarningsReactionInsight]
    forward_guidance: Optional[ForwardGuidanceSummary]
    next_earnings_date: Optional[str]


class FactProvenanceResponse(StrictAPIModel):
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


class FundamentalFactResponse(StrictAPIModel):
    value: Optional[float]
    availability_reason: Optional[str]
    provenance: Optional[FactProvenanceResponse]


class SECMetricInputResponse(StrictAPIModel):
    kind: Literal["sec_fact"]
    role: str = Field(min_length=1)
    value: float
    provenance: FactProvenanceResponse


class MarketObservationInputResponse(StrictAPIModel):
    kind: Literal["market_observation"]
    role: str = Field(min_length=1)
    source: Literal["yahoo"]
    symbol: str = Field(min_length=1)
    value: float
    observed_at: str = Field(min_length=1)


class MarketSeriesInputResponse(StrictAPIModel):
    kind: Literal["market_series"]
    role: str = Field(min_length=1)
    source: Literal["yahoo"]
    symbol: str = Field(min_length=1)
    start: str = Field(min_length=1)
    end: str = Field(min_length=1)
    observation_count: int = Field(ge=1)


class ProviderMetricInputResponse(StrictAPIModel):
    kind: Literal["provider_metric"]
    role: str = Field(min_length=1)
    source: Literal["yahoo"]
    symbol: str = Field(min_length=1)
    provider_field: str = Field(min_length=1)
    value: float
    retrieved_at: str = Field(min_length=1)


MetricInputResponse = Annotated[
    Union[
        SECMetricInputResponse,
        MarketObservationInputResponse,
        MarketSeriesInputResponse,
        ProviderMetricInputResponse,
    ],
    Field(discriminator="kind"),
]


class MetricEvidenceResponse(StrictAPIModel):
    value: Optional[float]
    available: bool
    source: Literal["sec", "yahoo", "sec+yahoo"]
    as_of: Optional[str]
    formula: str
    formula_version: str
    inputs: List[MetricInputResponse] = Field(default_factory=list)
    observation_count: Optional[int] = Field(default=None, ge=0)
    availability_reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_availability_contract(self):
        if self.available:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("Available evidence requires a finite value")
            if not self.inputs:
                raise ValueError("Available evidence requires typed inputs")
            if self.availability_reason is not None:
                raise ValueError("Available evidence cannot have an unavailable reason")
        else:
            if self.value is not None:
                raise ValueError("Unavailable evidence cannot expose a value")
            if not self.availability_reason:
                raise ValueError("Unavailable evidence requires a reason")
        return self


class FundamentalsAvailabilityStatus(StrictAPIModel):
    available: bool
    source: Literal["sec"]
    as_of: Optional[str]


class MarketAvailabilityStatus(StrictAPIModel):
    available: bool
    source: Literal["yahoo"]
    as_of: Optional[str]
    reason: Optional[str] = None


class AnalysisCompletenessStatus(StrictAPIModel):
    status: Literal["complete", "incomplete"]
    reason: Optional[str]


class AnalysisDataStatus(StrictAPIModel):
    fundamentals: FundamentalsAvailabilityStatus
    market_data: MarketAvailabilityStatus
    normalized_ratios: MarketAvailabilityStatus
    price_dependent_analysis: AnalysisCompletenessStatus


class ComparisonAgentResponse(StrictAPIModel):
    name: str
    style: str
    verdict: Optional[
        Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    ]
    score: Optional[float] = Field(default=None, ge=-100, le=100)
    key_metric: str


class ComparisonTableResponse(StrictAPIModel):
    agents: List[ComparisonAgentResponse]


class AnalysisResponse(StrictAPIModel):
    """Complete analysis response."""
    ticker: str
    company_name: str
    sector: str
    data_status: AnalysisDataStatus
    fundamentals: Dict[str, FundamentalFactResponse]
    fundamentals_history: Dict[str, List[FundamentalFactResponse]]
    metric_evidence: Dict[str, MetricEvidenceResponse]
    
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
    metric_evidence: NotRequired[Optional[Dict[str, Dict[str, Any]]]]
    
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


def _provenance_payload(provenance):
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
        "inputs": [_provenance_payload(item) for item in provenance.inputs],
    }


def _serialize_sec_fundamentals(fundamentals):
    """Expose authoritative values and their complete filing provenance."""

    def value_payload(item):
        return {
            "value": item.value,
            "availability_reason": item.availability_reason,
            "provenance": _provenance_payload(item.provenance),
        }

    return (
        {metric: value_payload(item) for metric, item in fundamentals.latest.items()},
        {
            metric: [value_payload(item) for item in items]
            for metric, items in fundamentals.history.items()
        },
    )


_METRIC_EVIDENCE_CONTRACTS = {
    "roe": ("sec", "net_income_divided_by_equity"),
    "profit_margin": ("sec", "net_income_divided_by_revenue"),
    "debt_to_equity": ("sec", "funded_debt_divided_by_equity"),
    "free_cash_flow": ("sec", "operating_cash_flow_minus_capex"),
    "earnings_growth": ("sec", "three_year_net_income_cagr"),
    "earnings_turnaround": (
        "sec",
        "negative_or_zero_to_positive_three_year_net_income_transition",
    ),
    "cash_and_equivalents": ("sec", "reported"),
    "current_ratio": ("sec", "current_assets_divided_by_current_liabilities"),
    "current_assets": ("sec", "reported"),
    "total_liabilities": ("sec", "reported"),
    "shares_outstanding": ("sec", "reported_period_end"),
    "eps": ("sec", "reported_diluted_eps"),
    "book_value_per_share": ("sec", "equity_divided_by_period_end_shares"),
    "consecutive_dividend_years": (
        "sec",
        "consecutive_positive_common_dividend_years",
    ),
    "gross_margin": ("sec", "gross_profit_divided_by_revenue"),
    "operating_margin": ("sec", "operating_income_divided_by_revenue"),
    "roic": ("sec", "nopat_divided_by_average_invested_capital"),
    "revenue_growth_5y": ("sec", "five_year_revenue_cagr"),
    "capex_to_revenue": ("sec", "capex_divided_by_revenue"),
    "margin_trend": ("sec", "five_year_operating_margin_ols_slope"),
    "debt_growth_3y": ("sec", "three_year_funded_debt_cagr"),
    "current_price": ("yahoo", "provider_regular_market_price"),
    "beta": (
        "yahoo",
        "aligned_daily_return_sample_covariance_divided_by_benchmark_variance",
    ),
    "volatility_252d": ("yahoo", "annualized_log_return_volatility_252_sessions"),
    "correlation_sp500": ("yahoo", "aligned_daily_return_pearson_correlation"),
    "realized_total_return_252d": (
        "yahoo",
        "adjusted_close_total_return_252_sessions",
    ),
    "pe_ratio": ("sec+yahoo", "current_price_divided_by_diluted_eps"),
    "pb_ratio": (
        "sec+yahoo",
        "current_price_divided_by_book_value_per_share",
    ),
    "dividend_yield": (
        "sec+yahoo",
        "annual_common_dividend_per_share_divided_by_current_price",
    ),
}


def _build_metric_evidence(normalized, snapshot=None, ratio_snapshot=None):
    evidence = {}
    market_as_of = (
        getattr(snapshot, "current_price_as_of", None) if snapshot is not None else None
    )
    market_window_start = (
        getattr(snapshot, "metric_session_start", None) if snapshot is not None else None
    )
    market_window_end = (
        getattr(snapshot, "metric_session_end", None) if snapshot is not None else None
    )

    def sec_inputs(metric):
        return [
            {
                "kind": "sec_fact",
                "role": item.role,
                "value": float(item.value),
                "provenance": _provenance_payload(item.provenance),
            }
            for item in metric.inputs
        ]

    def market_observation(value):
        if market_as_of is None or not metric_is_available(value):
            return []
        return [{
            "kind": "market_observation",
            "role": "price.current",
            "source": "yahoo",
            "symbol": normalized.get("ticker"),
            "value": float(value),
            "observed_at": market_as_of,
        }]

    def market_series(
        role, symbol, observations, start=market_window_start, end=market_window_end
    ):
        if (
            start is None
            or end is None
            or type(observations) is not int
            or observations < 1
        ):
            return []
        return [{
            "kind": "market_series",
            "role": role,
            "source": "yahoo",
            "symbol": symbol,
            "start": start,
            "end": end,
            "observation_count": observations,
        }]

    sec_metrics = getattr(normalized, "metrics", {})
    combined_sec_metric = {
        "pe_ratio": "eps",
        "pb_ratio": "book_value_per_share",
        "dividend_yield": "dividend_per_share",
    }
    for metric, (source, formula) in _METRIC_EVIDENCE_CONTRACTS.items():
        value = normalized.get(metric)
        observations = None
        inputs = []
        reason = "required_inputs_unavailable"
        ratio_value = (
            ratio_snapshot.values.get(metric)
            if ratio_snapshot is not None
            else None
        )
        if metric_is_available(ratio_value):
            value = ratio_value
            source = "yahoo"
            formula = "provider_normalized_ratio"
            as_of = None
            observations = 1
            inputs = [{
                "kind": "provider_metric",
                "role": f"ratio.{metric}",
                "source": "yahoo",
                "symbol": ratio_snapshot.symbol,
                "provider_field": ratio_snapshot.provider_fields[metric],
                "value": float(ratio_value),
                "retrieved_at": ratio_snapshot.fetched_at,
            }]
            available = True
            version = "yahoo_quote_summary_v1"
        elif source == "sec":
            canonical = sec_metrics.get(metric)
            if canonical is not None:
                value = canonical.value
                formula = canonical.formula
                inputs = sec_inputs(canonical)
                reason = canonical.availability_reason or "required_provenance_unavailable"
                as_of = (
                    max(item.provenance.filed for item in canonical.inputs).isoformat()
                    if canonical.inputs
                    else None
                )
                available = canonical.available
            else:
                as_of = None
                available = False
            version = canonical.formula_version if canonical is not None else "sec_companyfacts_v1"
        elif source == "yahoo":
            as_of = market_as_of if metric == "current_price" else market_window_end
            if metric == "current_price":
                observations = 1 if metric_is_available(value) else None
                inputs = market_observation(value)
            elif metric in {"beta", "correlation_sp500"}:
                observations = getattr(snapshot, "aligned_return_observations", None)
                aligned_start = getattr(
                    snapshot, "aligned_metric_session_start", market_window_start
                )
                aligned_end = getattr(
                    snapshot, "aligned_metric_session_end", market_window_end
                )
                as_of = aligned_end
                inputs = market_series(
                    "asset.adjusted_close_return_intervals",
                    normalized.get("ticker"),
                    observations,
                    aligned_start,
                    aligned_end,
                )
                inputs += market_series(
                    "benchmark.sp500.adjusted_close_return_intervals",
                    "^GSPC",
                    observations,
                    aligned_start,
                    aligned_end,
                )
            else:
                observations = getattr(snapshot, "asset_return_observations", None)
                inputs = market_series(
                    "asset.adjusted_close_return_intervals",
                    normalized.get("ticker"),
                    observations,
                )
            available = metric_is_available(value) and bool(inputs)
            version = "yahoo_chart_v1"
        else:
            as_of = market_as_of
            observations = 1 if metric_is_available(value) else None
            canonical = sec_metrics.get(combined_sec_metric[metric])
            if canonical is not None and canonical.available:
                inputs = sec_inputs(canonical)
            inputs += market_observation(normalized.get("current_price"))
            available = metric_is_available(value) and bool(inputs) and canonical is not None and canonical.available
            version = "sec_yahoo_formula_v1"

        evidence[metric] = {
            "value": float(value) if available else None,
            "available": available,
            "source": source,
            "as_of": as_of,
            "formula": formula,
            "formula_version": version,
            "inputs": inputs,
            "observation_count": observations,
            "availability_reason": None if available else reason,
        }
    return evidence


async def fetch_data_node(state: GOATState) -> GOATState:
    """Fetch SEC fundamentals first, then add best-effort Yahoo market data."""
    ticker = state["ticker"]
    market_data_only = state.get("market_data_only", False)

    if market_data_only:
        try:
            async with YahooChartClient() as client:
                snapshot = await client.get_snapshot(ticker)
            state["raw_data"] = {
                "current_price": snapshot.current_price,
                "as_of": snapshot.current_price_as_of,
                "fetched_at": snapshot.fetched_at,
            }
            state["data_source"] = "yahoo"
            return state
        except YahooChartError as yahoo_error:
            error_code = (
                "ticker_not_found"
                if isinstance(yahoo_error, YahooChartNotFound)
                else "market_data_temporarily_unavailable"
                if yahoo_error.retryable
                else "market_data_failure"
            )
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
        value.provenance.filed.isoformat()
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
        "normalized_ratios": {
            "available": False,
            "source": "yahoo",
            "as_of": None,
            "reason": "provider_temporarily_unavailable",
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

    snapshot = None
    ratio_snapshot = None
    try:
        async with YahooChartClient() as client:
            snapshot = await client.get_snapshot(ticker)

        current_price = snapshot.current_price
        normalized.update(
            {
                "current_price": current_price,
                "beta": snapshot.beta_sp500,
                "volatility_252d": snapshot.volatility_252d,
                "correlation_sp500": snapshot.correlation_sp500,
                "realized_total_return_252d": (
                    snapshot.realized_total_return_252d
                ),
            }
        )
        if current_price is not None:
            eps = normalized.get("eps")
            book_value = normalized.get("book_value_per_share")
            shares = normalized.get("shares_outstanding")
            revenue = normalized.get("revenue")
            normalized["pe_ratio"] = current_price / eps if eps is not None and eps > 0 else None
            normalized["pb_ratio"] = (
                current_price / book_value
                if book_value is not None and book_value > 0
                else None
            )
            normalized["market_cap"] = normalized["market_cap"] or (
                current_price * shares if shares not in (None, 0) else None
            )
            normalized["ps_ratio"] = (
                normalized["market_cap"] / revenue
                if normalized["market_cap"] is not None and revenue not in (None, 0)
                else None
            )
            dividend_per_share = normalized.get("dividend_per_share")
            normalized["dividend_yield"] = (
                dividend_per_share / current_price
                if dividend_per_share is not None and dividend_per_share >= 0
                else None
            )
        state["raw_data"] = {
            "ticker": snapshot.symbol,
            "currency": snapshot.currency,
            "fetched_at": snapshot.fetched_at,
            "current_price_as_of": snapshot.current_price_as_of,
            "history": [
                {
                    "date": point.session_date,
                    "timestamp": point.timestamp,
                    "close": point.close,
                    "adjusted_close": point.adjusted_close,
                }
                for point in snapshot.history
            ],
        }
        state["data_source"] = "sec+yahoo"
        state["data_status"]["market_data"] = {
            "available": current_price is not None,
            "source": "yahoo",
            "as_of": snapshot.current_price_as_of,
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
    except YahooChartError as yahoo_error:
        error_code = (
            "ticker_not_found"
            if isinstance(yahoo_error, YahooChartNotFound)
            else "market_data_temporarily_unavailable"
            if yahoo_error.retryable
            else "market_data_failure"
        )
        state["data_status"]["market_data"]["reason"] = error_code
    except Exception:
        state["data_status"]["market_data"]["reason"] = "market_data_failure"

    try:
        async with YahooRatiosClient() as client:
            ratio_snapshot = await client.get_snapshot(ticker)
        normalized.update(ratio_snapshot.values)
        state["data_source"] = "sec+yahoo"
        state["data_status"]["normalized_ratios"] = {
            "available": True,
            "source": "yahoo",
            "as_of": ratio_snapshot.fetched_at,
            "reason": None,
        }
    except YahooRatioError as ratio_error:
        ratio_snapshot = None
        state["data_status"]["normalized_ratios"]["reason"] = ratio_error.code
    except Exception:
        ratio_snapshot = None
        state["data_status"]["normalized_ratios"]["reason"] = (
            "ratio_data_failure"
        )

    state["normalized_data"] = normalized
    state["metric_evidence"] = _build_metric_evidence(
        normalized,
        snapshot,
        ratio_snapshot,
    )

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
    "buffett": (
        "roe",
        "profit_margin",
        "debt_to_equity",
        "free_cash_flow",
    ),
    "lynch": (
        "pe_ratio",
        "earnings_growth",
        "debt_to_equity",
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
    ),
    "munger": (
        "gross_margin",
        "operating_margin",
        "revenue_growth_5y",
        "capex_to_revenue",
        "debt_to_equity",
        "margin_trend",
    ),
    "dalio": (
        "beta",
        "volatility_252d",
        "debt_to_equity",
        "correlation_sp500",
        "realized_total_return_252d",
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
        required_metrics = _AGENT_REQUIRED_METRICS.get(name, ())
        if name == "lynch" and financials.get("earnings_turnaround") == 1.0:
            required_metrics = tuple(
                metric for metric in required_metrics if metric != "earnings_growth"
            ) + ("earnings_turnaround",)
        unavailable = [
            metric
            for metric in required_metrics
            if not metric_is_available(financials.get(metric))
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
                if not isinstance(result, dict):
                    raise TypeError("Agent output must be an object")
                public_result = {
                    field: result.get(field)
                    for field in AgentResult.model_fields
                    if field in result
                }
                validated_result = AgentResult.model_validate(public_result)
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
            "status": "incomplete",
            "verdict": "incomplete",
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
        "metric_evidence": state.get("metric_evidence") or {},
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
    "script-src 'self' 'sha256-NVbdyzSzSfcMqQh2NLZTqiqAV/36ktGEm11PDyYJfpg=' "
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
    provider_key = (YahooChartClient,)
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
    boundary_error = None
    fundamentals = None
    try:
        fundamentals = await _fundamentals_health_snapshot()
    except SECConfigurationError:
        boundary_error = HTTPException(
            status_code=503,
            detail={
                "code": "fundamentals_configuration_error",
                "message": "SEC fundamental data is not configured.",
                "retryable": False,
            },
        )
    except SECInvalidResponseError:
        boundary_error = HTTPException(
            status_code=502,
            detail={
                "code": "fundamentals_invalid_response",
                "message": "SEC returned invalid fundamental data.",
                "retryable": False,
            },
        )
    except SECUnavailableError:
        boundary_error = HTTPException(
            status_code=503,
            detail={
                "code": "fundamentals_temporarily_unavailable",
                "message": "SEC fundamental data is temporarily unavailable. Please retry.",
                "retryable": True,
            },
            headers={"Retry-After": "60"},
        )
    except (SECError, _FundamentalsHealthUnexpectedError):
        boundary_error = HTTPException(
            status_code=502,
            detail={
                "code": "fundamentals_failure",
                "message": "SEC fundamental data request failed.",
                "retryable": False,
            },
        )
    if boundary_error is not None:
        raise boundary_error
    assert fundamentals is not None

    filing_dates = [
        value.provenance.filed.isoformat()
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
    boundary_error = None
    result = None
    try:
        result = await _market_data_health_snapshot()
    except Exception:
        print("[Canary] Market-data health check failed unexpectedly.")
        boundary_error = HTTPException(
            status_code=502,
            detail={
                "code": "market_data_failure",
                "message": "Market data request failed.",
                "retryable": False,
            },
        )
    if boundary_error is not None:
        raise boundary_error
    assert result is not None
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
    boundary_error = None
    final_state = None
    try:
        final_state = await goat_workflow.ainvoke(initial_state)
    except Exception:
        boundary_error = HTTPException(status_code=500, detail="Analysis failed")
    if boundary_error is not None:
        raise boundary_error
    assert final_state is not None
    
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
        metric_evidence=report["metric_evidence"],
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
                "key_metrics": ["ROE", "Profit Margins", "Debt Levels", "Free Cash Flow"],
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
    """Report that optional earnings enrichment is currently unavailable."""
    ticker = _normalize_path_ticker(ticker)
    raise HTTPException(
        status_code=503,
        detail={
            "code": "earnings_enrichment_unavailable",
            "message": "Earnings enrichment is currently unavailable.",
            "retryable": False,
        },
    )


@app.get("/api/price-history/{ticker}", response_model=PriceHistoryResponse)
async def get_price_history(
    ticker: str,
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "ytd"] = "2y",
):
    """
    Get historical price data with earnings markers for charting.
    
    Returns daily OHLCV prices from the narrow chart endpoint.
    
    Query params:
        period: 1mo, 3mo, 6mo, 1y, 2y, ytd (default: 2y)
    """
    ticker = _normalize_path_ticker(ticker)
    boundary_error = None
    snapshot = None
    try:
        async with YahooChartClient() as client:
            snapshot = await client.get_snapshot(ticker)
    except YahooChartNotFound:
        boundary_error = HTTPException(
            status_code=404, detail="Price data was not found"
        )
    except YahooChartError as error:
        status_code = 503 if error.retryable else 502
        boundary_error = HTTPException(
            status_code=status_code, detail="Price data request failed"
        )
    except Exception:
        boundary_error = HTTPException(
            status_code=500, detail="Failed to fetch price data"
        )
    if boundary_error is not None:
        raise boundary_error
    assert snapshot is not None

    latest_session = date.fromisoformat(snapshot.history[-1].session_date)
    if period == "ytd":
        window_start = date(latest_session.year, 1, 1)
    else:
        window_days = {"1mo": 31, "3mo": 93, "6mo": 186, "1y": 366, "2y": 732}
        window_start = latest_session - timedelta(days=window_days[period])

    prices = [
        {
            "date": point.session_date,
            "open": point.open,
            "high": point.high,
            "low": point.low,
            "close": point.close,
            "volume": point.volume,
        }
        for point in snapshot.history
        if date.fromisoformat(point.session_date) >= window_start
    ]

    return PriceHistoryResponse(
        ticker=ticker,
        period=period,
        prices=[PricePoint(**p) for p in prices],
        earnings_markers=[],
        next_earnings_date=None,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
