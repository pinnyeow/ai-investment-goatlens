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
import hashlib
import json
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
from strategies import (
    StrategyResult,
    Verdict,
    calculate_consensus,
    calculate_consensus_with_llm,
    calculate_ordinal_consensus,
)
from llm import get_llm_client
from rag import RAGRetriever


# ============================================================================
# Pydantic Models
# ============================================================================

_TICKER_PATTERN = re.compile(r"(?=.{1,10}\Z)[A-Z0-9]+(?:[.-][A-Z0-9]+)?\Z")


# One inventory owns both the completion gate and every optional numeric metric
# a completed strategy may use for scoring, branching, insights, or concerns.
_AGENT_CONSUMPTION_CONTRACTS = {
    "buffett": {
        "agent": "Warren Buffett",
        "mandatory": ("roe", "profit_margin", "debt_to_equity", "free_cash_flow"),
        "optional": (),
    },
    "lynch": {
        "agent": "Peter Lynch",
        "mandatory": ("pe_ratio", "earnings_growth", "debt_to_equity"),
        "optional": ("institutional_ownership", "cash_and_equivalents"),
    },
    "graham": {
        "agent": "Benjamin Graham",
        "mandatory": (
            "pe_ratio", "pb_ratio", "current_ratio", "current_price",
            "current_assets", "total_liabilities", "shares_outstanding", "eps",
            "book_value_per_share",
        ),
        "optional": ("dividend_yield", "consecutive_dividend_years"),
    },
    "munger": {
        "agent": "Charlie Munger",
        "mandatory": (
            "gross_margin", "operating_margin", "revenue_growth_5y",
            "capex_to_revenue", "debt_to_equity", "margin_trend",
        ),
        "optional": ("roic", "insider_ownership"),
    },
    "dalio": {
        "agent": "Ray Dalio",
        "mandatory": (
            "beta", "volatility_252d", "debt_to_equity",
            "correlation_sp500", "realized_total_return_252d",
        ),
        "optional": ("interest_coverage", "debt_growth_3y"),
    },
}
_AGENT_CONSUMPTION_BY_NAME = {
    contract["agent"]: contract
    for contract in _AGENT_CONSUMPTION_CONTRACTS.values()
}


def _mandatory_consumed_metrics(contract, values):
    mandatory = contract["mandatory"]
    if (
        contract["agent"] == "Peter Lynch"
        and values.get("earnings_turnaround") == 1.0
    ):
        return tuple(metric for metric in mandatory if metric != "earnings_growth") + (
            "earnings_turnaround",
        )
    return mandatory


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
    required_metric_values: Dict[str, float] = Field(default_factory=dict)

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
        if any(not math.isfinite(value) for value in self.required_metric_values.values()):
            raise ValueError("Required metric values must be finite")
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
    adjusted_close: float
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
    series_id: str = Field(min_length=1)


class ReturnIntervalResponse(StrictAPIModel):
    start: str = Field(min_length=1)
    end: str = Field(min_length=1)
    value: float


class SnapshotSeriesResponse(StrictAPIModel):
    series_id: str = Field(min_length=1)
    source: Literal["yahoo"]
    symbol: str = Field(min_length=1)
    return_kind: Literal["log", "simple"]
    intervals: List[ReturnIntervalResponse] = Field(min_length=1, max_length=252)

    @model_validator(mode="after")
    def validate_interval_chronology(self):
        parsed = []
        for interval in self.intervals:
            try:
                start = date.fromisoformat(interval.start)
                end = date.fromisoformat(interval.end)
            except ValueError:
                raise ValueError("Return intervals require parseable ISO dates") from None
            if start >= end:
                raise ValueError("Return intervals require start before end")
            parsed.append((start, end))

        for previous, current in zip(parsed, parsed[1:]):
            if current <= previous:
                raise ValueError(
                    "Return intervals must be strictly ascending and distinct"
                )
            if current[0] < previous[1]:
                raise ValueError("Return intervals must be non-overlapping")
        return self


MetricInputResponse = Annotated[
    Union[
        SECMetricInputResponse,
        MarketObservationInputResponse,
        MarketSeriesInputResponse,
    ],
    Field(discriminator="kind"),
]


class MetricEvidenceResponse(StrictAPIModel):
    value: Optional[float]
    available: bool
    source: Literal["sec", "yahoo", "sec+yahoo"]
    as_of: Optional[str]
    filed_at: Optional[str] = None
    formula: str
    formula_version: str
    formula_owner: Literal["goatlens", "yahoo"] = "goatlens"
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
    retrieved_at: Optional[str] = Field(
        default=None,
        description="UTC timestamp when GOATlens retrieved the Yahoo snapshot."
    )
    currency: Optional[str] = Field(
        default=None,
        description="Yahoo quote currency, or null when market data is unavailable."
    )
    cache_age_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Age of the process-local Yahoo snapshot cache entry in seconds."
    )
    freshness: Literal["fresh", "stale", "unknown"] = Field(
        default="unknown",
        description="Fresh when the returned snapshot is inside the local cache TTL."
    )
    reason: Optional[str] = None


class AnalysisCompletenessStatus(StrictAPIModel):
    status: Literal["complete", "incomplete"]
    reason: Optional[str]


class AnalysisDataStatus(StrictAPIModel):
    fundamentals: FundamentalsAvailabilityStatus
    market_data: MarketAvailabilityStatus
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


class ExcludedAgentResponse(StrictAPIModel):
    agent: str = Field(min_length=1)
    unavailable_metrics: List[str]


class PartialIndicationResponse(StrictAPIModel):
    verdict: Literal[
        "strong_buy", "buy", "hold", "sell", "strong_sell", "mixed"
    ]
    directional_support: Optional[float] = Field(default=None, ge=0, le=1)
    supporting_analyses: int = Field(ge=0)
    completed_analyses: int = Field(ge=2)
    requested_analyses: int = Field(ge=2)
    excluded_agents: List[ExcludedAgentResponse]
    consensus_points: List[str] = Field(default_factory=list)
    divergence_points: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_directional_support(self):
        if self.verdict == "mixed":
            if self.directional_support is not None or self.supporting_analyses != 0:
                raise ValueError(
                    "Mixed partial indication cannot claim directional support"
                )
        elif (
            self.directional_support is None
            or self.supporting_analyses == 0
            or not math.isclose(
                self.directional_support,
                self.supporting_analyses / self.completed_analyses,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("Partial indication support fields are inconsistent")
        return self


class AnalysisResponse(StrictAPIModel):
    """Complete analysis response."""
    ticker: str
    company_name: str
    sector: str
    data_status: AnalysisDataStatus
    fundamentals: Dict[str, FundamentalFactResponse]
    fundamentals_history: Dict[str, List[FundamentalFactResponse]]
    metric_evidence: Dict[str, MetricEvidenceResponse]
    snapshot_series: List[SnapshotSeriesResponse] = Field(default_factory=list)
    
    # Consensus
    consensus_status: Literal["complete", "partial", "incomplete"]
    consensus_verdict: Literal[
        "strong_buy", "buy", "hold", "sell", "strong_sell", "mixed", "incomplete"
    ]
    agreement_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Share of completed agents supporting the consensus direction; "
            "buy and strong_buy share one direction, as do sell and strong_sell."
        ),
    )
    completed_analyses: int = Field(ge=0)
    requested_analyses: int = Field(ge=1)
    partial_indication: Optional[PartialIndicationResponse] = None
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
        if set(self.metric_evidence) != set(_METRIC_EVIDENCE_CONTRACTS):
            raise ValueError("Metric evidence keys must match the exact shipped registry")
        _validate_formula_evidence(self)
        series_by_id = {item.series_id: item for item in self.snapshot_series}
        if len(series_by_id) != len(self.snapshot_series):
            raise ValueError("Snapshot series IDs must be unique")
        for series in self.snapshot_series:
            canonical = json.dumps(
                {
                    "source": series.source,
                    "symbol": series.symbol,
                    "return_kind": series.return_kind,
                    "intervals": [
                        interval.model_dump() for interval in series.intervals
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            expected_id = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
            if series.series_id != expected_id:
                raise ValueError("Snapshot series ID must match its content digest")
        referenced_series_ids = {
            typed_input.series_id
            for evidence in self.metric_evidence.values()
            if evidence.available
            for typed_input in evidence.inputs
            if isinstance(typed_input, MarketSeriesInputResponse)
        }
        if set(series_by_id) != referenced_series_ids:
            raise ValueError(
                "Published snapshot series IDs must exactly match available evidence "
                "referenced series IDs"
            )
        expected_return_kinds = {
            "beta": "simple",
            "correlation_sp500": "simple",
            "volatility_252d": "log",
            "realized_total_return_252d": "log",
        }
        asset_role = "asset.adjusted_close_return_intervals"
        benchmark_role = "benchmark.sp500.adjusted_close_return_intervals"
        expected_dalio_inputs = {
            "beta": ((asset_role, self.ticker), (benchmark_role, "^GSPC")),
            "correlation_sp500": (
                (asset_role, self.ticker),
                (benchmark_role, "^GSPC"),
            ),
            "volatility_252d": ((asset_role, self.ticker),),
            "realized_total_return_252d": ((asset_role, self.ticker),),
        }
        for metric, expected_inputs in expected_dalio_inputs.items():
            evidence = self.metric_evidence.get(metric)
            if evidence is None or not evidence.available:
                continue
            series_inputs = [
                item
                for item in evidence.inputs
                if isinstance(item, MarketSeriesInputResponse)
            ]
            actual_inputs = tuple(
                (item.role, item.symbol) for item in series_inputs
            )
            if actual_inputs != expected_inputs or len(evidence.inputs) != len(
                expected_inputs
            ):
                raise ValueError("Dalio metric evidence requires exact series inputs")
            if (
                metric in {"beta", "correlation_sp500"}
                and series_inputs[0].series_id == series_inputs[1].series_id
            ):
                raise ValueError(
                    "Dalio market metrics require distinct series identities"
                )

        for metric, evidence in self.metric_evidence.items():
            for typed_input in evidence.inputs:
                if not isinstance(typed_input, MarketSeriesInputResponse):
                    continue
                series = series_by_id.get(typed_input.series_id)
                if series is None:
                    raise ValueError("Market-series evidence must reference snapshot_series")
                if (
                    typed_input.symbol != series.symbol
                    or typed_input.observation_count != len(series.intervals)
                    or typed_input.start != series.intervals[0].start
                    or typed_input.end != series.intervals[-1].end
                    or expected_return_kinds.get(metric) != series.return_kind
                ):
                    raise ValueError(
                        "Market-series evidence must match referenced series metadata"
                    )

        recomputed_values = {}
        for metric in expected_dalio_inputs:
            evidence = self.metric_evidence.get(metric)
            if evidence is None or not evidence.available:
                continue
            series_inputs = [
                item
                for item in evidence.inputs
                if isinstance(item, MarketSeriesInputResponse)
            ]
            referenced_series = [
                series_by_id[item.series_id] for item in series_inputs
            ]
            values_by_series = [
                [interval.value for interval in series.intervals]
                for series in referenced_series
            ]
            if any(
                not math.isfinite(value)
                for values in values_by_series
                for value in values
            ):
                raise ValueError(
                    "Dalio metric series must be finite and non-degenerate"
                )

            if metric in {"beta", "correlation_sp500"}:
                asset_series, benchmark_series = referenced_series
                asset_intervals = [
                    (interval.start, interval.end)
                    for interval in asset_series.intervals
                ]
                benchmark_intervals = [
                    (interval.start, interval.end)
                    for interval in benchmark_series.intervals
                ]
                asset_values, benchmark_values = values_by_series
                if len(asset_values) < 2:
                    raise ValueError(
                        "Dalio metric series must be finite and non-degenerate"
                    )
                if asset_intervals != benchmark_intervals:
                    raise ValueError(
                        "Dalio asset and benchmark series require aligned interval windows"
                    )
                asset_mean = sum(asset_values) / len(asset_values)
                benchmark_mean = sum(benchmark_values) / len(benchmark_values)
                asset_variance = sum(
                    (value - asset_mean) ** 2 for value in asset_values
                ) / (len(asset_values) - 1)
                benchmark_variance = sum(
                    (value - benchmark_mean) ** 2 for value in benchmark_values
                ) / (len(benchmark_values) - 1)
                if asset_variance <= 0 or benchmark_variance <= 0:
                    raise ValueError(
                        "Dalio metric series must be finite and non-degenerate"
                    )
                covariance = sum(
                    (asset - asset_mean) * (benchmark - benchmark_mean)
                    for asset, benchmark in zip(asset_values, benchmark_values)
                ) / (len(asset_values) - 1)
                if metric == "beta":
                    recomputed_values[metric] = covariance / benchmark_variance
                else:
                    recomputed_values[metric] = covariance / math.sqrt(
                        asset_variance * benchmark_variance
                    )
            elif metric == "volatility_252d":
                asset_values = values_by_series[0]
                if len(asset_values) < 2:
                    raise ValueError(
                        "Dalio metric series must be finite and non-degenerate"
                    )
                asset_mean = sum(asset_values) / len(asset_values)
                sample_variance = sum(
                    (value - asset_mean) ** 2 for value in asset_values
                ) / (len(asset_values) - 1)
                recomputed_values[metric] = math.sqrt(sample_variance) * math.sqrt(
                    252
                )
            else:
                try:
                    recomputed_values[metric] = math.exp(
                        sum(values_by_series[0])
                    ) - 1
                except OverflowError:
                    raise ValueError(
                        "Dalio metric series must be finite and non-degenerate"
                    ) from None

        for metric, recomputed_value in recomputed_values.items():
            evidence = self.metric_evidence[metric]
            if (
                not math.isfinite(recomputed_value)
                or evidence.value is None
                or not math.isclose(
                    evidence.value,
                    recomputed_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("Dalio metric evidence must match recomputed value")

        agent_names = [result.agent for result in self.agent_results]
        if len(agent_names) != len(set(agent_names)):
            raise ValueError("Agent results must have unique agent identities")
        if any(
            result.agent not in _AGENT_CONSUMPTION_BY_NAME
            for result in self.agent_results
        ):
            raise ValueError("Agent results require a known identity")
        expected_comparison = [
            ComparisonAgentResponse(
                name=result.agent,
                style=result.style,
                verdict=result.verdict,
                score=result.score,
                key_metric=result.insights[0] if result.insights else "",
            )
            for result in self.agent_results
        ]
        if self.comparison_table.agents != expected_comparison:
            raise ValueError("Comparison table must exactly project agent results")
        actual_requested = len(self.agent_results)
        completed_results = [
            result for result in self.agent_results if result.status == "complete"
        ]
        actual_completed = len(completed_results)
        for result in completed_results:
            contract = _AGENT_CONSUMPTION_BY_NAME.get(result.agent)
            if contract is None:
                continue
            evidence_values = {
                metric: evidence.value
                for metric, evidence in self.metric_evidence.items()
                if evidence.available
            }
            mandatory = _mandatory_consumed_metrics(contract, evidence_values)
            expected = mandatory + tuple(
                metric
                for metric in contract["optional"]
                if (
                    (evidence := self.metric_evidence.get(metric)) is not None
                    and evidence.available
                    and evidence.value is not None
                )
            )
            if set(result.required_metric_values) != set(expected):
                raise ValueError(
                    "Complete agents require exact consumed metric evidence keys"
                )
            for metric in expected:
                evidence = self.metric_evidence.get(metric)
                consumed = result.required_metric_values.get(metric)
                if (
                    evidence is None
                    or not evidence.available
                    or evidence.value is None
                    or consumed is None
                    or not math.isclose(
                        evidence.value, consumed, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    raise ValueError(
                        "Complete agents require matching required metric evidence"
                    )
        if (
            self.requested_analyses != actual_requested
            or self.completed_analyses != actual_completed
        ):
            raise ValueError("Analysis counts must match agent results")
        if self.completed_analyses > self.requested_analyses:
            raise ValueError("Completed analyses cannot exceed requested analyses")
        if (
            self.consensus_status == "partial"
            and self.consensus_verdict != "incomplete"
        ):
            raise ValueError(
                "Partial analysis cannot publish a final consensus verdict"
            )

        expected_verdict = None
        if actual_completed >= 2:
            _, expected_consensus = calculate_ordinal_consensus(
                [Verdict(result.verdict) for result in completed_results]
            )
            expected_verdict = expected_consensus.value
            published_verdict = (
                self.partial_indication.verdict
                if self.consensus_status == "partial" and self.partial_indication
                else self.consensus_verdict
            )
            if published_verdict != expected_verdict:
                raise ValueError(
                    "Published verdict does not match the expected ordinal "
                    f"consensus verdict {expected_verdict!r}"
                )

        indication = self.partial_indication
        direction_by_verdict = {
            "strong_sell": -1,
            "sell": -1,
            "hold": 0,
            "buy": 1,
            "strong_buy": 1,
        }
        if self.consensus_status == "complete":
            if self.consensus_verdict == "mixed":
                support_valid = self.agreement_score is None
            else:
                consensus_direction = direction_by_verdict.get(self.consensus_verdict)
                supporting = sum(
                    direction_by_verdict.get(result.verdict or "") == consensus_direction
                    for result in completed_results
                )
                support_valid = (
                    self.agreement_score is not None
                    and math.isclose(
                        self.agreement_score,
                        supporting / self.completed_analyses,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
            valid = (
                self.completed_analyses >= 2
                and self.completed_analyses == self.requested_analyses
                and self.consensus_verdict != "incomplete"
                and indication is None
                and support_valid
            )
        elif self.consensus_status == "partial":
            incomplete_projection = [
                ExcludedAgentResponse(
                    agent=result.agent,
                    unavailable_metrics=result.unavailable_metrics,
                )
                for result in self.agent_results
                if result.status == "incomplete"
            ]
            if indication is None or indication.verdict == "mixed":
                indication_support_valid = indication is not None
            else:
                indication_direction = direction_by_verdict[indication.verdict]
                supporting = sum(
                    direction_by_verdict.get(result.verdict or "") == indication_direction
                    for result in completed_results
                )
                indication_support_valid = (
                    indication.supporting_analyses == supporting
                    and indication.directional_support is not None
                    and math.isclose(
                        indication.directional_support,
                        supporting / self.completed_analyses,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
            valid = (
                2 <= self.completed_analyses < self.requested_analyses
                and self.consensus_verdict == "incomplete"
                and self.agreement_score is None
                and not self.consensus_points
                and indication is not None
                and indication.completed_analyses == self.completed_analyses
                and indication.requested_analyses == self.requested_analyses
                and len(indication.excluded_agents)
                == self.requested_analyses - self.completed_analyses
                and indication.supporting_analyses <= self.completed_analyses
                and indication.excluded_agents == incomplete_projection
                and indication_support_valid
            )
        else:
            valid = (
                self.completed_analyses < 2
                and self.consensus_verdict == "incomplete"
                and self.agreement_score is None
                and indication is None
                and not self.consensus_points
            )
        if not valid:
            raise ValueError("Consensus fields are inconsistent")
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
    snapshot_series: NotRequired[List[Dict[str, Any]]]
    
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
    "debt_to_equity": (
        "sec", "reviewed_funded_debt_branch_divided_by_equity"
    ),
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
    "interest_coverage": (
        "sec",
        "operating_income_divided_by_interest_expense",
    ),
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
    "market_cap": ("sec+yahoo", "current_price_times_period_end_common_shares"),
    "ps_ratio": ("sec+yahoo", "market_cap_divided_by_annual_revenue"),
}


_FORMULA_EVIDENCE_CONTRACTS = {
    "roe": ("sec", "net_income_divided_by_equity", "sec_companyfacts_v1",
            ("income.current.net_income", "capital.end.equity")),
    "profit_margin": ("sec", "net_income_divided_by_revenue", "sec_companyfacts_v1",
                      ("income.current.net_income", "income.current.revenue")),
    "debt_to_equity": (
        "sec",
        None,
        "sec_companyfacts_debt_to_equity_v3",
        None,
    ),
    "free_cash_flow": ("sec", "operating_cash_flow_minus_capex", "sec_companyfacts_v1",
                       ("cashflow.current.operating", "cashflow.current.capex")),
    "earnings_growth": ("sec", "three_year_net_income_cagr", "sec_companyfacts_v1",
                        ("income.begin", "income.current")),
    "earnings_turnaround": (
        "sec", "negative_or_zero_to_positive_three_year_net_income_transition",
        "sec_companyfacts_v1", ("income.begin", "income.current")),
    "cash_and_equivalents": ("sec", "reported", "sec_companyfacts_v1", ("cash.current",)),
    "current_ratio": ("sec", "current_assets_divided_by_current_liabilities",
                      "sec_companyfacts_v1", (
                          "balance.current.current_assets",
                          "balance.current.current_liabilities")),
    "current_assets": ("sec", "reported", "sec_companyfacts_v1",
                       ("balance.current.current_assets",)),
    "total_liabilities": ("sec", "reported_or_compatibly_derived", "sec_companyfacts_v1",
                          ("balance.current.liabilities",)),
    "shares_outstanding": ("sec", "reported_period_end", "sec_companyfacts_v1",
                           ("shares.period_end",)),
    "eps": ("sec", "reported_diluted_eps", "sec_companyfacts_v1",
            ("income.current.diluted_eps",)),
    "book_value_per_share": ("sec", "equity_divided_by_period_end_shares",
                             "sec_companyfacts_v1",
                             ("capital.end.equity", "shares.period_end")),
    "consecutive_dividend_years": (
        "sec", "consecutive_positive_common_dividend_years",
        "sec_companyfacts_v1", None),
    "gross_margin": ("sec", None, "sec_companyfacts_v1", None),
    "operating_margin": ("sec", "operating_income_divided_by_revenue",
                         "sec_companyfacts_v1", (
                             "income.current.operating_income",
                             "income.current.revenue")),
    "roic": ("sec", "nopat_divided_by_average_invested_capital",
             "sec_companyfacts_v1", (
                 "income.current.operating_income", "income.current.tax",
                 "income.current.pretax_income", "capital.begin.equity",
                 "capital.end.equity", "capital.begin.cash", "capital.end.cash",
                 "capital.begin.debt.long_term_current",
                 "capital.begin.debt.long_term_noncurrent",
                 "capital.begin.debt.short_term_borrowings",
                 "capital.end.debt.long_term_current",
                 "capital.end.debt.long_term_noncurrent",
                 "capital.end.debt.short_term_borrowings")),
    "revenue_growth_5y": ("sec", "five_year_revenue_cagr", "sec_companyfacts_v1",
                          ("revenue.begin", "revenue.current")),
    "capex_to_revenue": ("sec", "capex_divided_by_revenue", "sec_companyfacts_v1",
                         ("cashflow.current.capex", "income.current.revenue")),
    "interest_coverage": ("sec", "operating_income_divided_by_interest_expense",
                          "sec_companyfacts_v1", (
                              "income.current.operating_income",
                              "income.current.interest_expense")),
    "margin_trend": ("sec", "five_year_operating_margin_ols_slope",
                     "sec_companyfacts_v1", None),
    "debt_growth_3y": ("sec", "three_year_funded_debt_cagr", "sec_companyfacts_v1", (
        "debt.current.long_term_current", "debt.current.long_term_noncurrent",
        "debt.current.short_term_borrowings", "debt.begin.long_term_current",
        "debt.begin.long_term_noncurrent", "debt.begin.short_term_borrowings")),
    "current_price": ("yahoo", "provider_regular_market_price", "yahoo_chart_v1",
                      ("price.current",)),
    "beta": (
        "yahoo",
        "aligned_daily_return_sample_covariance_divided_by_benchmark_variance",
        "goatlens_beta_v1",
        (
            "asset.adjusted_close_return_intervals",
            "benchmark.sp500.adjusted_close_return_intervals",
        ),
    ),
    "volatility_252d": (
        "yahoo", "annualized_log_return_volatility_252_sessions",
        "goatlens_volatility_252d_v1",
        ("asset.adjusted_close_return_intervals",),
    ),
    "correlation_sp500": (
        "yahoo", "aligned_daily_return_pearson_correlation",
        "goatlens_correlation_sp500_v1",
        (
            "asset.adjusted_close_return_intervals",
            "benchmark.sp500.adjusted_close_return_intervals",
        ),
    ),
    "realized_total_return_252d": (
        "yahoo", "adjusted_close_total_return_252_sessions",
        "goatlens_realized_total_return_252d_v1",
        ("asset.adjusted_close_return_intervals",),
    ),
    "pe_ratio": ("sec+yahoo", "current_price_divided_by_diluted_eps",
                 "sec_yahoo_formula_v1", ("income.current.diluted_eps", "price.current")),
    "pb_ratio": ("sec+yahoo", "current_price_divided_by_book_value_per_share",
                 "sec_yahoo_formula_v1", (
                     "capital.end.equity", "shares.period_end", "price.current")),
    "dividend_yield": ("sec+yahoo",
                       "annual_common_dividend_per_share_divided_by_current_price",
                       "sec_yahoo_formula_v1", ("dividend.current", "price.current")),
    "market_cap": ("sec+yahoo", "current_price_times_period_end_common_shares",
                   "sec_yahoo_formula_v1", ("shares.period_end", "price.current")),
    "ps_ratio": ("sec+yahoo", "market_cap_divided_by_annual_revenue",
                 "sec_yahoo_formula_v1", (
                     "shares.period_end", "income.current.revenue", "price.current")),
}


def _recompute_formula_value(metric, values, formula):
    if metric in {
        "cash_and_equivalents", "current_assets", "total_liabilities",
        "shares_outstanding", "eps", "current_price",
    }:
        return values[0]
    if metric in {"roe", "profit_margin", "current_ratio", "operating_margin",
                  "capex_to_revenue", "interest_coverage"}:
        return values[0] / values[1]
    if metric == "debt_to_equity":
        return sum(values[:-1]) / values[-1]
    if metric == "free_cash_flow":
        return values[0] - values[1]
    if metric == "earnings_growth":
        return (values[1] / values[0]) ** (1 / 3) - 1
    if metric == "earnings_turnaround":
        return 1.0 if values[0] <= 0 < values[1] else 0.0
    if metric == "book_value_per_share":
        return values[0] / values[1]
    if metric == "gross_margin":
        return (
            values[0] / values[1]
            if formula == "gross_profit_divided_by_revenue"
            else (values[0] - values[1]) / values[0]
        )
    if metric == "roic":
        operating_income, tax, pretax = values[:3]
        begin_capital = values[3] + sum(values[7:10]) - values[5]
        end_capital = values[4] + sum(values[10:13]) - values[6]
        return operating_income * (1 - tax / pretax) / (
            (begin_capital + end_capital) / 2
        )
    if metric == "revenue_growth_5y":
        return (values[1] / values[0]) ** (1 / 5) - 1
    if metric == "margin_trend":
        margins = [values[index + 1] / values[index] for index in range(0, 10, 2)]
        mean = sum(margins) / 5
        return sum((index - 2) * (value - mean) for index, value in enumerate(margins)) / 10
    if metric == "debt_growth_3y":
        return (sum(values[:3]) / sum(values[3:])) ** (1 / 3) - 1
    if metric == "consecutive_dividend_years":
        return float(len(values) - 1)
    if metric == "pe_ratio":
        return values[1] / values[0]
    if metric == "pb_ratio":
        return values[2] / (values[0] / values[1])
    if metric == "dividend_yield":
        return values[0] / values[1]
    if metric == "market_cap":
        return values[0] * values[1]
    if metric == "ps_ratio":
        return values[2] * values[0] / values[1]
    raise ValueError("Unsupported formula evidence contract")


_SEC_ROLE_FUNDAMENTAL_KEYS = {
    "income.current.net_income": "net_income",
    "capital.end.equity": "equity",
    "income.current.revenue": "revenue",
    "debt.current.long_term_current": "long_term_debt_current",
    "debt.current.long_term_noncurrent": "long_term_debt_noncurrent",
    "debt.current.short_term_borrowings": "short_term_borrowings",
    "debt.current.debt_current": "debt_current",
    "debt.current.long_term_debt": "long_term_debt",
    "cashflow.current.operating": "operating_cash_flow",
    "cashflow.current.capex": "capex_payments",
    "income.begin": "net_income",
    "income.current": "net_income",
    "cash.current": "cash_and_equivalents",
    "balance.current.current_assets": "current_assets",
    "balance.current.current_liabilities": "current_liabilities",
    "balance.current.liabilities": "liabilities",
    "shares.period_end": "shares_outstanding",
    "income.current.diluted_eps": "diluted_eps",
    "income.current.gross_profit": "gross_profit",
    "income.current.cost_of_revenue": "cost_of_revenue",
    "income.current.operating_income": "operating_income",
    "income.current.tax": "income_tax",
    "income.current.pretax_income": "pretax_income",
    "capital.begin.equity": "equity",
    "capital.begin.cash": "cash_and_equivalents",
    "capital.end.cash": "cash_and_equivalents",
    "capital.begin.debt.long_term_current": "long_term_debt_current",
    "capital.begin.debt.long_term_noncurrent": "long_term_debt_noncurrent",
    "capital.begin.debt.short_term_borrowings": "short_term_borrowings",
    "capital.end.debt.long_term_current": "long_term_debt_current",
    "capital.end.debt.long_term_noncurrent": "long_term_debt_noncurrent",
    "capital.end.debt.short_term_borrowings": "short_term_borrowings",
    "revenue.begin": "revenue",
    "revenue.current": "revenue",
    "income.current.interest_expense": "interest_expense",
    "debt.begin.long_term_current": "long_term_debt_current",
    "debt.begin.long_term_noncurrent": "long_term_debt_noncurrent",
    "debt.begin.short_term_borrowings": "short_term_borrowings",
    "dividend.current": "dividend_per_share",
}


def _fundamental_key_for_role(role):
    if role.startswith("margin."):
        return "revenue" if role.endswith(".revenue") else "operating_income"
    if role.startswith("streak."):
        return "dividend_per_share"
    return _SEC_ROLE_FUNDAMENTAL_KEYS.get(role)


def _validate_formula_evidence(response):
    public_facts = {
        key: [item] + response.fundamentals_history.get(key, [])
        for key, item in response.fundamentals.items()
    }
    current_facts = {
        key: [item] for key, item in response.fundamentals.items()
    }
    for evidence in response.metric_evidence.values():
        sec_inputs = [
            item for item in evidence.inputs
            if isinstance(item, SECMetricInputResponse)
        ]
        observation_inputs = [
            item for item in evidence.inputs
            if isinstance(item, MarketObservationInputResponse)
        ]
        series_inputs = [
            item for item in evidence.inputs
            if isinstance(item, MarketSeriesInputResponse)
        ]
        expected_filed_at = (
            max(item.provenance.filed for item in sec_inputs)
            if sec_inputs else None
        )
        if evidence.source == "yahoo":
            expected_as_of = (
                observation_inputs[0].observed_at
                if len(observation_inputs) == 1
                else max(item.end for item in series_inputs)
                if series_inputs else None
            )
        else:
            expected_as_of = (
                max(item.provenance.end for item in sec_inputs)
                if sec_inputs else None
            )
        if observation_inputs:
            expected_observation_count = len(observation_inputs)
        elif series_inputs:
            series_counts = {item.observation_count for item in series_inputs}
            expected_observation_count = (
                next(iter(series_counts)) if len(series_counts) == 1 else None
            )
        else:
            expected_observation_count = None
        if (
            evidence.as_of != expected_as_of
            or evidence.filed_at != expected_filed_at
            or evidence.observation_count != expected_observation_count
        ):
            raise ValueError("Metric evidence metadata must derive from typed inputs")

    current_price = response.metric_evidence.get("current_price")
    canonical_price_input = None
    if current_price is not None and current_price.available:
        price_inputs = [
            item for item in current_price.inputs
            if isinstance(item, MarketObservationInputResponse)
        ]
        if len(price_inputs) == 1:
            canonical_price_input = price_inputs[0]
        if (
            canonical_price_input is None
            or canonical_price_input.observed_at
            != response.data_status.market_data.as_of
        ):
            raise ValueError("Current price timestamp must match market status as_of")

    for metric, contract in _FORMULA_EVIDENCE_CONTRACTS.items():
        evidence = response.metric_evidence.get(metric)
        if evidence is None:
            continue
        source, formula, version, expected_roles = contract
        expected_owner = "yahoo" if metric == "current_price" else "goatlens"
        if metric == "debt_to_equity" and not evidence.available:
            equity = response.fundamentals.get("equity")
            expected_reason = (
                "nonpositive_equity_not_meaningful"
                if equity is not None
                and equity.value is not None
                and equity.value <= 0
                else "required_inputs_unavailable"
            )
            if (
                evidence.source != "sec"
                or evidence.formula
                != "reviewed_funded_debt_branch_divided_by_equity"
                or evidence.formula_version != version
                or evidence.formula_owner != "goatlens"
                or evidence.availability_reason != expected_reason
                or evidence.inputs
                or evidence.as_of is not None
                or evidence.filed_at is not None
                or evidence.observation_count is not None
            ):
                raise ValueError(
                    "Unavailable debt-to-equity evidence contract is inconsistent"
                )
            continue
        if (
            evidence.source != source
            or evidence.formula_version != version
            or evidence.formula_owner != expected_owner
            or (formula is not None and evidence.formula != formula)
        ):
            raise ValueError("Metric formula evidence contract is inconsistent")
        if not evidence.available:
            continue

        roles = tuple(item.role for item in evidence.inputs)
        if metric == "debt_to_equity":
            equity_input = next(
                (
                    item
                    for item in evidence.inputs
                    if isinstance(item, SECMetricInputResponse)
                    and item.role == "capital.end.equity"
                ),
                None,
            )

            if equity_input is None or equity_input.value <= 0:
                raise ValueError(
                    "Available debt-to-equity evidence requires positive equity"
                )

            def compatible_public_fact(key):
                if equity_input is None:
                    return None
                return next(
                    (
                        fact
                        for fact in public_facts.get(key, [])
                        if fact.value is not None
                        and fact.provenance is not None
                        and (
                            fact.provenance.end,
                            fact.provenance.accession,
                            fact.provenance.form,
                            fact.provenance.filed,
                            fact.provenance.fiscal_year,
                        )
                        == (
                            equity_input.provenance.end,
                            equity_input.provenance.accession,
                            equity_input.provenance.form,
                            equity_input.provenance.filed,
                            equity_input.provenance.fiscal_year,
                        )
                    ),
                    None,
                )

            compatible_debt = {
                key: compatible_public_fact(key)
                for key in (
                    "long_term_debt_current",
                    "long_term_debt_noncurrent",
                    "short_term_borrowings",
                    "debt_current",
                    "long_term_debt",
                )
            }

            def publicly_unreported(key):
                current = current_facts.get(key, [None])[0]
                return (
                    compatible_debt[key] is None
                    and current is not None
                    and current.availability_reason
                    != "conflicting_standard_concepts"
                    and equity_input is not None
                    and not any(
                        fact.value is not None
                        and fact.provenance is not None
                        and fact.provenance.end == equity_input.provenance.end
                        for fact in public_facts.get(key, [])
                    )
                )

            expected_formula = None
            if (
                compatible_debt["long_term_debt_current"] is not None
                and compatible_debt["long_term_debt_noncurrent"] is not None
                and (
                    compatible_debt["short_term_borrowings"] is not None
                    or publicly_unreported("short_term_borrowings")
                )
            ):
                expected_formula = (
                    "long_term_debt_components_plus_optional_short_term_"
                    "borrowings_divided_by_equity"
                )
                expected_roles = (
                    "debt.current.long_term_current",
                    "debt.current.long_term_noncurrent",
                    *(
                        ("debt.current.short_term_borrowings",)
                        if compatible_debt["short_term_borrowings"] is not None
                        else ()
                    ),
                    "capital.end.equity",
                )
            elif (
                compatible_debt["debt_current"] is not None
                and compatible_debt["long_term_debt"] is not None
                and all(
                    publicly_unreported(key)
                    for key in (
                        "long_term_debt_current",
                        "long_term_debt_noncurrent",
                        "short_term_borrowings",
                    )
                )
            ):
                expected_formula = (
                    "debt_current_plus_long_term_debt_divided_by_equity"
                )
                expected_roles = (
                    "debt.current.debt_current",
                    "debt.current.long_term_debt",
                    "capital.end.equity",
                )
            elif (
                compatible_debt["short_term_borrowings"] is not None
                and compatible_debt["long_term_debt_noncurrent"] is not None
                and all(
                    publicly_unreported(key)
                    for key in (
                        "long_term_debt_current",
                        "debt_current",
                        "long_term_debt",
                    )
                )
            ):
                expected_formula = (
                    "short_term_borrowings_plus_long_term_debt_noncurrent_"
                    "divided_by_equity"
                )
                expected_roles = (
                    "debt.current.short_term_borrowings",
                    "debt.current.long_term_noncurrent",
                    "capital.end.equity",
                )
            else:
                expected_roles = ()
            if evidence.formula != expected_formula:
                raise ValueError("Metric formula evidence contract is inconsistent")
        elif metric == "gross_margin":
            gross_contracts = {
                "gross_profit_divided_by_revenue": (
                    "income.current.gross_profit", "income.current.revenue"),
                "revenue_minus_cost_of_revenue_divided_by_revenue": (
                    "income.current.revenue", "income.current.cost_of_revenue"),
            }
            expected_roles = gross_contracts.get(evidence.formula)
        elif metric == "margin_trend":
            if len(roles) != 10 or any(
                re.fullmatch(
                    r"margin\.\d{4}\.(revenue|operating_income)", role
                ) is None
                for role in roles
            ):
                expected_roles = ()
            else:
                years = [role.split(".")[1] for role in roles[::2]]
                numeric_years = [int(year) for year in years]
                expected_roles = tuple(
                    role
                    for year in years
                    for role in (
                        f"margin.{year}.revenue",
                        f"margin.{year}.operating_income",
                    )
                )
                if (
                    len(set(numeric_years)) != 5
                    or numeric_years != sorted(numeric_years)
                    or any(
                        later - earlier != 1
                        for earlier, later in zip(numeric_years, numeric_years[1:])
                    )
                    or any(
                        int(item.provenance.end[:4]) != int(item.role.split(".")[1])
                        for item in evidence.inputs
                        if isinstance(item, SECMetricInputResponse)
                    )
                ):
                    expected_roles = ()
        elif metric == "consecutive_dividend_years":
            expected_roles = roles
            valid_roles = bool(roles) and all(
                re.fullmatch(r"streak\.\d{4}\.dividend", role)
                for role in roles[:-1]
            ) and re.fullmatch(r"streak\.\d{4}\.terminator", roles[-1])
            if valid_roles:
                years = [int(role.split(".")[1]) for role in roles]
                valid_roles = all(
                    earlier - later == 1
                    for earlier, later in zip(years, years[1:])
                ) and all(
                    int(item.provenance.end[:4]) == int(item.role.split(".")[1])
                    for item in evidence.inputs
                    if isinstance(item, SECMetricInputResponse)
                )
            if not valid_roles:
                expected_roles = ()
        if expected_roles is None or roles != expected_roles:
            if metric in {
                "beta", "volatility_252d", "correlation_sp500",
                "realized_total_return_252d",
            }:
                raise ValueError(
                    "Dalio metric evidence requires exact series inputs"
                )
            raise ValueError("Metric formula evidence requires exact input roles")

        sec_role_inputs = [
            item for item in evidence.inputs
            if isinstance(item, SECMetricInputResponse)
        ]
        context_groups = {}
        for item in sec_role_inputs:
            segments = item.role.split(".")
            if "begin" in segments:
                group = "begin"
            elif "current" in segments or "end" in segments or "period_end" in segments:
                group = "current"
            elif item.role.startswith("margin."):
                group = ".".join(segments[:2])
            else:
                group = None
            if group is not None:
                context_groups.setdefault(group, []).append(item)
        for grouped_inputs in context_groups.values():
            contexts = {
                (
                    item.provenance.end,
                    item.provenance.accession,
                    item.provenance.form,
                    item.provenance.filed,
                    item.provenance.fiscal_year,
                )
                for item in grouped_inputs
            }
            duration_starts = {
                item.provenance.start
                for item in grouped_inputs
                if item.provenance.start is not None
            }
            if len(contexts) != 1 or len(duration_starts) > 1:
                raise ValueError(
                    "Same-period SEC formula inputs require compatible filing context"
                )
        begin_inputs = context_groups.get("begin", [])
        current_inputs = context_groups.get("current", [])
        if begin_inputs and current_inputs and not (
            max(item.provenance.end for item in begin_inputs)
            < min(item.provenance.end for item in current_inputs)
        ):
            raise ValueError("Historical SEC formula roles require strict chronology")

        if metric in {
            "beta", "volatility_252d", "correlation_sp500",
            "realized_total_return_252d",
        }:
            if not all(
                isinstance(item, MarketSeriesInputResponse)
                for item in evidence.inputs
            ):
                raise ValueError("Metric formula evidence has an invalid input source")
            # Exact series identity, windows, content digests, and independent
            # recomputation are enforced by AnalysisResponse below.
            continue

        for item in evidence.inputs:
            if isinstance(item, SECMetricInputResponse):
                fundamental_key = _fundamental_key_for_role(item.role)
                role_segments = item.role.split(".")
                allowed_facts = (
                    current_facts
                    if "current" in role_segments
                    else public_facts
                )
                if not any(
                    math.isclose(item.value, fact.value, rel_tol=0.0, abs_tol=1e-12)
                    and item.provenance == fact.provenance
                    for fact in allowed_facts.get(fundamental_key, [])
                    if fact.value is not None and fact.provenance is not None
                ):
                    message = (
                        "SEC current roles must bind to current fundamentals"
                        if "current" in role_segments
                        else "SEC formula evidence must bind to public fundamentals"
                    )
                    raise ValueError(message)
            elif isinstance(item, MarketObservationInputResponse):
                if (
                    canonical_price_input is None
                    or item != canonical_price_input
                    or item.symbol != response.ticker
                ):
                    raise ValueError(
                        "Market formula evidence must bind to the public observation"
                    )
            else:
                raise ValueError("Metric formula evidence has an invalid input source")

        values = [item.value for item in evidence.inputs]
        try:
            recomputed = _recompute_formula_value(metric, values, evidence.formula)
        except (ArithmeticError, OverflowError, ValueError):
            raise ValueError("Metric formula evidence cannot be recomputed") from None
        if (
            evidence.value is None
            or not math.isfinite(recomputed)
            or not math.isclose(
                evidence.value, recomputed, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError("Metric formula evidence must match recomputed value")


def _build_snapshot_series(snapshot, ticker):
    if snapshot is None:
        return [], {}

    def returns(points):
        if len(points) < 253:
            return [], [], []
        window = points[-253:]
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

    asset_logs, asset_simple, asset_intervals = returns(snapshot.history)
    benchmark_history = getattr(snapshot, "benchmark_history", [])
    _, benchmark_simple, benchmark_intervals = returns(benchmark_history)
    asset_simple_by_interval = dict(zip(asset_intervals, asset_simple))
    benchmark_simple_by_interval = dict(zip(benchmark_intervals, benchmark_simple))
    shared = sorted(set(asset_simple_by_interval) & set(benchmark_simple_by_interval))[-252:]
    series_by_id = {}
    references = {}

    def add_series(role, symbol, return_kind, intervals, values):
        if not intervals or len(intervals) != len(values):
            return
        observations = [
            {"start": start, "end": end, "value": float(value)}
            for (start, end), value in zip(intervals, values)
        ]
        canonical = json.dumps(
            {
                "source": "yahoo",
                "symbol": symbol,
                "return_kind": return_kind,
                "intervals": observations,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        series_id = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        series_by_id[series_id] = {
            "series_id": series_id,
            "source": "yahoo",
            "symbol": symbol,
            "return_kind": return_kind,
            "intervals": observations,
        }
        references[role] = series_id

    add_series("asset_log", ticker, "log", asset_intervals, asset_logs)
    add_series(
        "asset_aligned", ticker, "simple", shared,
        [asset_simple_by_interval[item] for item in shared],
    )
    add_series(
        "benchmark_aligned", "^GSPC", "simple", shared,
        [benchmark_simple_by_interval[item] for item in shared],
    )
    return list(series_by_id.values()), references


def _build_metric_evidence(normalized, snapshot=None, series_references=None):
    evidence = {}
    series_references = series_references or {}
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
        role,
        symbol,
        observations,
        series_role,
        start=market_window_start,
        end=market_window_end,
    ):
        if (
            start is None
            or end is None
            or type(observations) is not int
            or observations < 1
            or series_role not in series_references
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
            "series_id": series_references[series_role],
        }]

    sec_metrics = getattr(normalized, "metrics", {})
    combined_sec_metrics = {
        "pe_ratio": ("eps",),
        "pb_ratio": ("book_value_per_share",),
        "dividend_yield": ("dividend_per_share",),
        "market_cap": ("shares_outstanding",),
        "ps_ratio": ("shares_outstanding", "revenue"),
    }
    for metric, (source, formula) in _METRIC_EVIDENCE_CONTRACTS.items():
        value = normalized.get(metric)
        observations = None
        inputs = []
        reason = "required_inputs_unavailable"
        filed_at = None
        if source == "sec":
            canonical = sec_metrics.get(metric)
            if canonical is not None:
                value = canonical.value
                formula = canonical.formula
                inputs = sec_inputs(canonical)
                reason = canonical.availability_reason or "required_provenance_unavailable"
                as_of = (
                    max(item.provenance.end for item in canonical.inputs)
                    if canonical.inputs
                    else None
                )
                filed_at = (
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
                    "asset_aligned",
                    aligned_start,
                    aligned_end,
                )
                inputs += market_series(
                    "benchmark.sp500.adjusted_close_return_intervals",
                    "^GSPC",
                    observations,
                    "benchmark_aligned",
                    aligned_start,
                    aligned_end,
                )
            else:
                observations = getattr(snapshot, "asset_return_observations", None)
                inputs = market_series(
                    "asset.adjusted_close_return_intervals",
                    normalized.get("ticker"),
                    observations,
                    "asset_log",
                )
            if not metric_is_available(value):
                inputs = []
                as_of = None
                observations = None
            available = metric_is_available(value) and bool(inputs)
            version = {
                "beta": "goatlens_beta_v1",
                "volatility_252d": "goatlens_volatility_252d_v1",
                "correlation_sp500": "goatlens_correlation_sp500_v1",
                "realized_total_return_252d": (
                    "goatlens_realized_total_return_252d_v1"
                ),
            }.get(metric, "yahoo_chart_v1")
        else:
            canonical_inputs = [
                sec_metrics.get(name) for name in combined_sec_metrics[metric]
            ]
            available_canonicals = [
                item
                for item in canonical_inputs
                if item is not None and item.available
            ]
            if len(available_canonicals) == len(canonical_inputs):
                inputs = [
                    typed_input
                    for item in available_canonicals
                    for typed_input in sec_inputs(item)
                ]
                as_of = max(
                    item.provenance.end
                    for canonical in available_canonicals
                    for item in canonical.inputs
                )
                filed_at = max(
                    item.provenance.filed
                    for canonical in available_canonicals
                    for item in canonical.inputs
                ).isoformat()
            else:
                as_of = None
            inputs += market_observation(normalized.get("current_price"))
            observations = 1 if any(
                item["kind"] == "market_observation" for item in inputs
            ) else None
            available = (
                metric_is_available(value)
                and bool(inputs)
                and len(available_canonicals) == len(canonical_inputs)
            )
            version = "sec_yahoo_formula_v1"

        reason = normalized.get(f"{metric}_availability_reason") or reason

        evidence[metric] = {
            "value": float(value) if available else None,
            "available": available,
            "source": source,
            "as_of": as_of,
            "filed_at": filed_at,
            "formula": formula,
            "formula_version": version,
            "formula_owner": "yahoo" if metric == "current_price" else "goatlens",
            "inputs": inputs,
            "observation_count": observations,
            "availability_reason": None if available else reason,
        }
    return evidence


def _price_dependent_analysis_status(normalized):
    valuation_inputs = {
        "current_price": normalized.get("current_price"),
        "eps": normalized.get("eps"),
        "book_value_per_share": normalized.get("book_value_per_share"),
        "shares_outstanding": normalized.get("shares_outstanding"),
        "revenue": normalized.get("revenue"),
    }
    missing_inputs = [
        name for name, value in valuation_inputs.items() if value is None
    ]
    valuation_outputs = {
        "pe_ratio": normalized.get("pe_ratio"),
        "pb_ratio": normalized.get("pb_ratio"),
        "ps_ratio": normalized.get("ps_ratio"),
    }
    missing_outputs = [
        name for name, value in valuation_outputs.items() if value is None
    ]
    return {
        "status": "incomplete" if missing_inputs or missing_outputs else "complete",
        "reason": (
            "missing_inputs:" + ",".join(missing_inputs)
            if missing_inputs
            else "missing_outputs:" + ",".join(missing_outputs)
            if missing_outputs
            else None
        ),
    }


def _price_sec_compatibility_reason(ticker, normalized, snapshot):
    if "." in ticker or "-" in ticker:
        return "unproven_share_class_price_sec_compatibility"
    if snapshot.currency != "USD":
        return (
            "non_usd_quote_currency"
            if snapshot.currency
            else "unproven_quote_currency"
        )
    shares_metric = getattr(normalized, "metrics", {}).get("shares_outstanding")
    if shares_metric is None or not shares_metric.available:
        return "unproven_price_sec_compatibility"
    supported = all(
        item.provenance.namespace == "us-gaap"
        and item.provenance.form in {"10-K", "10-K/A"}
        and item.provenance.concept == "CommonStockSharesOutstanding"
        and item.provenance.unit == "shares"
        for item in shares_metric.inputs
    )
    if not supported:
        return "unsupported_filing_or_security_basis"
    split_events = getattr(snapshot, "split_events", None)
    if split_events is None:
        return "unproven_price_sec_compatibility"
    share_period_end = max(
        item.provenance.end for item in shares_metric.inputs
    )
    if any(event.effective_date > share_period_end for event in split_events):
        return "post_filing_split_unadjusted_sec_basis"
    return None


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
            "retrieved_at": None,
            "currency": None,
            "cache_age_seconds": None,
            "freshness": "unknown",
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

    snapshot = None
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
        price_sec_compatibility_reason = _price_sec_compatibility_reason(
            ticker, normalized, snapshot
        )
        if price_sec_compatibility_reason is not None:
            for metric in (
                "pe_ratio",
                "pb_ratio",
                "ps_ratio",
                "market_cap",
                "dividend_yield",
            ):
                normalized[metric] = None
                normalized[f"{metric}_availability_reason"] = (
                    price_sec_compatibility_reason
                )
        elif current_price is not None:
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
            "retrieved_at": snapshot.fetched_at,
            "currency": snapshot.currency or None,
            "cache_age_seconds": snapshot.cache_age_seconds,
            "freshness": "fresh",
            "reason": None if current_price is not None else "current_market_price_unavailable",
        }
        state["data_status"]["price_dependent_analysis"] = (
            _price_dependent_analysis_status(normalized)
        )
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

    if snapshot is not None:
        state["data_status"]["price_dependent_analysis"] = (
            _price_dependent_analysis_status(normalized)
        )

    state["normalized_data"] = normalized
    snapshot_series, series_references = _build_snapshot_series(snapshot, ticker)
    metric_evidence = _build_metric_evidence(
        normalized,
        snapshot,
        series_references,
    )
    referenced_series_ids = {
        typed_input["series_id"]
        for evidence in metric_evidence.values()
        if evidence["available"]
        for typed_input in evidence["inputs"]
        if typed_input["kind"] == "market_series"
    }
    state["snapshot_series"] = [
        series
        for series in snapshot_series
        if series["series_id"] in referenced_series_ids
    ]
    state["metric_evidence"] = metric_evidence

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
    key: contract["mandatory"]
    for key, contract in _AGENT_CONSUMPTION_CONTRACTS.items()
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
    task_metric_values = []
    for name, agent in agents.items():
        contract = _AGENT_CONSUMPTION_CONTRACTS.get(name)
        required_metrics = (
            _mandatory_consumed_metrics(contract, financials) if contract else ()
        )
        evidence = state.get("metric_evidence") or {}
        unavailable = []
        required_metric_values = {}
        consumed_candidates = required_metrics + (
            contract["optional"] if contract else ()
        )
        agent_financials = dict(financials)
        for metric in consumed_candidates:
            value = financials.get(metric)
            metric_evidence = evidence.get(metric)
            evidence_value = (
                metric_evidence.get("value")
                if isinstance(metric_evidence, dict)
                else None
            )
            evidence_matches = (
                isinstance(metric_evidence, dict)
                and metric_evidence.get("available") is True
                and metric_is_available(evidence_value)
                and metric_is_available(value)
                and value is not None
                and evidence_value is not None
                and math.isclose(value, evidence_value, rel_tol=0.0, abs_tol=1e-12)
            )
            if not evidence_matches:
                agent_financials[metric] = None
                if metric in required_metrics:
                    unavailable.append(metric)
            else:
                required_metric_values[metric] = float(value)
        if unavailable:
            task_metric_values.append({})
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

        async def _analyze(
            _, *, _agent=agent, _financials=agent_financials, config=None
        ):
            return await _agent.analyze(
                ticker, _financials,
                earnings_data=earnings_data,
                earnings_streak=earnings_streak,
                config=config,
            )
        
        runnable = RunnableLambda(_analyze).with_config({"run_name": f"agent.{name}"})
        tasks.append(runnable.ainvoke({"ticker": ticker, "agent": name}, config=config))
        task_metric_values.append(required_metric_values)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Preserve every requested agent. A failed agent is explicit and cannot
    # disappear from the consensus denominator.
    state["agent_results"] = []
    for (name, agent), result, consumed_metrics in zip(
        agents.items(), results, task_metric_values
    ):
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
                completed_payload = validated_result.model_dump()
                completed_payload["required_metric_values"] = consumed_metrics
                state["agent_results"].append(
                    AgentResult.model_validate(completed_payload).model_dump()
                )
    
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

    if completed_analyses < 2:
        state["consensus"] = {
            "status": "incomplete",
            "verdict": "incomplete",
            "agreement_score": None,
            "partial_indication": None,
            "consensus_points": [],
            "divergence_points": [
                "Consensus unavailable because fewer than two selected analyses "
                "are complete."
            ],
            "completed_analyses": completed_analyses,
            "requested_analyses": requested_analyses,
        }
    else:
        if llm:
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
        else:
            consensus = calculate_consensus(strategy_results)

        verdict = consensus.consensus_verdict.value
        if completed_analyses == requested_analyses:
            state["consensus"] = {
                "status": "complete",
                "verdict": verdict,
                "agreement_score": consensus.agreement_score,
                "partial_indication": None,
                "consensus_points": consensus.consensus_points,
                "divergence_points": (
                    [
                        "Single selected investor; final result reflects one completed opinion."
                    ]
                    if requested_analyses == 1
                    else consensus.divergence_points
                ),
                "completed_analyses": completed_analyses,
                "requested_analyses": requested_analyses,
            }
        else:
            direction_by_verdict = {
                "strong_sell": -1,
                "sell": -1,
                "hold": 0,
                "buy": 1,
                "strong_buy": 1,
            }
            consensus_direction = direction_by_verdict.get(verdict)
            supporting_analyses = (
                0
                if consensus_direction is None
                else sum(
                    direction_by_verdict[result["verdict"]] == consensus_direction
                    for result in complete_results
                )
            )
            state["consensus"] = {
                "status": "partial",
                "verdict": "incomplete",
                "agreement_score": None,
                "partial_indication": {
                    "verdict": verdict,
                    "directional_support": consensus.agreement_score,
                    "supporting_analyses": supporting_analyses,
                    "completed_analyses": completed_analyses,
                    "requested_analyses": requested_analyses,
                    "excluded_agents": [
                        {
                            "agent": result["agent"],
                            "unavailable_metrics": result.get(
                                "unavailable_metrics", []
                            ),
                        }
                        for result in agent_results
                        if result.get("status", "complete") != "complete"
                    ],
                    "consensus_points": consensus.consensus_points,
                    "divergence_points": consensus.divergence_points,
                },
                "consensus_points": [],
                "divergence_points": [
                    "Final consensus unavailable because "
                    f"{completed_analyses} of {requested_analyses} requested "
                    "analyses completed."
                ],
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
        "snapshot_series": state.get("snapshot_series") or [],
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
    version="0.2.0",
    lifespan=lifespan,
)

_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'sha256-3mMjJooNr/i21sq1ZM9hwrLZ/+wQWro9H60VSJDDSuA=' "
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
        snapshot_series=report.get("snapshot_series", []),
        consensus_status=consensus.get("status", "complete"),
        consensus_verdict=consensus["verdict"],
        agreement_score=consensus["agreement_score"],
        completed_analyses=consensus["completed_analyses"],
        requested_analyses=consensus["requested_analyses"],
        partial_indication=consensus.get("partial_indication"),
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
                required_metric_values=r.get("required_metric_values", {}),
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
            "adjusted_close": point.adjusted_close,
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
