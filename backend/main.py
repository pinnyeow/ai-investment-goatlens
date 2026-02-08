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
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from opentelemetry import context as otel_context

# Load environment variables from .env file
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the backend directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- Arize AX Auto-Instrumentation (MUST be before LangGraph imports) ---
if os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY"):
    try:
        from arize.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        tp = register(space_id=os.getenv("ARIZE_SPACE_ID"), api_key=os.getenv("ARIZE_API_KEY"), project_name="goatlens")
        LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
        print("Arize AX tracing enabled for project 'goatlens'")
    except Exception as e:
        print(f"Arize tracing setup failed: {e}")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Local imports
from agents import BuffettAgent, LynchAgent, GrahamAgent, MungerAgent, DalioAgent
from data_sources import YahooFinanceClient, YahooFinanceError
from data_sources import FMPClient, FMPError
from data_sources import NewsClient, NewsError
from temporal import TemporalAnalyzer
from strategies import calculate_consensus, calculate_consensus_with_llm, StrategyResult, Verdict
from llm import get_llm_client
from memory import MemoryStore
from rag import RAGRetriever


# ============================================================================
# Pydantic Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request to analyze a company."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    time_period: str = Field(
        default="1y",
        description="Time period for analysis: ytd, 3m, 6m, 1y, 5y",
    )
    agents: Optional[List[str]] = Field(
        default=None,
        description="Specific agents to run (default: all)",
    )


class AgentResult(BaseModel):
    """Result from a single GOAT agent."""
    agent: str
    style: str
    verdict: str
    score: float
    insights: List[str]
    concerns: List[str]


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
    # FMP guidance fields (None when FMP unavailable — graceful degradation)
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
    growth_trend: str = "unknown"
    analyst_price_target_mean: Optional[float] = None
    analyst_price_target_upside_pct: Optional[float] = None
    num_analysts: Optional[int] = None
    # New: full estimate ranges and confidence
    estimate_ranges: List[EstimateRange] = []
    consensus_confidence: str = "unknown"
    price_targets: Optional[PriceTargetRange] = None
    summary: str = ""


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    ticker: str
    company_name: str
    sector: str
    
    # Consensus
    consensus_verdict: str
    agreement_score: float
    consensus_points: List[str]
    divergence_points: List[str]
    
    # Individual agents
    agent_results: List[AgentResult]
    
    # Temporal analysis
    moat_trend: str
    
    # GOAT comparison table
    comparison_table: Dict[str, Any]
    
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


async def fetch_data_node(state: GOATState) -> GOATState:
    """
    Node: Fetch all financial data from Yahoo Finance,
    including earnings history and next earnings date.

    Fetches company data + next earnings in parallel so
    synthesize_node doesn't need a redundant Yahoo call.
    """
    ticker = state["ticker"]
    
    try:
        async with YahooFinanceClient() as client:
            # Fetch company data and next earnings date in parallel
            raw_task = client.get_company_data(ticker, years=10)
            next_earnings_task = client.get_next_earnings_date(ticker)
            raw_data, next_earnings = await asyncio.gather(
                raw_task, next_earnings_task
            )

            normalized = client.normalize_data(raw_data)
            
            # Extract earnings history (EPS actual vs. estimate)
            earnings = client.normalize_earnings(raw_data)
            streak = client.get_earnings_streak(earnings)
        
        state["raw_data"] = raw_data
        state["normalized_data"] = normalized.__dict__
        state["earnings_data"] = earnings
        state["earnings_streak"] = streak
        state["next_earnings_date"] = next_earnings
        
        # Optional: Fetch news sentiment (tool selection - only if conditions met)
        # This demonstrates Step 6: Tool Calling - agents decide when to use this tool
        try:
            news_client = NewsClient()
            # Tool selection: only fetch if conditions are met (high volatility, etc.)
            if await news_client.should_use_tool(ticker, normalized.__dict__):
                news_sentiment = await news_client.get_sentiment(
                    ticker,
                    company_name=normalized.company_name,
                    days=7
                )
                state["news_sentiment"] = news_sentiment
            else:
                state["news_sentiment"] = None
        except Exception as e:
            # Graceful degradation: news is optional
            state["news_sentiment"] = None
        
        # Optional: Retrieve RAG context (Step 8: RAG)
        # This enriches agent analysis with earnings transcripts and strategic insights
        try:
            rag_retriever = RAGRetriever()
            rag_context = await rag_retriever.retrieve_earnings_context(ticker, quarters=4)
            state["rag_context"] = rag_context if rag_context else None
        except Exception as e:
            # Graceful degradation: RAG is optional
            state["rag_context"] = None
        
    except YahooFinanceError as e:
        state["error"] = f"Data fetch failed: {str(e)}"
    
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


async def run_agents_node(state: GOATState) -> GOATState:
    """
    Node: Run all GOAT agents in parallel.
    Now passes earnings data so agents can factor beat/miss history.
    """
    if state.get("error"):
        return state
    
    ticker = state["ticker"]
    financials = state["normalized_data"]
    selected = state["selected_agents"]
    earnings_data = state.get("earnings_data") or []
    earnings_streak = state.get("earnings_streak") or {}
    news_sentiment = state.get("news_sentiment")  # Optional tool - agents decide when to use
    
    # Initialize agents with model routing
    # Each agent can specify a preferred model (e.g., Buffett uses gpt-4o for nuanced reasoning)
    # If OPENAI_API_KEY is not set, all agents work without LLM (rule-based fallback)
    all_agents = {}
    if os.getenv("OPENAI_API_KEY"):
        # Model routing: route each agent to its preferred model
        all_agents = {
            "buffett": BuffettAgent(llm_client=get_llm_client(getattr(BuffettAgent, "model_preference", "gpt-4o-mini"))),
            "lynch": LynchAgent(llm_client=get_llm_client(getattr(LynchAgent, "model_preference", "gpt-4o-mini"))),
            "graham": GrahamAgent(llm_client=get_llm_client(getattr(GrahamAgent, "model_preference", "gpt-4o-mini"))),
            "munger": MungerAgent(llm_client=get_llm_client(getattr(MungerAgent, "model_preference", "gpt-4o-mini"))),
            "dalio": DalioAgent(llm_client=get_llm_client(getattr(DalioAgent, "model_preference", "gpt-4o-mini"))),
        }
    else:
        # No API key: all agents work rule-based (no LLM)
        all_agents = {
            "buffett": BuffettAgent(llm_client=None),
            "lynch": LynchAgent(llm_client=None),
            "graham": GrahamAgent(llm_client=None),
            "munger": MungerAgent(llm_client=None),
            "dalio": DalioAgent(llm_client=None),
        }
    
    # Filter to selected agents (or use all)
    if selected:
        agents = {k: v for k, v in all_agents.items() if k in selected}
    else:
        agents = all_agents
    
    # Run all agents in parallel with context propagation
    current_ctx = otel_context.get_current()
    
    async def run_with_context(agent):
        token = otel_context.attach(current_ctx)
        try:
            return await agent.analyze(ticker, financials, earnings_data=earnings_data, earnings_streak=earnings_streak)
        finally:
            otel_context.detach(token)
    
    tasks = [asyncio.create_task(run_with_context(agent)) for agent in agents.values()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    agent_results = []
    for result in results:
        if isinstance(result, Exception):
            continue
        agent_results.append(result)
    
    state["agent_results"] = agent_results
    
    return state


async def synthesize_node(state: GOATState) -> GOATState:
    """
    Node: Synthesize results and calculate consensus.
    """
    if state.get("error"):
        return state
    
    agent_results = state["agent_results"]
    
    if not agent_results:
        state["error"] = "No agent results to synthesize"
        return state
    
    # Convert to StrategyResult for consensus calculation
    strategy_results = []
    for result in agent_results:
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
    consensus = await calculate_consensus_with_llm(strategy_results, llm, state["ticker"]) if llm else calculate_consensus(strategy_results)
    
    state["consensus"] = {
        "verdict": consensus.consensus_verdict.value,
        "agreement_score": consensus.agreement_score,
        "consensus_points": consensus.consensus_points,
        "divergence_points": consensus.divergence_points,
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
    
    # Build final report
    raw_data = state.get("raw_data", {})
    profile = raw_data.get("profile", {})
    # Yahoo Finance uses 'info' instead of 'profile'
    info = raw_data.get("info", {})
    
    # next_earnings_date already fetched in fetch_data_node — no extra call needed
    state["final_report"] = {
        "ticker": state["ticker"],
        "company_name": info.get("longName", info.get("shortName", profile.get("companyName", state["ticker"]))),
        "sector": info.get("sector", profile.get("sector", "Unknown")),
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
    1. Fetch Data (FMP API)
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """
    return {"status": "healthy", "service": "goatlens"}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_company(request: AnalysisRequest):
    """
    Run full GOAT analysis on a company.
    
    This endpoint:
    1. Fetches 10 years of financial data
    2. Performs time-travel analysis across anchor years
    3. Runs all GOAT agents in parallel
    4. Synthesizes results into consensus/divergence
    
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
    }
    
    # Run workflow
    try:
        final_state = await goat_workflow.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    # Check for errors
    if final_state.get("error"):
        raise HTTPException(status_code=400, detail=final_state["error"])
    
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
    
    # Record analysis in memory (Step 9: Memory)
    try:
        memory = MemoryStore()
        selected_agents = request.agents or ["buffett", "lynch", "graham", "munger", "dalio"]
        memory.record_analysis(
            ticker=request.ticker.upper(),
            consensus_verdict=consensus.get("verdict", "hold") if consensus else "hold",
            consensus_score=consensus.get("score", 0.0) if consensus else 0.0,
            selected_agents=selected_agents,
            key_metrics={
                "pe_ratio": final_state.get("normalized_data", {}).get("pe_ratio"),
                "roe": final_state.get("normalized_data", {}).get("roe"),
                "profit_margin": final_state.get("normalized_data", {}).get("profit_margin"),
            },
        )
    except Exception as e:
        # Memory is optional - don't fail if it errors
        print(f"[Memory] Failed to record analysis: {e}")
    
    return AnalysisResponse(
        ticker=report["ticker"],
        company_name=report["company_name"],
        sector=report["sector"],
        consensus_verdict=consensus["verdict"],
        agreement_score=consensus["agreement_score"],
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


@app.get("/api/earnings/{ticker}")
async def get_earnings(ticker: str):
    """
    Get earnings history for a ticker (standalone endpoint).
    
    Returns last 8 quarters of EPS actual vs. consensus estimate,
    beat/miss classification, streak summary, price reactions,
    per-quarter reaction insights (enriched with FMP revenue + CapEx
    when FMP_API_KEY is set), forward guidance summary, and the next
    upcoming earnings date.

    Primary data: Yahoo Finance (free, no API key)
    Supplementary: FMP (revenue actuals + CapEx — graceful degradation)
    """
    ticker = ticker.upper()
    
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

            # ── FMP supplementary data (graceful degradation) ──
            # If FMP_API_KEY is set, fetch revenue actuals + CapEx.
            # If not, fmp_guidance stays empty and the frontend shows
            # EPS-only cards — the core experience is unaffected.
            fmp_guidance = None
            fmp_api_key = os.environ.get("FMP_API_KEY")
            if fmp_api_key:
                try:
                    async with FMPClient(api_key=fmp_api_key) as fmp:
                        fmp_guidance = await fmp.get_quarterly_guidance_data(ticker)
                except Exception as fmp_err:
                    # Log but don't fail — Yahoo data is sufficient
                    print(f"[FMP] Guidance data fetch failed for {ticker}: {fmp_err}")

            # Build per-quarter reaction insights (with FMP enrichment)
            reaction_insights = client.build_earnings_insights(
                earnings, price_history, analyst_reactions, quarterly_revenue,
                fmp_guidance=fmp_guidance,
            )

            # Build forward guidance summary
            normalized = client.normalize_data(raw_data)
            forward_guidance = client.build_forward_guidance_summary(
                forward_estimates, current_price=normalized.current_price,
            )
    except YahooFinanceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Earnings fetch failed: {str(e)}")

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
    ticker = ticker.upper()
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
    except YahooFinanceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch price data: {str(e)}")
    
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


@app.get("/api/demo")
async def demo_analysis():
    """
    Return demo analysis for testing UI without API key.
    
    Uses static data for AAPL.
    """
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "consensus_verdict": "buy",
        "agreement_score": 0.72,
        "consensus_points": [
            "Strong brand moat and pricing power",
            "Excellent capital allocation and shareholder returns",
            "Services segment providing recurring revenue growth",
        ],
        "divergence_points": [
            "Graham finds valuation too high relative to book value",
            "Dalio notes concentration risk in consumer electronics",
        ],
        "agent_results": [
            {
                "agent": "Warren Buffett",
                "style": "Value Investing with Quality Focus",
                "verdict": "buy",
                "score": 65,
                "insights": [
                    "Strong ROE of 147% indicates efficient capital deployment",
                    "Exceptional profit margin of 25.3% suggests pricing power",
                    "Services segment provides recurring, high-margin revenue",
                ],
                "concerns": [
                    "Concentration in iPhone revenue stream",
                ],
            },
            {
                "agent": "Peter Lynch",
                "style": "Growth at Reasonable Price (GARP)",
                "verdict": "hold",
                "score": 35,
                "insights": [
                    "Stalwart with consistent earnings",
                    "Low institutional ownership upside exhausted",
                ],
                "concerns": [
                    "PEG ratio of 2.8 suggests growth priced in",
                    "Heavy institutional ownership - fully discovered",
                ],
            },
            {
                "agent": "Benjamin Graham",
                "style": "Deep Value / Margin of Safety",
                "verdict": "hold",
                "score": 15,
                "insights": [
                    "Strong balance sheet with current ratio of 1.0",
                ],
                "concerns": [
                    "High P/E of 28 exceeds Graham's maximum of 15",
                    "High P/B ratio of 45.8",
                    "Limited margin of safety at current prices",
                ],
            },
            {
                "agent": "Charlie Munger",
                "style": "Quality-Focused Mental Models",
                "verdict": "strong_buy",
                "score": 78,
                "insights": [
                    "Exceptional quality business",
                    "Outstanding capital allocation with 25%+ ROIC",
                    "Strong pricing power evident in 43% gross margin",
                ],
                "concerns": [],
            },
            {
                "agent": "Ray Dalio",
                "style": "Macro & Risk Parity",
                "verdict": "buy",
                "score": 55,
                "insights": [
                    "Sustainable debt levels provide flexibility",
                    "Beta near 1.0 indicates market-like risk profile",
                ],
                "concerns": [
                    "High correlation to market reduces diversification benefit",
                ],
            },
        ],
        "moat_trend": "stable",
        "comparison_table": {
            "agents": [
                {"name": "Warren Buffett", "style": "Value", "verdict": "buy", "score": 65},
                {"name": "Peter Lynch", "style": "GARP", "verdict": "hold", "score": 35},
                {"name": "Benjamin Graham", "style": "Deep Value", "verdict": "hold", "score": 15},
                {"name": "Charlie Munger", "style": "Quality", "verdict": "strong_buy", "score": 78},
                {"name": "Ray Dalio", "style": "Macro", "verdict": "buy", "score": 55},
            ]
        },
        "earnings_history": [
            {"quarter": "Q4 2025", "date": "2026-01-30", "eps_actual": 2.40, "eps_estimate": 2.36, "eps_surprise": 0.04, "surprise_pct": 1.7, "beat_miss": "beat"},
            {"quarter": "Q3 2025", "date": "2025-10-30", "eps_actual": 1.64, "eps_estimate": 1.60, "eps_surprise": 0.04, "surprise_pct": 2.5, "beat_miss": "beat"},
            {"quarter": "Q2 2025", "date": "2025-08-01", "eps_actual": 1.40, "eps_estimate": 1.35, "eps_surprise": 0.05, "surprise_pct": 3.7, "beat_miss": "beat"},
            {"quarter": "Q1 2025", "date": "2025-05-01", "eps_actual": 1.65, "eps_estimate": 1.62, "eps_surprise": 0.03, "surprise_pct": 1.9, "beat_miss": "beat"},
            {"quarter": "Q4 2024", "date": "2025-01-30", "eps_actual": 2.18, "eps_estimate": 2.14, "eps_surprise": 0.04, "surprise_pct": 1.9, "beat_miss": "beat"},
            {"quarter": "Q3 2024", "date": "2024-10-31", "eps_actual": 1.55, "eps_estimate": 1.50, "eps_surprise": 0.05, "surprise_pct": 3.3, "beat_miss": "beat"},
        ],
        "earnings_streak": {
            "streak_type": "beat",
            "streak_count": 6,
            "beats": 6,
            "misses": 0,
            "inline": 0,
            "total": 6,
            "summary": "Strong momentum: Beat 6 consecutive quarters",
        },
        "next_earnings_date": "2026-04-30",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
