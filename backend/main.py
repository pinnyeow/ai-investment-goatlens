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
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# Load environment variables from .env file
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the backend directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

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
from temporal import TemporalAnalyzer
from strategies import calculate_consensus, StrategyResult, Verdict


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


def get_years_from_period(period: str) -> List[int]:
    """Convert time period to anchor years for analysis."""
    current_year = datetime.now().year
    
    if period == "ytd":
        return [current_year - 1, current_year]
    elif period == "3m":
        return [current_year - 1, current_year]
    elif period == "6m":
        return [current_year - 1, current_year]
    elif period == "1y":
        return [current_year - 1, current_year]
    elif period == "5y":
        return [current_year - 5, current_year - 2, current_year]
    else:
        return [current_year - 1, current_year]


class AgentResult(BaseModel):
    """Result from a single GOAT agent."""
    agent: str
    style: str
    verdict: str
    score: float
    insights: List[str]
    concerns: List[str]


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
    anchor_year_comparison: Dict[str, Any]
    
    # GOAT comparison table
    comparison_table: Dict[str, Any]


# ============================================================================
# LangGraph State & Workflow
# ============================================================================

class GOATState(TypedDict):
    """State passed through the LangGraph workflow."""
    ticker: str
    time_period: str
    anchor_years: List[int]
    selected_agents: List[str]
    
    # Data
    raw_data: Optional[Dict[str, Any]]
    normalized_data: Optional[Dict[str, Any]]
    anchor_year_data: Optional[Dict[int, Dict[str, Any]]]
    
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
    Node: Fetch all financial data from Yahoo Finance.
    """
    ticker = state["ticker"]
    anchor_years = state["anchor_years"]
    
    try:
        async with YahooFinanceClient() as client:
            raw_data = await client.get_company_data(ticker, years=10)
            normalized = client.normalize_data(raw_data)
            anchor_data = client.extract_anchor_year_data(raw_data, anchor_years)
        
        state["raw_data"] = raw_data
        state["normalized_data"] = normalized.__dict__
        state["anchor_year_data"] = anchor_data
        
    except YahooFinanceError as e:
        state["error"] = f"Data fetch failed: {str(e)}"
    
    return state


async def temporal_analysis_node(state: GOATState) -> GOATState:
    """
    Node: Perform time-travel analysis.
    """
    if state.get("error"):
        return state
    
    analyzer = TemporalAnalyzer()
    anchor_years = state["anchor_years"]
    anchor_data = state["anchor_year_data"]
    
    if not anchor_data:
        state["temporal_results"] = {"error": "No anchor year data available"}
        return state
    
    comparison = analyzer.compare_anchor_years(anchor_years, anchor_data)
    
    state["temporal_results"] = {
        "comparison": comparison,
        "moat_trend": comparison["moat_analysis"].overall_trend.value,
        "moat_observations": comparison["moat_analysis"].key_observations,
        "risk_factors": comparison["moat_analysis"].risk_factors,
    }
    
    return state


async def run_agents_node(state: GOATState) -> GOATState:
    """
    Node: Run all GOAT agents in parallel.
    """
    if state.get("error"):
        return state
    
    ticker = state["ticker"]
    financials = state["normalized_data"]
    anchor_years = state["anchor_years"]
    selected = state["selected_agents"]
    
    # Initialize agents
    all_agents = {
        "buffett": BuffettAgent(),
        "lynch": LynchAgent(),
        "graham": GrahamAgent(),
        "munger": MungerAgent(),
        "dalio": DalioAgent(),
    }
    
    # Filter to selected agents (or use all)
    if selected:
        agents = {k: v for k, v in all_agents.items() if k in selected}
    else:
        agents = all_agents
    
    # Run all agents in parallel
    tasks = [
        agent.analyze(ticker, financials, anchor_years)
        for agent in agents.values()
    ]
    
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
    
    consensus = calculate_consensus(strategy_results)
    
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
    
    state["final_report"] = {
        "ticker": state["ticker"],
        "company_name": profile.get("companyName", state["ticker"]),
        "sector": profile.get("sector", "Unknown"),
        "consensus": state["consensus"],
        "agent_results": agent_results,
        "temporal_results": state.get("temporal_results", {}),
        "comparison_table": comparison_table,
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
    """Serve the frontend."""
    return FileResponse("../frontend/index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
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
    
    # Convert time period to anchor years
    time_period = request.time_period or "1y"
    anchor_years = get_years_from_period(time_period)
    
    # Initialize state
    initial_state: GOATState = {
        "ticker": request.ticker.upper(),
        "time_period": time_period,
        "anchor_years": anchor_years,
        "selected_agents": request.agents or [],
        "raw_data": None,
        "normalized_data": None,
        "anchor_year_data": None,
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
        anchor_year_comparison=temporal.get("comparison", {}),
        comparison_table=report["comparison_table"],
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
        "anchor_year_comparison": {
            "years": [2014, 2019, 2024],
            "metrics": {
                "revenue": {2014: 182795000000, 2019: 260174000000, 2024: 383285000000},
                "net_income": {2014: 39510000000, 2019: 55256000000, 2024: 96995000000},
                "gross_margin": {2014: 0.387, 2019: 0.378, 2024: 0.438},
            },
            "revenue_cagr": 0.077,
        },
        "comparison_table": {
            "agents": [
                {"name": "Warren Buffett", "style": "Value", "verdict": "buy", "score": 65},
                {"name": "Peter Lynch", "style": "GARP", "verdict": "hold", "score": 35},
                {"name": "Benjamin Graham", "style": "Deep Value", "verdict": "hold", "score": 15},
                {"name": "Charlie Munger", "style": "Quality", "verdict": "strong_buy", "score": 78},
                {"name": "Ray Dalio", "style": "Macro", "verdict": "buy", "score": 55},
            ]
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
