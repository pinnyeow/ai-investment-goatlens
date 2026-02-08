# GOATlens Architecture Guide

**For AI agents, new contributors, and future you.**

This document captures the critical knowledge needed to understand, modify, and extend GOATlens. It's structured for AI collaboration — clear patterns, gotchas, and rationale.

---

## 🚨 Critical Gotchas

These are the things that will trip you up if you don't know about them:

### 1. Frontend Path: `../frontend/index.html`

**The Issue:** FastAPI serves the frontend from `backend/main.py` using a relative path:
```python
return FileResponse("../frontend/index.html")
```

**Why it matters:** 
- The working directory is `backend/` when running `python main.py`
- Docker sets `WORKDIR /app/backend`, so the relative path works there too
- If you change the working directory, this breaks

**Fix:** Always run from `backend/` directory, or use absolute paths in Docker.

### 2. NaN JSON Serialization

**The Issue:** Yahoo Finance returns `float('nan')` and `float('inf')` values, which are valid in Python but **illegal in JSON**. FastAPI crashes with `ValueError: Out of range float values are not JSON compliant`.

**The Fix:** `_safe_float()` helper in `backend/data_sources/yahoo.py`:
```python
def _safe_float(val, default=0) -> Optional[float]:
    """Convert NaN/inf to safe default for JSON serialization."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default
```

**Where it's used:**
- `normalize_data()` - All metrics from Yahoo's info dict
- `normalize_earnings()` - EPS values (especially future dates with NaN)
- `get_price_history_sync()` - OHLCV values
- `calculate_earnings_reactions()` - Percentage calculations

**Lesson:** Always sanitize external data before JSON serialization.

### 3. FMP API Migration: `/api/v3` → `/stable`

**The Issue:** FMP migrated their API. Old code using `/api/v3/income-statement/AAPL` returns `403 Legacy Endpoint`.

**The Fix:** Updated `FMPClient` to use `/stable` base URL with query parameters:
```python
BASE_URL = "https://financialmodelingprep.com/stable"
# Old: /api/v3/income-statement/AAPL
# New: /stable?symbol=AAPL&query=income-statement
```

**Lesson:** External APIs change. Always check API docs when things break.

### 4. `lxml` Dependency for Earnings Scraping

**The Issue:** `yfinance` uses `pandas.read_html()` to scrape earnings data, which requires `lxml`. Without it, you get `ModuleNotFoundError: No module named 'lxml'`.

**The Fix:** Added `lxml` to `requirements.txt`.

**Why it's not obvious:** `yfinance` doesn't list `lxml` as a direct dependency, but it's needed for HTML parsing.

### 5. Render Cold Starts (30-50s delay)

**The Issue:** Render free tier sleeps your app after 15 minutes of inactivity. First visitor waits 30-50 seconds for the server to wake up.

**The Fix:** Use an uptime monitor (UptimeRobot, cron-job.org) to ping `/health` every 14 minutes. Keeps the server warm.

**Tradeoff:** Free tier limitation. Paid tiers don't have this issue.

### 6. Context Window Management

**The Issue:** Long conversations in Cursor hit context limits. The system resets when it gets too full.

**The Fix:** 
- Start new chats for new topics
- Use selective context (only include relevant files)
- The system automatically manages this, but be aware

---

## 🏗️ Architecture Decisions

### Why LangGraph?

**Answer:** State machine perfect for multi-agent orchestration.

**The Flow:**
```
fetch_data → temporal_analysis → run_agents → synthesize → END
```

**Benefits:**
- Clear data flow (state passed between nodes)
- Easy to add new nodes (e.g., RAG retrieval)
- Built-in error handling (state can carry `error` field)
- Parallel execution support (`asyncio.gather` in `run_agents_node`)

**Alternative considered:** Direct function calls. Rejected because:
- Harder to visualize workflow
- No built-in state management
- Difficult to add observability (Arize tracing)

### Why Separate Agent Files?

**Answer:** Each investor has a distinct philosophy — separation enables:
- **Parallel execution** - Agents don't depend on each other
- **Clear scoring logic** - Each file is self-contained
- **Easy modification** - Change Buffett's logic without touching Lynch
- **Testability** - Can test each agent independently

**Structure:**
```
backend/agents/
├── buffett.py    # ROE, margins, moats
├── lynch.py      # PEG, growth, ten-baggers
├── graham.py     # P/E, P/B, margin of safety
├── munger.py     # Quality, red flags
└── dalio.py      # Risk, debt cycles
```

**Tradeoff:** Some code duplication (each has `_score_to_verdict()`), but worth it for clarity.

### Why TTL Cache?

**Answer:** Yahoo Finance scraping is slow (1-2s per call). Multiple endpoints request the same ticker.

**The Problem:**
- `/api/analyze` requests AAPL data
- `/api/price-history/AAPL` requests AAPL data
- `/api/earnings/AAPL` requests AAPL data
- **Result:** 3 separate scrapes of the same ticker = 3-6 seconds wasted

**The Solution:** Module-level TTL cache (5 minutes):
```python
_cache: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL = 300  # 5 minutes
```

**Impact:**
- First call: 1.78 seconds (cold)
- Second call: 0.01 seconds (cache hit) = **178x faster**

**Why 5 minutes?** Balance between freshness (earnings update daily) and performance.

### Why FastAPI?

**Answer:** Async support + easy static file serving.

**Benefits:**
- **Async/await** - Perfect for parallel agent execution (`asyncio.gather`)
- **Pydantic models** - Automatic request/response validation
- **Static files** - Serves `frontend/index.html` with one line
- **CORS middleware** - Easy cross-origin setup for local dev

**Alternative considered:** Flask. Rejected because:
- Less async support
- More boilerplate for validation
- Static file serving requires more setup

### Why Data Source Abstraction?

**Answer:** Easy to swap data sources without changing agent code.

**Structure:**
```
backend/data_sources/
├── yahoo.py    # Free, no API key (scraping)
└── fmp.py      # Paid, API key (structured data)
```

**Benefits:**
- Agents don't know/care where data comes from
- Can add new sources (e.g., Alpha Vantage) without touching agents
- Graceful degradation (FMP fails → continue with Yahoo only)

**Pattern:** All data sources implement similar interfaces:
- `get_company_data(ticker)` - Main entry point
- `normalize_data(raw_data)` - Convert to standard format
- Error handling with custom exceptions (`YahooFinanceError`, `FMPError`)

### Why LLM Client Singleton?

**Answer:** One LLM client instance, shared across all agents.

**Implementation:**
```python
_client: Optional["LLMClient"] = None

def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
```

**Benefits:**
- **Cost efficiency** - Reuse connection/context
- **Consistent configuration** - All agents use same model/temperature
- **Tracing** - Arize can track all LLM calls from one client

**Tradeoff:** All agents use the same model. For model routing (Step 5-6), we'll modify this.

---

## 📊 Data Flow

### High-Level Flow

```
User Request (ticker: "AAPL")
    ↓
FastAPI /api/analyze endpoint
    ↓
LangGraph Workflow (GOATState)
    │
    ├─→ fetch_data_node
    │   ├─→ YahooFinanceClient.get_company_data() [Tool: Yahoo scraping]
    │   ├─→ YahooFinanceClient.get_next_earnings_date() [Tool: Yahoo scraping]
    │   ├─→ FMPClient.get_quarterly_guidance_data() [Tool: FMP API] (optional)
    │   └─→ Store in state: raw_data, normalized_data, earnings_data, next_earnings_date
    │
    ├─→ temporal_analysis_node
    │   ├─→ TemporalAnalyzer.calculate_moat_from_period()
    │   └─→ Store in state: temporal_results (moat_trend, moat_score)
    │
    ├─→ run_agents_node
    │   ├─→ Initialize 5 agents (Buffett, Lynch, Graham, Munger, Dalio)
    │   ├─→ Run all agents in parallel (asyncio.gather)
    │   │   Each agent.analyze(ticker, financials, earnings_data, earnings_streak)
    │   │   ├─→ Calculate metrics
    │   │   ├─→ Apply investment philosophy
    │   │   ├─→ Generate score (-100 to +100)
    │   │   ├─→ Convert score to verdict (strong_buy, buy, hold, sell, strong_sell)
    │   │   └─→ Generate insights/concerns (optional LLM call)
    │   └─→ Store in state: agent_results (list of 5 agent outputs)
    │
    └─→ synthesize_node
        ├─→ calculate_consensus_with_llm() or calculate_consensus()
        ├─→ Build comparison_table
        ├─→ Build final_report
        └─→ Store in state: consensus, final_report
            ↓
FastAPI Response (AnalysisResponse)
    ↓
Frontend displays results
```

### State Flow (GOATState)

```python
{
    "ticker": "AAPL",
    "time_period": "1y",
    "selected_agents": [],
    
    # After fetch_data_node
    "raw_data": {...},              # Raw Yahoo/FMP data
    "normalized_data": {...},       # Standardized financials
    "earnings_data": [...],         # Last 8 quarters
    "earnings_streak": {...},       # Beat/miss summary
    "next_earnings_date": "2026-04-30",
    
    # After temporal_analysis_node
    "temporal_results": {
        "moat_trend": "stable",
        "moat_score": 75.5
    },
    
    # After run_agents_node
    "agent_results": [
        {"agent": "buffett", "score": 65, "verdict": "buy", ...},
        {"agent": "lynch", "score": 35, "verdict": "hold", ...},
        ...
    ],
    
    # After synthesize_node
    "consensus": {
        "verdict": "buy",
        "agreement_score": 0.72,
        "consensus_points": [...],
        "divergence_points": [...]
    },
    "final_report": {...}
}
```

### Parallel Execution Points

1. **Data Fetching** (fetch_data_node):
   ```python
   raw_task = client.get_company_data(ticker, years=10)
   next_earnings_task = client.get_next_earnings_date(ticker)
   raw_data, next_earnings = await asyncio.gather(raw_task, next_earnings_task)
   ```

2. **Agent Execution** (run_agents_node):
   ```python
   tasks = [asyncio.create_task(run_with_context(agent)) for agent in agents.values()]
   results = await asyncio.gather(*tasks, return_exceptions=True)
   ```

3. **Frontend API Calls** (browser):
   ```javascript
   const analyzePromise = fetch('/api/analyze', {...});
   const pricePromise = fetch('/api/price-history/AAPL');
   const earningsPromise = fetch('/api/earnings/AAPL');
   // All fire in parallel, analyzePromise is awaited first
   ```

---

## 📁 File Structure Rationale

```
ai-investment-goatlens/
├── backend/
│   ├── main.py                 # FastAPI app + LangGraph workflow
│   ├── agents/                 # Each agent is a separate file
│   │   ├── buffett.py         # Self-contained: metrics, scoring, insights
│   │   ├── lynch.py
│   │   ├── graham.py
│   │   ├── munger.py
│   │   └── dalio.py
│   ├── data_sources/           # Abstraction layer for external APIs
│   │   ├── yahoo.py            # Free data (scraping)
│   │   └── fmp.py              # Paid data (structured API)
│   ├── llm/                    # LLM client (singleton pattern)
│   │   └── client.py
│   ├── strategies/             # Consensus calculation logic
│   │   └── __init__.py
│   └── temporal/               # Time-travel analysis
│       └── anchor_years.py
├── frontend/
│   └── index.html              # Single-page app (no build step)
├── Dockerfile                  # Container definition
├── render.yaml                 # Render deployment config
└── ARCHITECTURE.md             # This file
```

**Design Principles:**
- **Separation of concerns** - Agents, data, LLM, strategies are separate
- **Easy to extend** - Add new agent = new file, no changes to existing code
- **No magic** - Everything is explicit (no auto-discovery, no decorators)

---

## 🔄 Common Patterns

### Pattern 1: Graceful Degradation

**Example:** FMP API is optional
```python
try:
    async with FMPClient(api_key=fmp_api_key) as fmp:
        fmp_guidance = await fmp.get_quarterly_guidance_data(ticker)
except Exception as fmp_err:
    # Log but don't fail — Yahoo data is sufficient
    print(f"[FMP] Guidance data fetch failed: {fmp_err}")
    fmp_guidance = None
```

**Why:** Core experience (Yahoo data) works without FMP. FMP enriches but doesn't break.

### Pattern 2: Optional LLM

**Example:** Agents work with or without LLM
```python
llm = get_llm_client() if os.getenv("OPENAI_API_KEY") else None
agent = BuffettAgent(llm_client=llm)

# Inside agent.analyze():
if self.llm_client:
    insights = await self._generate_llm_insights(...)
else:
    insights = self._generate_insights(...)  # Rule-based fallback
```

**Why:** System works without API key. LLM enhances but isn't required.

### Pattern 3: Data Normalization

**Example:** Convert external data to standard format
```python
# Yahoo returns: {"marketCap": 3000000000000, "trailingPE": 28.5}
# We normalize to: FinancialData(market_cap=3000000000000, pe_ratio=28.5)
```

**Why:** Agents don't need to know Yahoo's field names. Change data source = change normalization, not agents.

---

## 🧪 Testing Patterns

### How to Test an Agent

```python
from agents import BuffettAgent

agent = BuffettAgent(llm_client=None)  # No LLM for deterministic tests
result = await agent.analyze(
    ticker="AAPL",
    financials={"roe": 0.20, "profit_margin": 0.15, "debt_to_equity": 0.3},
    earnings_data=[],
    earnings_streak={}
)
assert result["verdict"] == "buy"
assert result["score"] > 20
```

### How to Test Data Sources

```python
from data_sources import YahooFinanceClient

async with YahooFinanceClient() as client:
    data = await client.get_company_data("AAPL")
    assert data["ticker"] == "AAPL"
    assert "info" in data
```

---

## 🚀 Performance Optimizations

### 1. TTL Cache (5 minutes)
- **What:** Module-level cache for Yahoo Finance data
- **Impact:** 178x faster on cache hits
- **Location:** `backend/data_sources/yahoo.py`

### 2. Parallel API Calls
- **What:** Frontend fires 3 API calls simultaneously
- **Impact:** Charts ready by the time main analysis completes
- **Location:** `frontend/index.html` (runAnalysis function)

### 3. Parallel Agent Execution
- **What:** All 5 agents run simultaneously
- **Impact:** 5x faster than sequential
- **Location:** `backend/main.py` (run_agents_node)

### 4. Eliminated Redundant Calls
- **What:** `next_earnings_date` fetched once in `fetch_data_node`, not again in `synthesize_node`
- **Impact:** Saves 1-2 seconds per analysis
- **Location:** `backend/main.py`

---

## 🔮 Future Enhancements (Steps 8-10)

### Step 8: RAG
- Add vector store for earnings transcripts
- Retrieve relevant context before agent analysis
- Enrich insights with CEO commentary

### Step 9: Memory
- Remember user's favorite agents
- Store last 5 analyses for comparison
- Track user preferences

### Step 10: Context Engineering
- Selective context per agent (Graham doesn't need growth metrics)
- Progressive disclosure (only send earnings if agent needs it)
- Context pruning (summarize old data if context too long)

---

## 📚 Key Concepts from AI Product Sense

### Context Engineering
You're already doing this! Passing `financials`, `earnings_data`, `earnings_streak` to agents is context engineering — filling the context window with relevant data.

### Tool Calling
Your "tools" are: Yahoo Finance scraping, FMP API, LLM calls. They're orchestrated by LangGraph workflow (not directly by agents), similar to how Cursor uses tools.

### Model Routing
Currently all agents use `gpt-4o-mini`. Step 5-6 will experiment with routing different models to different agents based on complexity.

### Graceful Degradation
System works without FMP API key, without OpenAI API key. Core experience is always available.

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'lxml'"
**Fix:** `pip install lxml` or ensure it's in `requirements.txt`

### "ValueError: Out of range float values are not JSON compliant"
**Fix:** Check that `_safe_float()` is applied to all external data before JSON serialization

### "403 Legacy Endpoint" from FMP
**Fix:** Update FMP client to use `/stable` base URL instead of `/api/v3`

### Modal appears cut off at top
**Fix:** Changed modal positioning from `top: 50%` to `top: 5vh` in CSS

### Server not reflecting changes
**Fix:** Hard refresh browser (Cmd+Shift+R) or restart server

---

## 📖 Further Reading

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Lenny's Newsletter: How to Build AI Product Sense](https://www.lennysnewsletter.com/p/how-to-build-ai-product-sense)

---

**Last Updated:** February 2026  
**Maintained by:** Pin (with help from AI agents)
