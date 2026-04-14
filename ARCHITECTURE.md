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

## 🧠 AI Decision Framework: Task → Paradigm → Framework → Model

Every AI product decision should follow this sequence. Starting from the model and working backwards leads to over-engineered, expensive solutions.

### Step 1: Task — what job are you asking AI to do?

Not every task needs an LLM. The rule of thumb: if you can write an `if` statement for it, don't use an LLM. If the output needs the word "because," you probably should.

| Task | LLM needed? | Rationale |
|------|-------------|-----------|
| Fetch & normalize financial data | **No** | API calls, no judgment |
| Moat trend detection | **No** | Formula with clear thresholds |
| Score a company per investor | **No** | Rule-based point system |
| Interpret what scores mean | **Yes** | Judgment, narrative, reasoning |
| Synthesize consensus across agents | **Yes** | Comparing perspectives, finding patterns |
| Generate readable insights | **Yes** | Writing, nuance, persuasion |

### Step 2: Paradigm — what AI pattern fits?

GOATlens uses **multi-agent (fan-out/fan-in) + chain**, layered together:

```
fetch_data ──► temporal_analysis ──► run_agents ──► synthesize
                                        │
    ◄──────── CHAIN ────────────►       │
    (sequential, output feeds next)     │
                                        │
                                   ┌────┴────┐
                                   │ MULTI-  │
                              ┌────┤ AGENT   ├────┐
                              │    └────┬────┘    │
                              ▼         ▼         ▼
                          Buffett    Graham     Lynch ...
                              │         │         │
                              └────┬────┘─────────┘
                                   ▼
                              synthesize
```

Why multi-agent: agents are independent (no cross-dependency), apply diverse philosophies to the same data, and a synthesis step needs to see all outputs. A single LLM call doing "pretend to be 5 investors" produces blurred perspectives.

Why chain: the pipeline is sequential — you can't run agents before fetching data, and you can't synthesize before agents complete.

**Next paradigm to add: RAG** — retrieval of SEC filings to ground agent insights in real management commentary (see RAG Design below).

### Step 3: Framework — how do you orchestrate it?

**LangGraph** (built on LangChain) — a state machine with explicit nodes, edges, and shared state (`GOATState`).

| Layer | Role | Car analogy |
|-------|------|-------------|
| **Paradigm** (multi-agent + chain) | The design intent | "Sedan that can go off-road" |
| **Framework** (LangGraph) | Connects everything | Chassis and drivetrain |
| **Library** (LangChain) | Individual components | Engine, brakes, steering |

### Step 4: Model — which LLM(s)?

Match model power to task complexity. Better context + cheap model often beats poor context + expensive model.

| Task | Recommended model | Rationale |
|------|-------------------|-----------|
| Agent interpretation (C2) | `gpt-4o-mini` | Single-perspective reasoning, philosophy is well-defined |
| Consensus synthesis (D) | `gpt-4o` or `claude-sonnet` | Must hold 5 viewpoints simultaneously |
| Narrative generation (E) | Same as D | Writing quality directly impacts UX |

**Current state:** All agents use `gpt-4o-mini`. Model routing is a future optimization — context engineering has higher ROI first.

---

## 🎯 Context Engineering

Context engineering is the most important lever for improving GOATlens output quality. It's the art of deciding what goes into the LLM's context window (its working memory) for each call.

### The problem: context is a zero-sum game

Every LLM has a hard limit on tokens it can process at once:

| Model | Context window | Roughly |
|-------|---------------|---------|
| gpt-4o-mini | 128K tokens | ~200 pages |
| claude-sonnet | 200K tokens | ~300 pages |
| gemini-2.5-pro | 1M tokens | ~1,500 pages |

Everything competes for that space: system prompt, financial data, retrieved documents, conversation history, and the model's own response. More context ≠ better — **context rot** degrades performance as the window fills, especially for precision tasks like financial analysis.

### Current state: same context to all agents (naive)

```
Every agent gets:
  - Full 10-year financials         (~3,000 tokens)
  - All earnings data               (~800 tokens)
  - Moat analysis                   (~400 tokens)
  - Agent philosophy prompt         (~500 tokens)
  Total per agent:                  ~4,700 tokens
  × 5 agents = ~23,500 tokens
```

This works today because context is lean. But adding RAG (filings, transcripts) without selective filtering would bloat each agent to ~44,000 tokens — mostly wasted on data the agent doesn't use.

### Target state: selective context per agent

Each agent gets a **context profile** — only the metrics and retrieved documents relevant to their investment philosophy:

| Agent | Metrics they need | RAG retrieval query | What to exclude |
|-------|-------------------|---------------------|-----------------|
| **Buffett** | ROE, margins, debt, FCF, owner earnings | "competitive advantage, pricing power, moat durability" | Macro commentary, beta |
| **Lynch** | PEG, revenue growth, EPS growth | "growth rate, new products, market expansion" | Debt ratios, dividend history |
| **Graham** | P/E, P/B, current ratio, dividend yield | "book value, debt repayment, dividend policy" | Growth narrative, market share |
| **Munger** | ROIC, management tenure, capital allocation | "management quality, capital allocation, pricing decisions" | Technical ratios, beta |
| **Dalio** | Debt/equity, interest coverage, beta | "macro environment, debt cycle, interest rate sensitivity" | PEG, ten-bagger potential |

**Impact:** ~3,400 tokens per agent (down from ~44,000 with naive RAG). 13x reduction. Less context rot, faster responses, lower cost, better results.

### The synthesis agent is the exception

The synthesis node *should* get broad context — all 5 agent outputs — because its job is finding consensus and divergence across perspectives. Broad context is correct here.

### Caching strategy by data type

| Data type | How often it changes | Cache TTL |
|-----------|---------------------|-----------|
| Stock price | Every second | 1 minute or don't cache |
| Financial statements | Every quarter | 24 hours |
| SEC filings (10-K, 10-Q) | **Never** (historical records) | **Permanent** (refresh quarterly) |

SEC filings are immutable. Once AAPL's 2024 10-K is in ChromaDB, it stays forever. The second user analyzing AAPL costs zero retrieval API calls.

---

## 📄 RAG Design: SEC EDGAR Filings

### Why SEC EDGAR (not earnings call transcripts)

| | SEC Filings (EDGAR) | Earnings Calls (FMP) |
|---|--------------------|-----------------------|
| **Cost** | Free (government, public domain) | $149/month (Ultimate plan) |
| **Reliability** | Government infrastructure | API may change |
| **Legal** | Public domain, meant to be accessed | Varies by source |
| **Content** | Strategy, risks, MD&A, financials explained | CEO sentiment, analyst Q&A |
| **Best for** | Graham, Dalio (risk-focused) | Buffett, Munger (management quality) |

Earnings call transcripts can be added later as a premium data source. SEC filings are the right v1 foundation.

### What to retrieve from filings

| Filing | Section | Value for agents |
|--------|---------|-----------------|
| **10-K** (annual) | MD&A (Management Discussion & Analysis) | Strategy, moat narrative, long-term view |
| **10-K** | Risk Factors | Red flags, competitive threats |
| **10-Q** (quarterly) | MD&A updates | Recent performance, emerging concerns |
| **8-K** (event-driven) | Material events | Leadership changes, guidance updates |

### Free RAG tech stack

```
SEC EDGAR (free, public)         ← filing source
       │
       ▼
edgartools (Python library)      ← fetch + parse filings
       │
       ▼
sentence-transformers            ← local embeddings (all-MiniLM-L6-v2, free)
       │
       ▼
ChromaDB                         ← local vector store (free, persists to disk)
       │
       ▼
Agent prompt                     ← top 3 relevant chunks injected per agent
```

Total additional cost: $0. Everything runs locally.

### Updated LangGraph workflow with RAG

```
START
  │
  ▼
┌─────────────┐
│ fetch_data  │  ← Yahoo/FMP API (parallel)
└─────────────┘
  │
  ▼
┌──────────────────┐
│ retrieve_filings │  ← NEW: SEC EDGAR → chunk → embed → ChromaDB
└──────────────────┘     (skipped if already cached)
  │
  ▼
┌─────────────────┐
│temporal_analysis│  ← Moat trend detection (deterministic)
└─────────────────┘
  │
  ▼
┌─────────────┐
│ run_agents  │  ← Each agent retrieves top 3 chunks using
└─────────────┘     agent-specific query (selective context)
  │
  ▼
┌─────────────┐
│ synthesize  │  ← Consensus across all agent outputs
└─────────────┘
  │
  ▼
 END
```

### Updated GOATState with RAG fields

```python
{
    # ... existing fields ...

    # After retrieve_filings_node (NEW)
    "filings_indexed": True,          # Whether filings are in ChromaDB
    "filings_metadata": {
        "latest_10k": "2024",
        "latest_10q": "Q3 2024",
        "total_chunks": 142
    },

    # Inside run_agents_node (CHANGED)
    # Each agent now includes retrieved_context in its output
    "agent_results": [
        {
            "agent": "buffett",
            "score": 65,
            "verdict": "buy",
            "retrieved_context": [       # NEW: what the agent grounded on
                "10-K MD&A: installed base of 2.2B active devices...",
                "10-K MD&A: services revenue grew 14% to $24.2B...",
                "10-Q Risk: increasing regulatory scrutiny in EU..."
            ],
            "insights": [...],
            "concerns": [...]
        }
    ]
}
```

### API budget with RAG

| Scenario | API calls | Within free tier? |
|----------|----------|-------------------|
| Single stock (first time) | 6 (financials) + 3 (filings) = 9 | Yes (250/day) |
| Single stock (cached) | 6 (financials) + 0 (cached) = 6 | Yes |
| 5-stock comparison (first time) | 45 | Yes |
| 5-stock comparison (all cached) | 30 | Yes |

---

## 🔮 Future Enhancements

### Phase 2: Memory
- Persist last N analyses for comparison ("AAPL was a buy last quarter, what changed?")
- User preferences (favorite agents, default anchor years)
- Investment style profile to weight agent opinions

### Phase 3: Earnings Call Transcripts
- Add FMP Ultimate ($149/month) or alternative source when product value justifies cost
- Richer grounding: CEO tone, analyst Q&A, forward guidance
- Strongest benefit for Buffett and Munger agents (management quality focus)

### Phase 4: Advanced Context Engineering
- Context pruning: summarize old filings instead of passing raw text
- Progressive disclosure: pull in context only when the agent requests it
- Subagent pattern: spawn focused sub-queries for deep-dive questions

---

## 📚 Key Concepts

### Context Engineering
Deciding what goes into the LLM's context window for each call. The most important lever for output quality. Better context + cheap model > poor context + expensive model.

### Tool Calling
GOATlens' "tools" are: Yahoo Finance scraping, FMP API, SEC EDGAR, LLM calls. They're orchestrated by LangGraph (the "agent harness"), not called directly by agents.

### RAG (Retrieval Augmented Generation)
"Before I start talking, I gotta go look everything up first." Retrieve relevant filing chunks, inject them into the agent prompt, then generate insights grounded in real management commentary.

### Model Routing
Matching model capability to task complexity. Rule-based scoring → no model. Agent interpretation → cheap model. Consensus synthesis → capable model.

### Graceful Degradation
System works without FMP API key, without OpenAI API key, without SEC filings. Each layer enriches but never blocks the core experience.

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

**Last Updated:** March 2026  
**Maintained by:** Pin (with help from AI agents)
