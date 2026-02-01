# GOATlens - Multi-Agent Investment Analysis Framework

## Product Requirements Document (PRD)

**Version:** 1.0  
**Last Updated:** January 2026  
**Status:** In Development

---

## 1. Executive Summary

GOATlens is an AI-powered investment research tool that performs comprehensive company analysis using the mental models of legendary investors ("GOATs" - Greatest Of All Time). The platform combines **10-year Time-Travel Analysis** with a **Multi-Agent Debate System** to provide nuanced, multi-perspective investment insights.

### Key Differentiators

1. **Multi-Perspective Analysis** - Five distinct investor agents analyze the same data simultaneously
2. **Time-Travel Analysis** - Compare company fundamentals across configurable anchor years
3. **Moat Decay Detection** - Automatically identify strengthening or weakening competitive advantages
4. **Consensus vs. Divergence** - Highlight where experts agree and where they'd debate
5. **Automated Data Pipeline** - No manual data entry; fully automated via FMP API

---

## 2. Vision & Goals

### Vision Statement

> A research tool that empowers investors to see companies through the eyes of the greatest investors in history, revealing insights that single-perspective analysis would miss.

### Primary Goals

| Goal | Metric | Target |
|------|--------|--------|
| Multi-perspective insights | Agents providing divergent views | ≥2 unique concerns/insights per analysis |
| Time-travel clarity | Anchor year comparisons | 10-year trend visibility |
| Fast analysis | End-to-end response time | <10 seconds |
| Accuracy | Correct data fetching | 99.9% API success rate |

### Non-Goals (v1.0)

- Portfolio management features
- Trade execution
- Real-time price alerts
- Social/community features
- Backtesting engine

---

## 3. Core Features

### 3.1 Time-Travel / Temporal Analysis

The Time-Travel feature enables historical comparison of company fundamentals across selected "anchor years."

#### Capabilities

| Feature | Description |
|---------|-------------|
| **Anchor Year Selection** | User selects 2-5 years to compare (e.g., 2014, 2019, 2024) |
| **Metric Comparison** | Side-by-side view of key metrics across years |
| **CAGR Calculation** | Compound annual growth rates for revenue, EPS, FCF |
| **Moat Decay Detection** | Algorithmic detection of competitive advantage trends |
| **Story Evolution** | Track how company narrative changed over time |

#### Moat Decay Detection Algorithm

```
Moat Score = (Gross Margin Score × 0.30) + 
             (Operating Margin Score × 0.30) +
             (ROE Score × 0.25) +
             (Debt Health Score × 0.15)

Trend Classification:
- STRENGTHENING: Score improved >15 points, positive margin trends
- STABLE: Score change within ±15 points
- WEAKENING: Score declined >15 points, negative margin trends  
- COLLAPSED: Score declined >30 points
```

#### Anchor Year Snapshot Data

For each anchor year, the system captures:

- Revenue, Net Income, EPS
- ROE, ROA, ROIC
- Gross Margin, Operating Margin, Profit Margin
- Debt-to-Equity, Current Ratio
- Free Cash Flow
- Key events (when available via LLM)

### 3.2 Multi-Agent Debate System

Five legendary investor agents analyze the same company data in parallel, each applying their unique mental models and investment criteria.

#### The GOATs

| Agent | Philosophy | Key Criteria | Moat Focus |
|-------|------------|--------------|------------|
| **Warren Buffett** | Buy wonderful companies at fair prices | ROE >15%, profit margins, low debt, owner earnings | Durable competitive advantage |
| **Peter Lynch** | Invest in what you know; GARP | PEG <1, earnings growth, low institutional ownership | Growth at reasonable price |
| **Benjamin Graham** | Margin of safety | P/E <15, P/B <1.5, current ratio >2, dividend history | Deep value, margin of safety |
| **Charlie Munger** | Quality over cheapness | Business quality, management, pricing power | Mental models, avoid mistakes |
| **Ray Dalio** | Principles-based, macro aware | Debt cycles, risk parity, diversification | Risk-adjusted returns |

#### Agent Scoring System

Each agent produces:

1. **Score** (-100 to +100): Quantitative assessment
2. **Verdict** (5 levels): strong_buy, buy, hold, sell, strong_sell
3. **Key Insights** (list): Positive observations
4. **Concerns** (list): Red flags and risks
5. **Agent-specific extras**: Buffett quotes, Lynch tips, Graham margin of safety, etc.

#### Consensus Calculation

```python
# Consensus Algorithm
consensus_verdict = weighted_average(all_agent_verdicts)
agreement_score = 1 - (variance(scores) / max_possible_variance)

# Identify consensus points: insights mentioned by majority
consensus_points = insights where count >= (num_agents // 2 + 1)

# Identify divergence: where agents significantly disagree
divergence_points = cases where verdict_spread > 2 levels
```

### 3.3 Automated Data Pipeline

All financial data is fetched automatically via the Financial Modeling Prep (FMP) API.

#### Data Sources

| Source | Type | Usage |
|--------|------|-------|
| **FMP API** (Primary) | Financials, ratios, prices, earnings | All analysis |
| Alpha Vantage (Future) | Backup data source | Failover |
| Yahoo Finance (Future) | Price validation | Cross-reference |

#### FMP API Endpoints Used

| Endpoint | Data Retrieved |
|----------|----------------|
| `/profile/{ticker}` | Company info, sector, market cap, beta |
| `/income-statement/{ticker}` | Revenue, net income, EPS (10 years) |
| `/balance-sheet-statement/{ticker}` | Assets, liabilities, equity (10 years) |
| `/cash-flow-statement/{ticker}` | FCF, capex, dividends (10 years) |
| `/key-metrics/{ticker}` | Per-share data, growth rates |
| `/ratios/{ticker}` | Profitability, liquidity, leverage ratios |
| `/historical-price-full/{ticker}` | Historical prices (for volatility) |

#### Rate Limits

- Free Tier: 250 API calls/day
- Single analysis: ~6 API calls
- Max analyses per day: ~40

---

## 4. Architecture

### 4.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (HTML/JS)                       │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ Ticker   │  │ Anchor Year  │  │    Results Display      │   │
│  │ Input    │  │ Selector     │  │ (Agents, Consensus, ...)│   │
│  └──────────┘  └──────────────┘  └─────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   LangGraph Workflow                      │   │
│  │                                                           │   │
│  │  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐  │   │
│  │  │ Fetch   │──▶│ Temporal │──▶│  Run    │──▶│Synthesize│  │   │
│  │  │ Data    │   │ Analysis │   │ Agents  │   │ Results  │  │   │
│  │  └─────────┘   └──────────┘   └─────────┘   └─────────┘  │   │
│  │       │                            │                       │   │
│  └───────┼────────────────────────────┼───────────────────────┘   │
│          │                            │                           │
│          ▼                            ▼                           │
│  ┌──────────────┐      ┌─────────────────────────────────┐       │
│  │   FMP API    │      │         GOAT Agents             │       │
│  │   Client     │      │  ┌───────┐ ┌───────┐ ┌───────┐ │       │
│  └──────────────┘      │  │Buffett│ │ Lynch │ │Graham │ │       │
│                        │  └───────┘ └───────┘ └───────┘ │       │
│                        │  ┌───────┐ ┌───────┐           │       │
│                        │  │Munger │ │ Dalio │           │       │
│                        │  └───────┘ └───────┘           │       │
│                        └─────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   FMP API       │
                  │ (External)      │
                  └─────────────────┘
```

### 4.2 LangGraph Workflow

The analysis pipeline is implemented as a LangGraph state machine:

```
START
  │
  ▼
┌─────────────┐
│ fetch_data  │  ← Calls FMP API (parallel requests)
└─────────────┘
  │
  ▼
┌─────────────────┐
│temporal_analysis│  ← Compares anchor years, detects moat trends
└─────────────────┘
  │
  ▼
┌─────────────┐
│ run_agents  │  ← Runs 5 GOAT agents in parallel
└─────────────┘
  │
  ▼
┌─────────────┐
│ synthesize  │  ← Calculates consensus, builds report
└─────────────┘
  │
  ▼
 END
```

### 4.3 Project Structure

```
ai-investment-goatlens/
├── PRD.md                    # This document
├── README.md                 # Quick start guide
├── backend/
│   ├── main.py              # FastAPI + LangGraph workflow
│   ├── requirements.txt     # Python dependencies
│   ├── agents/              # GOAT investor agents
│   │   ├── __init__.py
│   │   ├── buffett.py       # Warren Buffett agent
│   │   ├── lynch.py         # Peter Lynch agent
│   │   ├── graham.py        # Benjamin Graham agent
│   │   ├── munger.py        # Charlie Munger agent
│   │   └── dalio.py         # Ray Dalio agent
│   ├── strategies/          # Strategy evaluation logic
│   │   └── __init__.py      # Consensus calculation
│   ├── data_sources/        # Data adapters
│   │   ├── __init__.py
│   │   └── fmp.py           # FMP API client
│   └── temporal/            # Time-travel analysis
│       ├── __init__.py
│       └── anchor_years.py  # Temporal comparison logic
└── frontend/
    └── index.html           # Single-page UI
```

---

## 5. API Specification

### 5.1 Endpoints

#### `POST /api/analyze`

Run full GOAT analysis on a company.

**Request:**
```json
{
  "ticker": "AAPL",
  "anchor_years": [2014, 2019, 2024],
  "agents": null  // optional: ["buffett", "lynch"] to filter
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "sector": "Technology",
  "consensus_verdict": "buy",
  "agreement_score": 0.72,
  "consensus_points": ["Strong brand moat", "..."],
  "divergence_points": ["Graham finds valuation too high", "..."],
  "agent_results": [
    {
      "agent": "Warren Buffett",
      "style": "Value Investing with Quality Focus",
      "verdict": "buy",
      "score": 65,
      "insights": ["Strong ROE...", "..."],
      "concerns": ["Concentration in iPhone..."]
    }
    // ... more agents
  ],
  "moat_trend": "stable",
  "anchor_year_comparison": {
    "years": [2014, 2019, 2024],
    "metrics": { "revenue": {...}, "net_income": {...} },
    "revenue_cagr": 0.077
  },
  "comparison_table": {
    "agents": [
      {"name": "Warren Buffett", "verdict": "buy", "score": 65}
      // ...
    ]
  }
}
```

#### `GET /api/agents`

List available GOAT agents.

#### `GET /api/demo`

Get demo analysis (for testing without API key).

#### `GET /health`

Health check endpoint.

---

## 6. User Experience

### 6.1 User Flow

1. **Enter Ticker** - User types stock symbol (e.g., "AAPL")
2. **Select Anchor Years** - Choose comparison years (default: 2014, 2019, 2024)
3. **Run Analysis** - Click "Analyze" button
4. **View Results** - See consensus, individual agent views, comparison table
5. **Explore Details** - Expand agent cards for insights/concerns

### 6.2 UI Components

| Component | Purpose |
|-----------|---------|
| Ticker Input | Stock symbol entry |
| Anchor Year Selector | Multi-select for comparison years |
| Consensus Panel | Overall verdict, agreement score |
| Agent Cards | Individual GOAT analysis |
| Comparison Table | Side-by-side agent comparison |
| Moat Trend Chart | Visual moat decay indicator |
| Metrics Table | Anchor year metrics comparison |

---

## 7. Technical Requirements

### 7.1 Dependencies

**Backend:**
- Python 3.10+
- FastAPI 0.109+
- LangGraph 0.0.40+
- httpx 0.26+
- pydantic 2.5+

**Frontend:**
- Vanilla HTML/CSS/JavaScript
- No build process required

### 7.2 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FMP_API_KEY` | Yes | Financial Modeling Prep API key |
| `OPENAI_API_KEY` | No | For LLM-enhanced narrative analysis |
| `PORT` | No | Server port (default: 8000) |

### 7.3 Performance Requirements

| Metric | Target |
|--------|--------|
| API Response Time | <10s for full analysis |
| FMP API Calls | <10 per analysis |
| Memory Usage | <512MB |
| Concurrent Users | 10+ |

---

## 8. Future Enhancements (Post-v1.0)

### Phase 2: Enhanced Analysis
- LLM-powered narrative generation
- News sentiment integration
- Peer comparison analysis

### Phase 3: Portfolio Features
- Save/compare analyses
- Watchlist functionality
- Historical analysis tracking

### Phase 4: Advanced Features
- Custom agent creation
- Backtesting against historical data
- API for programmatic access

---

## 9. Success Metrics

| Metric | Measurement | Target |
|--------|-------------|--------|
| User Engagement | Analyses per session | >2 |
| Analysis Quality | Divergent insights surfaced | >80% of analyses |
| Performance | P95 response time | <15s |
| Reliability | Uptime | 99% |
| Data Accuracy | Correct financial data | 99.9% |

---

## 10. Appendix

### A. Agent Scoring Details

#### Buffett Agent Scoring
- ROE contribution: up to 25 points
- Profit margin contribution: up to 20 points
- Debt level contribution: up to 20 points
- Moat assessment: up to 35 points

#### Lynch Agent Scoring  
- PEG ratio contribution: up to 35 points
- Growth rate contribution: up to 25 points
- Institutional ownership (contrarian): up to 15 points
- Ten-bagger potential bonus: up to 25 points

#### Graham Agent Scoring
- Margin of safety: up to 40 points
- P/E ratio contribution: up to 20 points
- Current ratio contribution: up to 15 points
- Dividend history: up to 15 points
- Net-Net bonus: 20 points

#### Munger Agent Scoring
- Business quality score: up to 100 points
- Red flag penalties: -20 points each
- Management ownership bonus: up to 15 points

#### Dalio Agent Scoring
- Risk-adjusted returns: up to 30 points
- Diversification score: up to 20 points
- Debt cycle positioning: up to 25 points
- Beta adjustment: up to 15 points

### B. Glossary

| Term | Definition |
|------|------------|
| **GOAT** | Greatest Of All Time - legendary investors |
| **Anchor Year** | Specific year selected for historical comparison |
| **Moat** | Competitive advantage that protects a company |
| **PEG Ratio** | P/E ratio divided by earnings growth rate |
| **ROIC** | Return on Invested Capital |
| **Ten-Bagger** | Stock that grows 10x in value |
| **Margin of Safety** | Discount to intrinsic value |
| **GARP** | Growth At a Reasonable Price |
