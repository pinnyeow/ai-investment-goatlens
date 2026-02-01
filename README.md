# 🐐 GOATlens

**Multi-Agent Investment Analysis using Legendary Investor Mental Models**

GOATlens is an AI-powered research tool that analyzes companies through the eyes of the greatest investors in history — Buffett, Lynch, Graham, Munger, and Dalio. Each "GOAT" agent evaluates the same company from their unique perspective, highlighting where they agree and where they'd debate.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-gray)

## ✨ Features

### 🕰️ Time-Travel Analysis
Compare company fundamentals across configurable **anchor years** (e.g., 2014, 2019, 2024). See how revenue, margins, and competitive position have evolved over a decade.

### 🤖 Multi-Agent Debate
Five legendary investor agents analyze in parallel:

| Agent | Style | Key Focus |
|-------|-------|-----------|
| **Warren Buffett** | Value + Quality | Moats, ROE, Owner Earnings |
| **Peter Lynch** | GARP | PEG Ratio, Growth, Ten-Baggers |
| **Benjamin Graham** | Deep Value | Margin of Safety, P/E, P/B |
| **Charlie Munger** | Quality First | Mental Models, Avoid Mistakes |
| **Ray Dalio** | Macro/Risk | Debt Cycles, Diversification |

### 📊 Consensus & Divergence
Automatically surfaces where agents agree (consensus) and where they'd debate (divergence) — revealing nuances that single-perspective analysis would miss.

### 🏰 Moat Decay Detection
Algorithmic detection of competitive advantage trends across time — strengthening, stable, weakening, or collapsed.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [FMP API Key](https://site.financialmodelingprep.com/developer/docs) (free tier: 250 calls/day)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-investment-goatlens.git
cd ai-investment-goatlens

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the `backend/` directory:

```env
# Required: Financial Modeling Prep API Key
# Get your free key at: https://site.financialmodelingprep.com/developer/docs
FMP_API_KEY=your_api_key_here

# Optional: OpenAI API Key (for LLM-enhanced narratives)
OPENAI_API_KEY=your_openai_key_here

# Optional: Server configuration
PORT=8000
```

#### Getting Your FMP API Key

1. Visit [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs)
2. Sign up for a free account
3. Navigate to Dashboard → API Keys
4. Copy your API key to the `.env` file

**Free Tier Limits:**
- 250 API calls per day
- ~40 company analyses per day

### Running the Server

```bash
# From the backend directory
cd backend

# Start the server
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --port 8000
```

The server will start at `http://localhost:8000`

### Using the Frontend

Open `frontend/index.html` in your browser, or access `http://localhost:8000/` if running the backend.

---

## 📖 Usage

### Basic Analysis

Enter a stock ticker (e.g., `AAPL`, `MSFT`, `GOOGL`) and click **Analyze**. The system will:

1. Fetch 10 years of financial data
2. Perform time-travel analysis across anchor years
3. Run all 5 GOAT agents in parallel
4. Calculate consensus and surface divergences
5. Display comprehensive results

### API Usage

```bash
# Analyze a company
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "anchor_years": [2014, 2019, 2024]}'

# List available agents
curl "http://localhost:8000/api/agents"

# Demo mode (no API key required)
curl "http://localhost:8000/api/demo"
```

### Python Client Example

```python
import httpx
import asyncio

async def analyze_company(ticker: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/analyze",
            json={
                "ticker": ticker,
                "anchor_years": [2014, 2019, 2024]
            }
        )
        return response.json()

# Run analysis
result = asyncio.run(analyze_company("AAPL"))
print(f"Consensus: {result['consensus_verdict']}")
print(f"Agreement: {result['agreement_score']:.0%}")
```

---

## 🏗️ Project Structure

```
ai-investment-goatlens/
├── PRD.md                    # Full product requirements
├── README.md                 # This file
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
│   ├── strategies/          # Consensus calculation
│   │   └── __init__.py
│   ├── data_sources/        # API adapters
│   │   ├── __init__.py
│   │   └── fmp.py           # FMP API client
│   └── temporal/            # Time-travel analysis
│       ├── __init__.py
│       └── anchor_years.py
└── frontend/
    └── index.html           # Single-page UI
```

---

## 🔧 Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FMP_API_KEY` | **Yes** | - | Financial Modeling Prep API key |
| `OPENAI_API_KEY` | No | - | OpenAI key for enhanced narratives |
| `PORT` | No | `8000` | Server port |
| `HOST` | No | `0.0.0.0` | Server host |

### Analysis Parameters

```json
{
  "ticker": "AAPL",           // Stock symbol (required)
  "anchor_years": [2014, 2019, 2024],  // Years to compare (2-5 years)
  "agents": ["buffett", "lynch"]       // Optional: specific agents only
}
```

---

## 📊 Understanding Results

### Verdict Scale

| Verdict | Score Range | Meaning |
|---------|-------------|---------|
| `strong_buy` | 60 to 100 | Compelling investment opportunity |
| `buy` | 20 to 59 | Favorable, consider buying |
| `hold` | -20 to 19 | Neutral, monitor position |
| `sell` | -60 to -21 | Unfavorable, consider selling |
| `strong_sell` | -100 to -61 | Significant red flags |

### Agreement Score

- **0.8 - 1.0**: Strong consensus across agents
- **0.6 - 0.8**: General agreement with some divergence
- **0.4 - 0.6**: Mixed views, significant debate
- **0.0 - 0.4**: High divergence, agents strongly disagree

### Moat Trend

| Trend | Meaning |
|-------|---------|
| `strengthening` | Competitive advantages improving |
| `stable` | Moat maintained over time |
| `weakening` | Competitive position eroding |
| `collapsed` | Significant moat deterioration |

---

## 🧪 Development

### Running in Development Mode

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest
```

### Adding a New Agent

1. Create `backend/agents/new_agent.py`
2. Implement the `analyze()` method following existing agent patterns
3. Add to `backend/agents/__init__.py`
4. Register in `main.py` workflow

---

## 📝 API Reference

### POST `/api/analyze`

Analyze a company with all GOAT agents.

**Request:**
```json
{
  "ticker": "AAPL",
  "anchor_years": [2014, 2019, 2024],
  "agents": null
}
```

**Response:** See PRD.md for full schema.

### GET `/api/agents`

List available GOAT agents with their strategies.

### GET `/api/demo`

Returns demo analysis for testing (no API key required).

### GET `/health`

Health check endpoint.

---

## 🤝 Contributing

Contributions are welcome! Please read the PRD.md for architectural guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📜 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Warren Buffett**, **Peter Lynch**, **Benjamin Graham**, **Charlie Munger**, and **Ray Dalio** for their timeless investment wisdom
- [Financial Modeling Prep](https://financialmodelingprep.com) for the excellent financial data API
- [LangGraph](https://github.com/langchain-ai/langgraph) for the workflow orchestration framework

---

## ⚠️ Disclaimer

GOATlens is for **educational and research purposes only**. This is not financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
