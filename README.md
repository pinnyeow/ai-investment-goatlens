# GOATlens

GOATlens is an educational investment-research application that evaluates U.S. public companies through five investor lenses: Buffett, Lynch, Graham, Munger, and Dalio.

## Data architecture

- **SEC EDGAR Company Facts** is authoritative for annual accounting fundamentals and provenance.
- **Yahoo Finance** is best-effort enrichment: the chart endpoint supplies prices and adjusted history, while a narrow `yfinance` adapter supplies allowlisted provider-normalized current ratios.
- SEC-backed fundamental analysis remains available when Yahoo is unavailable.
- Price-dependent agents return `incomplete` when required market inputs are missing.
- Missing financial values remain unavailable; they are never replaced with fabricated zeroes.

The current SEC scope is U.S. issuers present in the SEC ticker map. A genuine unknown issuer returns a sanitized `404`; SEC transport or capacity failures return retryable `503` responses.

## Quick start

### Requirements

- Python 3.10+
- An identifying SEC operator string containing an application name and contact email

```bash
git clone https://github.com/pinnyeow/ai-investment-goatlens.git
cd ai-investment-goatlens/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:

```env
# Required by SEC fair-access policy; use your real operational contact.
SEC_USER_AGENT=GOATlens your-contact@example.com

# Optional: enables LLM-enhanced narratives.
OPENAI_API_KEY=your_openai_key_here

PORT=8000
```

Run from `backend/`:

```bash
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000/>.

## API

Analyze a company:

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","time_period":"1y"}'
```

The response includes `data_status` for SEC fundamentals, Yahoo market data, and Yahoo normalized ratios. Each agent result includes `status` and `unavailable_metrics` so partial analysis is explicit.

Other endpoints:

- `GET /api/agents`
- `GET /api/price-history/{ticker}`
- `GET /api/earnings/{ticker}`
- `GET /health` (includes the exact deployed release SHA)

## Project structure

```text
backend/
  main.py                 FastAPI and LangGraph workflow
  agents/                 Investor scoring agents
  data_sources/sec.py     SEC Company Facts adapter
  data_sources/yahoo_chart.py   Best-effort price/history enrichment
  data_sources/yahoo_ratios.py  Narrow normalized-ratio enrichment
  temporal/               Current-period moat assessment
  tests/                  Pytest suite and SEC fixtures
frontend/index.html       Browser application
```

## Development

```bash
cd backend
venv/bin/python -m pytest -q
venv/bin/ruff check .
```

See `ARCHITECTURE.md` for contracts and `docs/verification.md` for the release procedure.

## Disclaimer

GOATlens is for educational and research purposes only. It is not financial advice. Verify source filings and consult a qualified professional before making investment decisions.
