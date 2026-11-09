# GOATlens

A free educational site that shows how five legendary investors (Buffett, Lynch, Graham, Munger, Dalio) would look at a stock. Each investor is a short, published checklist; every check shows the number it used and why it passed or failed.

**Live:** https://goatlens.onrender.com · Not financial advice.

## How it works

```
Nightly GitHub Action (free)                      Render (free)
  Yahoo Finance → lens/data.py   (~600 companies)    main.py serves the page and
  checklists    → lens/investors.py                  reads the JSON from the data
  JSON files    → lens/build.py  → `data` branch  →  branch (cached 30 min)
```

- **No live data calls while a user waits.** Everything is prepared nightly, so pages load instantly and Yahoo outages or throttling never break the site; if a company fails one night, yesterday's data stays up and is labelled.
- **Missing data is skipped, never counted as zero.** An investor needs at least half of its checklist to give a verdict; the page says exactly which checks were skipped and why (for example, "not meaningful for banks").
- **Covered companies** are listed in `backend/lens/universe.txt` (S&P 500 plus popular names). Add a ticker there and it appears after the next nightly run.
- **Cost:** $0. GitHub Actions (public repo), Yahoo Finance via `yfinance`, Render free tier.

See [ARCHITECTURE.md](ARCHITECTURE.md) for details and the reasoning behind this design.

## Run locally

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
cd .. && make dev        # builds 5 companies, then serves http://localhost:8000
make check               # lint + tests
```

Build any set of companies: `cd backend && python -m lens.build --out ../.snapshots --tickers NVDA,KO`.

## API

| Endpoint | What it returns |
|---|---|
| `POST /api/analyze` `{"ticker": "AAPL", "agents": ["buffett", ...]}` | Verdicts, checklists, consensus, metrics |
| `GET /api/price-history/{ticker}?period=1y` | Prices plus earnings markers |
| `GET /api/earnings/{ticker}` | Last quarters' EPS vs estimates |
| `GET /api/tickers` | Covered companies (for search) |
| `GET /api/coverage` | Last night's coverage report |
| `GET /health` | Status and deployed commit |

## Data and terms

Data comes from Yahoo Finance through the unofficial `yfinance` library. It is used here for a free, non-commercial educational project; revisit the source if GOATlens ever charges money.
