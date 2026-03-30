# GOATlens

Multi-agent investment analysis tool. Users enter a stock ticker and get scoring from 5 legendary investors (Buffett, Lynch, Graham, Munger, Dalio).

## Project Structure

- `frontend/index.html` — single-page UI
- `backend/main.py` — FastAPI server, all API endpoints
- `backend/agents/` — one file per investor agent
- `backend/data_sources/` — Yahoo Finance + FMP API clients
- `backend/tests/` — pytest test suite

## Commands

```bash
make test       # run all tests
make lint       # run linter (ruff)
make check      # run both lint + tests
```

## Running locally

```bash
cd backend && source venv/bin/activate && uvicorn main:app --reload
```

Then open http://localhost:8000

## Key rules

- Never commit directly to main — always use a branch + PR
- Run `make check` before opening a PR
- Tests live in `backend/tests/` — use pytest + pytest-asyncio
- Linter is ruff — auto-fix with `ruff check . --fix`
