# GOATlens

Free educational site: five investor checklists (Buffett, Lynch, Graham, Munger, Dalio) applied to ~600 companies. Data is built nightly from Yahoo Finance by GitHub Actions and served by a tiny FastAPI app on Render. Read ARCHITECTURE.md first.

## Structure

- `backend/lens/data.py`: Yahoo Finance → metrics (number or None)
- `backend/lens/investors.py`: the five checklists, consensus
- `backend/lens/build.py`: nightly build CLI; `backend/lens/universe.txt` lists covered tickers
- `backend/main.py`: web server (reads snapshots; no live data calls)
- `frontend/index.html`: single page
- `backend/tests/`: pytest

## Commands

```bash
make check   # lint + tests
make dev     # build 5 companies locally and serve http://localhost:8000
```

## Rules

- Never commit directly to main; use a branch + PR. Run `make check` first.
- Missing data is skipped and explained, never treated as zero, and never blocks the whole answer.
- Do not add checks that make a user request fail or crash because data is imperfect. Validate at build time; log and label instead.
- Every PR that touches data or scoring reports coverage from the CI `live-build-sample` job. Do not lower coverage (all-five rate ≥ 85%).
- Keep it free: no paid APIs, no per-request LLM calls.
