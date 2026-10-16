# GOATlens verification

This is the smallest repeatable verification procedure for the SEC/Yahoo split. It is repository-owned because the release boundary spans tests, a live API, a browser, GitHub, and Render.

## 1. Deterministic pre-PR checks

From the repository root:

```bash
python scripts/verify_goatlens.py
```

This runs the full backend test suite, Ruff, `git diff --check`, and parses both GitHub Actions workflows.

## 2. Local live API

Use a real operational contact only in the shell environment; never commit it:

```bash
cd backend
SEC_USER_AGENT='GOATlens local verification contact@example.com' \
  venv/bin/uvicorn main:app --host 127.0.0.1 --port 8010
```

In another shell:

```bash
curl --fail http://127.0.0.1:8010/health
curl --fail http://127.0.0.1:8010/api/fundamentals-health
curl --fail -X POST http://127.0.0.1:8010/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"AAPL","agents":["buffett"]}'
```

Verify the analysis reports `fundamentals.available=true`, `fundamentals.source=sec`, a complete individual Buffett result, and consensus unavailable/incomplete because one completed opinion does not meet the two-analysis minimum. Yahoo failure must not turn AAPL into `ticker_not_found`.

## 3. Local browser journey

Open <http://127.0.0.1:8010/>, analyze AAPL, and verify:

- the analysis renders without a page or console error;
- the SEC-backed Buffett analysis remains available if Yahoo is unavailable;
- the market-data notice appears when Yahoo is unavailable;
- no price chart is fetched or left stale in that state;
- incomplete agents name their unavailable metrics;
- an all-incomplete consensus is labeled `Incomplete`, not `Hold`.

Preserve a screenshot and browser/server evidence before stopping Uvicorn.

## 4. Independent review and PR

Before committing, run a fail-closed independent review of the complete diff. Security concerns or logic errors block the PR. Push one branch and open one PR only after the review passes.

GitHub Actions must pass on the exact PR head SHA.

## 5. Merge and production gate

Merging and production configuration require explicit approval. Configure `SEC_USER_AGENT` in Render with a real application identifier and operational contact; do not commit personal contact data.

After merge:

1. Poll `/health` until its `release` equals the merge SHA.
2. Call `/api/fundamentals-health` exactly once and require SEC health.
3. Call `/api/market-data-health` exactly once and record Yahoo status without treating Yahoo failure as SEC failure.
4. Run the documented Buffett-only AAPL `/api/analyze` request and require the expected AAPL revenue literal and accession provenance, a complete Buffett result, and consensus unavailable/incomplete because fewer than two analyses completed.
5. Verify the production browser behavior, including degraded market data.
6. Inspect Render logs before changing code if any production boundary fails.

A healthy server, merged PR, or successful deployment is not sufficient by itself.
