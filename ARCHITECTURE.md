# GOATlens Architecture

## Core rule

Accounting fundamentals and volatile market data are different dependency classes. GOATlens keeps them separate so a market-data outage cannot erase authoritative filing analysis.

## Providers

### SEC EDGAR Company Facts — authoritative fundamentals

`backend/data_sources/sec.py` resolves SEC tickers, retrieves Company Facts, selects annual 10-K/10-K/A facts as of a declared date, and preserves provenance:

- namespace and concept
- unit and value
- period start/end
- accession, form, and filing date
- fiscal year
- derivation inputs or an availability reason

Conflicting aliases fail closed. Derived liabilities require matching period end, accession, filing date, and form. Bank cash-flow semantics remain explicitly non-comparable to industrial companies.

`SEC_USER_AGENT` must contain an application identifier and a contact email. Missing configuration returns a sanitized non-retryable `503`. SEC rate-limit, transport, and capacity failures return sanitized retryable `503` responses. A symbol absent from the SEC ticker map returns a sanitized `404`.

### Yahoo Finance — best-effort market enrichment

`backend/data_sources/yahoo_chart.py` is the only Yahoo boundary. It supplies a timestamped current regular-market price, adjusted daily price history, split events, and price-series risk inputs. It does not supply financial statements, normalized ratios, earnings, or analyst context. Its data is optional for the SEC fundamentals path. Public market status separates observation time from retrieval time and includes quote currency, cache age, and freshness.

When Yahoo is unavailable:

- the SEC-backed response still succeeds;
- `data_status.market_data.available` is `false`;
- price-dependent agent results are `incomplete`;
- unavailable metrics are named;
- no missing input is silently converted to a favorable score.

Yahoo is a prototype market-data source, not an assertion of permanent public redistribution rights.

## Request flow

```text
POST /api/analyze
  -> validate ticker
  -> retrieve and normalize SEC fundamentals
  -> try Yahoo market enrichment
  -> run only agents whose required inputs are available
  -> synthesize explicit complete/incomplete results
  -> return data_status and provenance-aware analysis
```

SEC errors stop before Yahoo because market data cannot substitute for authoritative fundamentals. Yahoo errors degrade only market-dependent behavior.

## Missing-data contract

- Missing values remain `None`/unavailable.
- Every unsupported derived metric has an availability reason.
- Missing inputs earn no scoring points.
- Agents that require absent inputs return `status="incomplete"` with `unavailable_metrics`.
- Current prices include observation/fetch timestamps; cached values are never described as timelessly current.

## Workflow and files

```text
backend/main.py                  API and LangGraph orchestration
backend/data_sources/sec.py      filing fundamentals and provenance
backend/data_sources/yahoo_chart.py  narrow price/history market boundary
backend/agents/                  investor-specific scoring
backend/temporal/                current-period moat assessment
backend/tests/fixtures/sec/      reduced Company Facts fixtures
frontend/index.html              browser UI
```

## Release boundary

A release is not successful merely because code merged or Render is healthy. Verification must establish:

1. local focused and full tests, lint, and security checks;
2. a real local API and browser journey;
3. independent fail-closed review;
4. passing GitHub Actions;
5. `/health` reports a `release` equal to the merged commit;
6. one production SEC fundamentals canary;
7. production market-data status;
8. one production AAPL analysis;
9. production browser behavior.

See `docs/verification.md` for the executable procedure.
