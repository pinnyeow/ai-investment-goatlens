# GOATlens Product Requirements

## Purpose

GOATlens is an educational research tool that evaluates U.S. public companies through five investor mental models and makes incomplete evidence visible rather than inventing certainty.

## User outcome

A user enters a ticker and receives:

- SEC-backed annual fundamentals and provenance;
- public annual SEC history with per-fact provenance;
- five investor-lens results;
- consensus and divergence;
- best-effort current price and adjusted price history;
- explicit incomplete states whenever required inputs are unavailable.

## Data requirements

### Authoritative fundamentals

SEC EDGAR Company Facts is the source for annual accounting data. Selection must be deterministic, as-of bounded, limited to annual 10-K/10-K/A facts, and preserve filing provenance. Conflicting concepts must fail closed. Arithmetic derivations must use facts from the same filing context and retain all inputs.

The initial product scope is U.S. issuers present in the SEC ticker map.

### Market enrichment

Yahoo Finance may supply current price and adjusted price history through the narrow chart boundary only. This dependency is best effort and cannot gate SEC-backed fundamental analysis. Market observations include source and timestamp.

### Missing data

- Never substitute zero, one, or a favorable default for an unavailable financial input.
- Price-dependent analysis is `incomplete` without current market data.
- Unsupported sector calculations remain unavailable with a reason.
- A Yahoo outage is not a missing ticker.
- A genuine unknown SEC issuer is a sanitized `404`.
- SEC transient failures are sanitized retryable `503` responses.
- Missing SEC operator configuration is a distinct sanitized non-retryable `503`.

## Investor agents

- **Buffett:** quality, profitability, leverage, and free cash flow.
- **Lynch:** growth at a reasonable price; requires market valuation inputs.
- **Graham:** margin of safety; requires market valuation inputs.
- **Munger:** business quality and avoidable risks.
- **Dalio:** leverage and risk context.

Each result includes `status` and `unavailable_metrics`. Missing required inputs earn no points and cannot produce a plausible complete verdict.

## Reliability and safety

- External exceptions must not expose URLs, query strings, credentials, provider text, stack traces, or connection details.
- SEC requests require an identifying `SEC_USER_AGENT` with an operational contact.
- Provider roles are explicit; no generic provider framework is required.
- Health and release identity are separate from feature readiness.
- Canaries use fixed inputs and avoid unbounded retries.

## Non-goals

- Trade execution or portfolio management.
- Real-time price guarantees or alerts.
- Support for non-SEC issuers in the initial architecture.
- A paid market-data dependency without an explicit product and licensing decision.
- Presenting automated analysis as financial advice.

## Acceptance criteria

1. Fixture-backed AAPL, AMZN, and JPM SEC facts match independent filing literals and provenance.
2. AMZN liabilities use same-filing derivation inputs.
3. JPM industrial free cash flow remains unavailable with an explicit sector reason.
4. SEC analysis succeeds when Yahoo is unavailable.
5. All affected agent results explicitly report incomplete inputs.
6. Unknown issuer, SEC outage, rate limit, malformed response, and missing configuration have distinct sanitized contracts.
7. Local API/browser verification, independent review, GitHub CI, exact-release verification, production canaries, and production browser verification all pass before release is called successful.

## Configuration

| Variable | Required | Purpose |
|---|---:|---|
| `SEC_USER_AGENT` | Yes | SEC application identity and operational contact |
| `OPENAI_API_KEY` | No | LLM-enhanced narratives |
| `PORT` | No | Server port; default `8000` |

## Disclaimer

GOATlens is for education and research, not investment advice. Users should verify source filings and consult qualified professionals.
