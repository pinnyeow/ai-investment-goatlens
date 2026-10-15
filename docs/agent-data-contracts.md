# GOATlens Agent Data Contracts

This document defines the evidence required for a strategy to return a score and verdict. Missing data never becomes zero, a default beta, an assumed return, or a synthetic conclusion.

## Shared rules

- SEC EDGAR Company Facts is authoritative for U.S. filing fundamentals.
- Yahoo's direct chart endpoint is best-effort market enrichment for timestamped prices and adjusted price history. GOATlens does not use Yahoo `quoteSummary`, `Ticker.info`, or financial statements in the core analysis path.
- Derived metrics carry their formula version, source timestamps, filing periods, and input provenance.
- A reported zero is valid only when the source explicitly reports zero. Absence is unavailable.
- Inputs combined within one derivation period must share the applicable full filing context: duration facts share start, end, accession, form, filing date, and fiscal year; instant facts share end, accession, form, filing date, and fiscal year. Multi-period formulas may compare internally compatible period groups from different annual filings; an unrelated revenue fact may locate fiscal chronology but is not forced into a debt derivation's filing context.
- Restatements supersede older facts. Equally ranked incompatible contexts fail closed.
- Industrial-company formulas are not silently applied to banks or insurers.
- Optional modifiers may improve or reduce a strategy score only when present. Their absence cannot be interpreted favorably.

## Contract inventory

The current strategy contract contains **31 required strategy slots across five agents, representing 27 unique metrics** after shared metrics are deduplicated. This replaces the 36-field legacy inventory: institutional ownership, management beneficial ownership, unscaled cash balance, industrial interest coverage, and revenue variability are optional context rather than completeness requirements; fabricated expected return was removed and replaced by required trailing realized return. The executable inventory is `_AGENT_REQUIRED_METRICS`, and every unique required metric must have exactly one typed public metric-evidence contract.

## Buffett

Required evidence:

- return on equity;
- net profit margin;
- debt to equity;
- free cash flow.

The current cash-flow metric is `operating cash flow - capital expenditures` and must be labeled **free cash flow**, not owner earnings. A future owner-earnings estimate requires an explicit maintenance-capex and working-capital methodology.

## Lynch

Required evidence:

- price-to-earnings ratio, derived from timestamped market price and compatible SEC diluted EPS;
- positive multi-year earnings growth;
- debt to equity.

Cash and cash equivalents is optional reported context, not a score input: an unscaled cash balance is not comparable across issuers and must not create a favorable default. `reported_13f_long_ownership_pct` is an optional discovery modifier, not a completeness requirement. If present, it must be labeled as lagged 13F-reported long ownership with period and coverage limitations. Missing ownership cannot receive the historical low-ownership bonus.

## Graham

Required evidence:

- timestamped market price;
- SEC diluted EPS and period-end common shares;
- SEC common equity, current assets, current liabilities, and total liabilities;
- SEC common-dividend history.

Derived values:

- `P/E = price / diluted EPS`, only for positive EPS;
- `book value per share = common equity / period-end common shares`;
- `P/B = price / book value per share`, only for positive book value;
- `current ratio = current assets / current liabilities`;
- `NCAV/share = (current assets - total liabilities) / period-end common shares`;
- trailing-FY dividend yield = latest completed-FY common DPS / timestamped price;
- consecutive dividend years = uninterrupted completed fiscal years with positive common DPS.

The dividend record uses the declared common-dividend-per-share concept first. Missing dividend facts are unavailable, not zero. The defensive-investor threshold is twenty years; the implementation must not silently relax it to ten.

## Munger

Required evidence:

- gross margin;
- operating margin;
- approximate accounting ROIC;
- five-year revenue growth;
- capital expenditures to revenue;
- debt to equity;
- five-year operating-margin trend.

`management_beneficial_ownership_pct` is an optional incentives modifier sourced from an issuer proxy Item 403 table. Missing ownership cannot become a low-ownership red flag.

Approximate accounting ROIC:

- `tax rate = tax expense / pretax income`, accepted only when pretax income is positive and the rate is within `[0, 1]`;
- `NOPAT = operating income * (1 - tax rate)`;
- `invested capital = equity + funded debt - cash` at adjacent fiscal year ends;
- `ROIC = NOPAT / average(beginning invested capital, ending invested capital)`.

Operating-margin trend is the ordinary-least-squares slope of five consecutive annual operating margins, expressed as decimal change per fiscal year.

## Dalio

GOATlens provides a **single-company risk-resilience lens**, not a claim of full portfolio risk parity. It does not fabricate a forward expected return.

Required evidence:

- trailing beta to the S&P 500;
- annualized 252-session volatility;
- S&P 500 return correlation;
- realized total return over the same 252-session window;
- debt to equity;
- three-year funded-debt CAGR.

Market calculations use total-return-adjusted closes. Asset and benchmark returns are inner-joined on the full prior-session/current-session pair without forward filling. Volatility requires 252 valid asset return intervals derived from 253 adjusted-close observations. Beta and correlation require at least 126 aligned return intervals. Public series evidence labels these as return intervals and discloses their exact count and window rather than mislabeling the count as raw close observations.

`realized_total_return_252d` is backward-looking and must never be labeled `expected_return`. Any risk-adjusted ratio is explicitly trailing and uses a same-as-of risk-free-rate source or is omitted.

Revenue growth variability may be shown descriptively but is not labeled macroeconomic cyclicality. A future macro-regime model would require an independently versioned GDP/inflation contract.

Industrial interest coverage is optional context because some issuers, including AAPL in recent filings, do not separately disclose current-period interest expense. Missing coverage cannot become zero, a low-coverage concern, or a completeness blocker when debt to equity and funded-debt trend are available.

Debt reduction is not automatically distress. Debt trend is reported with leverage and coverage rather than mapped to a distressed state solely because debt fell rapidly.

## Yahoo chart contract

Per uncached analysis, request at most:

1. the selected symbol; and
2. `^GSPC`.

Use `range=2y`, `interval=1d`, `events=div,splits`, and no pre/post-market bars. Do not invoke crumb-authenticated quote-summary endpoints.

The normalized response includes:

- provider and fetch timestamp;
- quote timestamp and currency;
- current regular-market price;
- raw and adjusted daily closes;
- observation counts;
- volatility, beta, correlation, and realized-return outputs;
- field-level availability reasons.

Malformed, stale, incomplete, rate-limited, or unavailable responses remain unavailable and expose only stable sanitized errors.

## Release gate

PR #9 cannot be merge-ready until an exact-SHA Render preview proves:

- all five selected strategies complete for the fixed AAPL canary from real provider inputs;
- five requested and five completed results are preserved;
- GOAT Agreement is non-null and independently checked;
- every required metric is finite and carries source/formula metadata;
- no missing input was converted to zero or another default;
- provider outage leaves SEC-backed results available and affected strategies explicitly incomplete;
- the live browser renders all five cards, agreement, provenance, and degraded states without runtime or console errors;
- a repeated cold or expired-cache market request succeeds, so a one-off warm provider session is insufficient evidence.

