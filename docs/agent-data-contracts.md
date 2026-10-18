# GOATlens Agent Data Contracts

This document defines the evidence required for a strategy to return a score and verdict. Missing data never becomes zero, a default beta, an assumed return, or a synthetic conclusion.

## Shared rules

- SEC EDGAR Company Facts is authoritative for U.S. filing facts and historical series.
- Yahoo is a narrow, best-effort market boundary: the direct chart endpoint supplies timestamped current prices and adjusted price history. GOATlens does not ingest Yahoo financial statements or normalized ratios.
- Derived metrics carry their formula version, source timestamps, filing periods, and input provenance.
- A reported zero is valid only when the source explicitly reports zero. Absence is unavailable.
- Inputs combined within one derivation period must share the applicable full filing context: duration facts share start, end, accession, form, filing date, and fiscal year; instant facts share end, accession, form, filing date, and fiscal year. Multi-period formulas may compare internally compatible period groups from different annual filings; an unrelated revenue fact may locate fiscal chronology but is not forced into a debt derivation's filing context.
- Restatements supersede older facts. Equally ranked incompatible contexts fail closed.
- Industrial-company formulas are not silently applied to banks or insurers.
- Optional modifiers may improve or reduce a strategy score only when present. Their absence cannot be interpreted favorably. A completed result's `required_metric_values` is the exact set of available numeric metrics the strategy consumed, including optional context used for scoring, branches, insights, or concerns; unavailable optional metrics are omitted, and missing, mismatched, or extra keys invalidate the response.

## Contract inventory

The primary strategy contract contains **27 mandatory strategy slots across five agents, representing 23 unique metrics** after shared metrics are deduplicated. Lynch's SEC-backed turnaround branch substitutes `earnings_turnaround` for `earnings_growth`, expanding the union of executable mandatory metrics to 24. The centralized executable inventory is `_AGENT_CONSUMPTION_CONTRACTS`; it distinguishes mandatory completion evidence from optional-but-consumed evidence. This replaces the 36-field legacy inventory: institutional ownership, management beneficial ownership, unscaled cash balance, industrial interest coverage, current dividend yield, dividend-history length, approximate ROIC, historical debt growth, and revenue variability are optional context rather than completeness requirements. Fabricated expected return was removed and replaced by required trailing realized return. Every available consumed metric must have exactly one typed public metric-evidence contract.

## Buffett

Required evidence:

- return on equity;
- net profit margin;
- debt to equity;
- free cash flow.

The current cash-flow metric is `operating cash flow - capital expenditures` and must be labeled **free cash flow**, not owner earnings. A future owner-earnings estimate requires an explicit maintenance-capex and working-capital methodology.

## Lynch

Required evidence:

- timestamped market price divided by compatible SEC diluted EPS;
- positive multi-year earnings growth, or an SEC-backed negative-to-positive earnings-turnaround flag when CAGR is mathematically undefined;
- debt to equity.

Cash and cash equivalents is optional reported context, not a score input: an unscaled cash balance is not comparable across issuers and must not create a favorable default. A negative-to-positive three-year earnings transition is classified and scored as a distinct recovery signal; conventional CAGR and PEG are explicitly not applicable, and the public result warns that two endpoints do not establish recovery durability. `reported_13f_long_ownership_pct` is an optional discovery modifier, not a completeness requirement. If present, it must be labeled as lagged 13F-reported long ownership with period and coverage limitations. Missing ownership cannot receive the historical low-ownership bonus.

## Graham

Required evidence:

- timestamped market price;
- SEC diluted EPS and period-end common shares;
- SEC common equity, current assets, current liabilities, and total liabilities.

Ratio and derived-value formulas:

- `P/E = price / diluted EPS`, only for positive EPS;
- `book value per share = common equity / period-end common shares`;
- `P/B = price / book value per share`, only for positive book value;
- `current ratio = current assets / current liabilities`;
- `NCAV/share = (current assets - total liabilities) / period-end common shares`;
- trailing-FY dividend yield = latest completed-FY common DPS / timestamped price;
- consecutive dividend years = uninterrupted completed fiscal years with positive common DPS.

The dividend record uses the declared common-dividend-per-share concept first. A reported SEC dividend per share of zero produces a known zero trailing-FY yield and fails the dividend criterion without invalidating the analysis. Current yield and dividend-history length are optional criteria: missing evidence earns no points, is disclosed publicly, and does not invalidate an otherwise supported Graham analysis. When history is available, the defensive-investor threshold is twenty years and must not silently relax to ten. Missing dividend facts remain unavailable, not zero.

## Munger

Required evidence:

- gross margin;
- operating margin;
- five-year revenue growth;
- capital expenditures to revenue;
- debt to equity;
- five-year operating-margin trend.

`management_beneficial_ownership_pct` is an optional incentives modifier sourced from an issuer proxy Item 403 table. Missing ownership cannot become a low-ownership red flag.

Approximate accounting ROIC is an optional quality signal, not a completeness blocker. When absent, the public result states that capital-efficiency points were excluded. When present it uses:

- `tax rate = tax expense / pretax income`, accepted only when pretax income is positive and the rate is within `[0, 1]`;
- `NOPAT = operating income * (1 - tax rate)`;
- `invested capital = equity + funded debt - cash` at adjacent fiscal year ends;
- `ROIC = NOPAT / average(beginning invested capital, ending invested capital)`.

Operating-margin trend is the ordinary-least-squares slope of five consecutive annual operating margins, expressed as decimal change per fiscal year.

Gross margin uses compatible SEC gross profit and revenue when reported. If reported gross profit is absent, it may be derived only from compatible facts in the same filing context as `(revenue - cost of revenue) / revenue`; the evidence exposes both facts and the versioned formula.

## Dalio

GOATlens provides a **single-company risk-resilience lens**, not a claim of full portfolio risk parity. It does not fabricate a forward expected return.

Required evidence:

- trailing beta to the S&P 500;
- annualized 252-session volatility;
- S&P 500 return correlation;
- realized total return over the same 252-session window;
- debt to equity;

Market calculations use total-return-adjusted closes. Asset and benchmark returns are inner-joined on the full prior-session/current-session pair without forward filling. Volatility requires 252 valid asset return intervals derived from 253 adjusted-close observations. Beta and correlation require at least 126 aligned return intervals. Public series evidence labels these as return intervals and discloses their exact count and window rather than mislabeling the count as raw close observations.

`realized_total_return_252d` is backward-looking and must never be labeled `expected_return`. Any risk-adjusted ratio is explicitly trailing and uses a same-as-of risk-free-rate source or is omitted.

Revenue growth variability may be shown descriptively but is not labeled macroeconomic cyclicality. A future macro-regime model would require an independently versioned GDP/inflation contract.

Industrial interest coverage is optional context because some issuers, including AAPL in recent filings, do not separately disclose current-period interest expense. Missing coverage cannot become zero, a low-coverage concern, or a completeness blocker when debt to equity and funded-debt trend are available.

Debt reduction is not automatically distress. Debt trend is reported with leverage and coverage rather than mapped to a distressed state solely because debt fell rapidly.

Three-year funded-debt CAGR is optional historical context. When absent, Dalio may assess current leverage and market risk, but the public insight must label debt trajectory unavailable and must not infer a trend.

## Yahoo chart contract

Yahoo Finance is an unofficial, best-effort dependency with no SLA. Schema
drift, throttling, and outages may make market evidence unavailable. Its use in
a public product carries public-use risk; this temporary MVP dependency still
requires product-owner and legal acceptance or migration to a licensed source.

Per uncached analysis, request at most:

1. the selected symbol; and
2. `^GSPC`.

Use `range=2y`, `interval=1d`, `events=div,splits`, and no pre/post-market bars. Do not invoke normalized-fundamentals endpoints.

The normalized response includes:

- provider and retrieval timestamp, kept distinct from observation time;
- quote timestamp and currency;
- process-local cache age and an explicit fresh/stale/unknown label;
- current regular-market price;
- raw and adjusted daily closes;
- observation counts;
- volatility, beta, correlation, and realized-return outputs;
- field-level availability reasons.

Malformed, stale, incomplete, rate-limited, or unavailable responses remain unavailable and expose only stable sanitized errors.

## Consensus semantics

Completed investor verdicts are equally weighted ordinal votes from `strong_sell`
through `strong_buy`; raw strategy scores are not averaged because the five strategy
rubrics are not calibrated as one common scale. With an odd number of completed
analyses, the consensus is the median verdict. With an even number, middle votes in
the same direction resolve conservatively toward `hold`; middle votes from different
directions produce `mixed` with directional support unavailable.

The legacy API field `agreement_score` is retained for compatibility but means
**directional support**: the share of completed agents on the consensus side.
`buy` and `strong_buy` share one direction, as do `sell` and `strong_sell`.
It does not measure identical verdict intensity, confidence calibration, model
independence, track record, or forecast accuracy. Structural mixed/no-majority
warnings are deterministic and cannot be replaced by generated consensus prose.

## Release gate

PR #9 cannot be merge-ready until an exact-SHA Render preview proves:

- all five selected strategies complete for the fixed AAPL canary from real provider inputs;
- five requested and five completed results are preserved;
- directional support is independently checked against completed agent verdicts;
- every required metric is finite and carries source/formula metadata;
- no missing input was converted to zero or another default;
- provider outage leaves SEC-backed results available and affected strategies explicitly incomplete;
- the live browser renders all five cards, agreement, provenance, and degraded states without runtime or console errors;
- a repeated cold or expired-cache market request succeeds, so a one-off warm provider session is insufficient evidence.

