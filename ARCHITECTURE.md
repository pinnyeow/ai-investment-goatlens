# GOATlens Architecture

## What the product is

An educational tool: "how would Buffett, Graham, Lynch, Munger and Dalio each think about this stock, and why?" Trust comes from showing the numbers and the reasoning, not from refusing to answer. The success measures are:

1. **Coverage:** every covered company gets a verdict, and among non-financial companies all five investors give a view ≥ 85% of the time (CI's 25-company sample requires ≥ 90%; the daily smoke test checks the full set). Banks and insurers legitimately skip some industrial checks, so they're measured by verdict, not all-five.
2. **Accuracy:** headline numbers match the company's annual report (spot-check; Yahoo's revenue and net income matched SEC filings exactly for AAPL, WMT and KO when this design was chosen).

Any change that lowers either one should not ship.

## Design

| Piece | File | Job |
|---|---|---|
| Fetch | `backend/lens/data.py` | One Yahoo Finance call set per company → plain metrics (number or `None`) |
| Score | `backend/lens/investors.py` | Five checklists; each criterion returns points + a sentence, or `None` if its inputs are missing |
| Build | `backend/lens/build.py` | Runs nightly in GitHub Actions; writes `snapshots/{TICKER}.json`, `index.json`, `coverage.json` |
| Publish | `.github/workflows/nightly-snapshots.yml` | Builds with a read-only token, then a separate job (main only) force-pushes the JSON to the `data` branch (one commit, no history growth) |
| Serve | `backend/main.py` | FastAPI on Render; reads the `data` branch over HTTPS, caches 30 minutes |
| Page | `frontend/index.html` | Single page; shows each investor's checklist and the numbers used |

### Rules

- **Missing ≠ zero.** A criterion with missing inputs is skipped and shown as "not scored" with the reason. Score = points earned ÷ points available on the checks that ran, scaled to −100…100.
- **Half a checklist minimum.** Below 50% of an investor's checklist, that investor shows "not enough data" instead of a verdict. The consensus uses whichever investors have a view.
- **Banks and insurers:** free cash flow, gross margin, debt-to-equity, current ratio and similar industrial measures are marked "not meaningful" rather than scored.
- **Fit, not advice.** Verdicts are phrased as the investor's reaction (Would buy / Leaning yes / On the fence / Leaning no / Would avoid), and an investor can't give a "strong" verdict on less than 75% of its checklist. When investors disagree (agreement < 50%), the headline says "Split: 3 say yes, 1 says no" instead of an average.
- **Foreign filers (ADRs)** report in their home currency. Absolute and per-share figures are converted at Yahoo's daily FX rate only if the converted numbers reproduce Yahoo's own P/E (which proves the share count is in ADR units); otherwise they're marked "not comparable" rather than shown wrong.
- **Never crash a user request on data quality.** Validation happens at build time; the server only reads finished files.
- **A bad night never removes a company.** If a fetch fails, the previous snapshot is kept and labelled stale. If fewer than 80% of companies fetch, nothing is published.

## Why this design (November 2026)

The previous design fetched SEC XBRL and Yahoo data live on every request and refused to answer unless every input could be proven consistent. Each fix added stricter checks; by v0.2 it was ~24,000 lines and about half of well-known companies showed an error or no verdict. Tested alternatives:

| Source | Result |
|---|---|
| Yahoo Finance full data from GitHub Actions | 100 of 100 companies, 0 errors, ~1.2 s each |
| Yahoo Finance full data from Render | Throttled (HTTP 429 / 401 invalid crumb): shared IPs |
| SEC Company Facts | Free and official but every company tags figures differently; needs per-company special cases |
| Stooq | Now behind a bot check |

So: fetch once a night from GitHub's servers, and let Render only serve files.

## Known limits

- Yahoo is unofficial and can change without notice; the CI `live-build-sample` job and daily smoke test catch breakage, and the site keeps serving the last good data.
- Growth figures use the 4–5 years of annual reports Yahoo provides (Graham's 10-year tests are approximated).
- Companies outside `universe.txt` aren't covered; the API returns a helpful 404 (and explains that funds and crypto aren't rated).
- GitHub pauses scheduled workflows in public repos after 60 days without commits; the smoke test's data-age check will flag it. Re-enable it in the Actions tab.
