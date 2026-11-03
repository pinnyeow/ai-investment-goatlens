# SEC Company Facts debt-to-equity formula v4

`debt_to_equity` now publishes formula version
`sec_companyfacts_debt_to_equity_v4` and one of three mutually exclusive,
reviewed formula names:

- `long_term_debt_components_plus_optional_short_term_borrowings_divided_by_equity`
- `debt_current_plus_long_term_debt_noncurrent_divided_by_equity`
- `short_term_borrowings_plus_long_term_debt_noncurrent_divided_by_equity`

Version 4 selects exactly one funded-debt reporting branch from compatible annual
facts in the same filing context:

```text
LongTermDebtCurrent + LongTermDebtNoncurrent
+ ShortTermBorrowings (only when compatibly reported)
```

or:

```text
DebtCurrent + LongTermDebtNoncurrent
```

or:

```text
ShortTermBorrowings + LongTermDebtNoncurrent
```

Every additive branch is mutually exclusive. `DebtCurrent` already includes
short-term debt and current maturities, so it is combined only with the explicitly
noncurrent `LongTermDebtNoncurrent` fact. When a same-context `DebtCurrent` fact
coexists with the narrower current-debt components, the `DebtCurrent` branch has
priority so additional current debt cannot be silently omitted. It is never added
to `LongTermDebt`, which can overlap current maturities, and a broader
`LongTermDebt` fact cannot justify omitting a compatibly reported `DebtCurrent`
fact. When the retained facts do not prove one complete exclusive branch, the
metric fails closed.

An unreported component is not zero. A branch may omit a concept only where that
reviewed branch defines the concept as absent; a conflicting or incompatible
same-period fact makes the ratio unavailable. The public evidence contains every
fact consumed, identifies the selected branch by formula name, and the response
validator independently reconstructs applicability from retained debt and equity
leaves. It rejects both omitted components and a coordinated available-to-
unavailable downgrade when a complete reviewed branch remains reconstructable.

This supersedes the unreleased v3 recipe, which admitted
`DebtCurrent + LongTermDebt` and could double count current maturities. The public
response shape also adds canonical CIK binding in the same unreleased release:
top-level analysis, fundamentals, metric evidence and SEC inputs, investor cards,
and comparison projections must all carry the ticker-map CIK. The SEC adapter
validates that the Companyfacts payload CIK exactly matches that authoritative
lookup before normalization. This proves lookup/payload identity agreement, not
provider authenticity beyond the HTTPS SEC retrieval boundary.
