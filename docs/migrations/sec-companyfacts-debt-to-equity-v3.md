# SEC Company Facts debt-to-equity formula v3

`debt_to_equity` now publishes formula version
`sec_companyfacts_debt_to_equity_v3` and one of three mutually exclusive,
reviewed formula names:

- `long_term_debt_components_plus_optional_short_term_borrowings_divided_by_equity`
- `debt_current_plus_long_term_debt_divided_by_equity`
- `short_term_borrowings_plus_long_term_debt_noncurrent_divided_by_equity`

Version 3 selects exactly one funded-debt reporting branch from compatible annual
facts in the same filing context:

```text
LongTermDebtCurrent + LongTermDebtNoncurrent
+ ShortTermBorrowings (only when compatibly reported)
```

or:

```text
DebtCurrent + LongTermDebt
```

or:

```text
ShortTermBorrowings + LongTermDebtNoncurrent
```

Aggregate and component branches are never combined. An unreported component is
not zero. A branch may omit a concept only where that reviewed branch defines the
concept as absent; a conflicting or incompatible same-period fact makes the
ratio unavailable. The public evidence contains every fact consumed, identifies
the selected branch by formula name, and the response validator independently
selects the branch, requires its exact roles, and recomputes the ratio.

This supersedes the released `sec_companyfacts_v1` debt-to-equity contract,
which supported only the `LongTermDebtCurrent + LongTermDebtNoncurrent` pattern
with optional `ShortTermBorrowings`. Version 3 adds the TSLA and NFLX reporting
patterns without weakening conflict or provenance checks. The public response
shape and API version remain unchanged; clients that key formula behavior must
recognize the new formula version and branch-specific names.
