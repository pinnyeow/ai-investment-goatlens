"""Authoritative annual company fundamentals from SEC EDGAR Company Facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
import re
from typing import Any, Dict, List, Optional

import httpx


class SECError(Exception):
    """Base exception for SEC fundamentals failures."""


class SECConfigurationError(SECError):
    """Raised when SEC request identification is not configured."""


class SECRateLimitError(SECError):
    """Raised when SEC asks the caller to reduce request rate."""

    retryable = True


class SECIssuerNotFoundError(SECError):
    """Raised when a symbol has no issuer in the SEC company map."""

    retryable = False


class SECUnavailableError(SECError):
    """Raised for retryable SEC transport or capacity failures."""

    retryable = True


class SECInvalidResponseError(SECError):
    """Raised when SEC returns malformed JSON or an unusable response shape."""

    retryable = False


@dataclass(frozen=True)
class FactProvenance:
    namespace: str
    concept: str
    unit: str
    start: Optional[str]
    end: str
    accession: str
    form: str
    filed: date
    fiscal_year: Optional[int]
    derived: bool = False
    inputs: List["FactProvenance"] = field(default_factory=list)


@dataclass(frozen=True)
class FundamentalValue:
    value: Optional[float]
    provenance: Optional[FactProvenance]
    availability_reason: Optional[str] = None


@dataclass(frozen=True)
class SECFundamentals:
    ticker: str
    cik: str
    company_name: str
    latest: Dict[str, FundamentalValue]
    history: Dict[str, List[FundamentalValue]]


def normalize_sec_fundamentals(data: SECFundamentals) -> Dict[str, Any]:
    """Build the analysis contract without inventing values for missing facts."""

    reference = data.latest["revenue"].provenance

    def compatible_with(
        provenance: Optional[FactProvenance],
        reference_provenance: Optional[FactProvenance],
    ) -> bool:
        if provenance is None or reference_provenance is None:
            return False
        return (
            provenance.end == reference_provenance.end
            and provenance.accession == reference_provenance.accession
            and provenance.form == reference_provenance.form
            and provenance.filed == reference_provenance.filed
            and provenance.fiscal_year == reference_provenance.fiscal_year
            and (
                provenance.start is None
                or provenance.start == reference_provenance.start
            )
        )

    def latest(metric: str) -> Optional[float]:
        if reference is None:
            return None
        declared_latest = data.latest[metric]
        if declared_latest.availability_reason == "conflicting_standard_concepts":
            return None

        if declared_latest.provenance is None:
            return None
        if compatible_with(declared_latest.provenance, reference):
            return declared_latest.value
        for item in data.history[metric]:
            if item.value is not None and compatible_with(item.provenance, reference):
                return item.value
        return None

    def ratio(numerator: Optional[float], denominator: Optional[float]):
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator

    def growth(metric: str, years: int = 1):
        items = data.history[metric]
        if reference is None:
            return None
        current = next(
            (
                item for item in items
                if item.value is not None
                and compatible_with(item.provenance, reference)
            ),
            None,
        )
        if current is None or current.provenance is None:
            return None
        current_year = date.fromisoformat(reference.end).year
        prior_reference = next(
            (
                item.provenance for item in data.history["revenue"]
                if item.provenance is not None
                and date.fromisoformat(item.provenance.end).year
                == current_year - years
            ),
            None,
        )
        if prior_reference is None:
            return None
        prior = next(
            (
                item for item in items
                if item.value is not None
                and compatible_with(item.provenance, prior_reference)
            ),
            None,
        )
        if prior is None or prior.value is None or current.value <= 0 or prior.value <= 0:
            return None
        return (current.value / prior.value) ** (1 / years) - 1

    revenue = latest("revenue")
    net_income = latest("net_income")
    equity = latest("equity")
    operating_cash_flow = latest("operating_cash_flow")
    capex = latest("capex_payments")
    shares = latest("shares_outstanding")
    debt_current = latest("long_term_debt_current")
    debt_noncurrent = latest("long_term_debt_noncurrent")
    short_term_borrowings = latest("short_term_borrowings")
    total_debt = (
        debt_current + debt_noncurrent + short_term_borrowings
        if debt_current is not None
        and debt_noncurrent is not None
        and short_term_borrowings is not None
        else None
    )
    current_assets = latest("current_assets")
    current_liabilities = latest("current_liabilities")
    gross_profit = latest("gross_profit")
    operating_income = latest("operating_income")
    revenue_source = data.latest["revenue"].provenance
    is_financial_institution = bool(
        revenue_source
        and revenue_source.concept == "RevenuesNetOfInterestExpense"
    )
    free_cash_flow = (
        operating_cash_flow - capex
        if (
            not is_financial_institution
            and operating_cash_flow is not None
            and capex is not None
        )
        else None
    )
    if is_financial_institution:
        free_cash_flow_reason = (
            "industrial_free_cash_flow_not_applicable_to_financial_institution"
        )
        cash_flow_comparability = "not_comparable_to_industrial_companies"
    elif free_cash_flow is None:
        free_cash_flow_reason = "required_cash_flow_fact_unavailable"
        cash_flow_comparability = "industrial"
    else:
        free_cash_flow_reason = None
        cash_flow_comparability = "industrial"
    return {
        "ticker": data.ticker,
        "company_name": data.company_name,
        "sector": "Unknown",
        "industry": "Unknown",
        "revenue": revenue,
        "net_income": net_income,
        "total_assets": latest("assets"),
        "total_liabilities": latest("liabilities"),
        "stockholders_equity": equity,
        "operating_cash_flow": operating_cash_flow,
        "capex": capex,
        "free_cash_flow": free_cash_flow,
        "free_cash_flow_availability_reason": free_cash_flow_reason,
        "operating_cash_flow_comparability": cash_flow_comparability,
        "eps": latest("diluted_eps"),
        "shares_outstanding": shares,
        "diluted_weighted_average_shares": latest("diluted_shares"),
        "book_value_per_share": ratio(equity, shares),
        "profit_margin": ratio(net_income, revenue),
        "roe": ratio(net_income, equity),
        "revenue_growth": growth("revenue"),
        "earnings_growth": growth("net_income"),
        "revenue_growth_5y": growth("revenue", years=5),
        "free_cash_flow_per_share": ratio(free_cash_flow, shares),
        "debt_to_equity": ratio(total_debt, equity),
        "gross_margin": ratio(gross_profit, revenue),
        "operating_margin": ratio(operating_income, revenue),
        "roa": None,
        "roic": None,
        "quick_ratio": None,
        "interest_coverage": None,
        "cash_and_equivalents": latest("cash_and_equivalents"),
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "current_ratio": ratio(current_assets, current_liabilities),
        "capex_to_revenue": ratio(capex, revenue),
        "margin_trend": None,
        "consecutive_dividend_years": None,
        "market_cap": None,
        "current_price": None,
        "pe_ratio": None,
        "pb_ratio": None,
        "ps_ratio": None,
        "peg_ratio": None,
        "dividend_yield": None,
        "beta": None,
        "volatility_252d": None,
        "insider_ownership": None,
        "institutional_ownership": None,
        "revenue_cyclicality": None,
        "correlation_sp500": None,
        "debt_growth_3y": None,
        "expected_return": None,
    }


_METRICS = {
    "revenue": ("duration", "USD", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenuesNetOfInterestExpense",
        "Revenues",
        "SalesRevenueNet",
    ]),
    "net_income": ("duration", "USD", ["NetIncomeLoss"]),
    "assets": ("instant", "USD", ["Assets"]),
    "liabilities": ("instant", "USD", ["Liabilities"]),
    "equity": ("instant", "USD", ["StockholdersEquity"]),
    "operating_cash_flow": (
        "duration",
        "USD",
        ["NetCashProvidedByUsedInOperatingActivities"],
    ),
    "capex_payments": ("duration", "USD", [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ]),
    "diluted_eps": ("duration", "USD/shares", ["EarningsPerShareDiluted"]),
    "diluted_shares": (
        "duration",
        "shares",
        ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    ),
    "long_term_debt_current": ("instant", "USD", ["LongTermDebtCurrent"]),
    "long_term_debt_noncurrent": (
        "instant",
        "USD",
        ["LongTermDebtNoncurrent"],
    ),
    "short_term_borrowings": ("instant", "USD", ["ShortTermBorrowings"]),
    "current_assets": ("instant", "USD", ["AssetsCurrent"]),
    "current_liabilities": ("instant", "USD", ["LiabilitiesCurrent"]),
    "cash_and_equivalents": (
        "instant",
        "USD",
        ["CashAndCashEquivalentsAtCarryingValue"],
    ),
    "gross_profit": ("duration", "USD", ["GrossProfit"]),
    "operating_income": ("duration", "USD", ["OperatingIncomeLoss"]),
    "shares_outstanding": ("instant", "shares", ["CommonStockSharesOutstanding"]),
}


class SECFundamentalsClient:
    """Fetch and normalize annual SEC Company Facts with provenance."""

    def __init__(
        self,
        user_agent: str,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        parts = user_agent.split()
        has_contact = any(
            re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", part)
            for part in parts
        )
        has_application = any("@" not in part and len(part) >= 2 for part in parts)
        if not has_contact or not has_application:
            raise SECConfigurationError(
                "SEC_USER_AGENT must identify the application and contact email."
            )
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=httpx.Timeout(30.0, connect=10.0),
            transport=transport,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self._client.aclose()

    async def get_fundamentals(
        self,
        ticker: str,
        as_of: Optional[date] = None,
    ) -> SECFundamentals:
        ticker = ticker.strip().upper()
        as_of = as_of or date.today()
        ticker_response = await self._get_response(
            "https://www.sec.gov/files/company_tickers.json"
        )
        self._raise_structured_status(ticker_response)
        ticker_response.raise_for_status()
        ticker_map = self._json_object(ticker_response)
        if any(not isinstance(item, dict) for item in ticker_map.values()):
            raise SECInvalidResponseError("SEC returned an invalid response.")
        company = next(
            (
                item
                for item in ticker_map.values()
                if str(item.get("ticker", "")).upper() == ticker
            ),
            None,
        )
        if company is None:
            raise SECIssuerNotFoundError(
                "Ticker is not present in the SEC company map."
            )
        try:
            cik = str(int(company["cik_str"])).zfill(10)
        except (KeyError, TypeError, ValueError):
            raise SECInvalidResponseError("SEC returned an invalid response.") from None
        facts_response = await self._get_response(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        )
        self._raise_structured_status(facts_response)
        facts_response.raise_for_status()
        payload = self._json_object(facts_response)
        if not isinstance(payload.get("facts"), dict):
            raise SECInvalidResponseError("SEC returned an invalid response.")

        try:
            resolved = {
                metric: self._resolve_metric(
                    payload,
                    kind=kind,
                    unit=unit,
                    aliases=aliases,
                    as_of=as_of,
                )
                for metric, (kind, unit, aliases) in _METRICS.items()
            }
            # Prefer the broader authoritative short-term borrowing total. Use
            # commercial paper only when that broader concept is wholly absent;
            # never add both because commercial paper may be included in it.
            short_term_values, short_term_conflict = resolved[
                "short_term_borrowings"
            ]
            if not short_term_values and not short_term_conflict:
                resolved["short_term_borrowings"] = self._resolve_metric(
                    payload,
                    kind="instant",
                    unit="USD",
                    aliases=["CommercialPaper"],
                    as_of=as_of,
                )
            if not resolved["liabilities"][0]:
                derived_liabilities, derived_conflict = self._derive_liabilities(
                    payload, as_of
                )
                if derived_liabilities or derived_conflict:
                    resolved["liabilities"] = (
                        derived_liabilities,
                        derived_conflict,
                    )
        except (AttributeError, KeyError, TypeError, ValueError):
            raise SECInvalidResponseError("SEC returned an invalid response.") from None
        if not any(values for values, _ in resolved.values()):
            raise SECInvalidResponseError("SEC returned an invalid response.")
        if not resolved["revenue"][0] or resolved["revenue"][1]:
            raise SECInvalidResponseError("SEC returned an invalid response.")
        history = {metric: values for metric, (values, _) in resolved.items()}
        latest = {
            metric: values[0]
            if values and not resolved[metric][1]
            else FundamentalValue(
                value=None,
                provenance=None,
                availability_reason=(
                    "conflicting_standard_concepts"
                    if resolved[metric][1]
                    else "issuer_does_not_report_a_standard_entity_wide_annual_fact"
                ),
            )
            for metric, values in history.items()
        }
        return SECFundamentals(
            ticker=ticker,
            cik=cik,
            company_name=payload.get("entityName") or company.get("title") or ticker,
            latest=latest,
            history=history,
        )

    @staticmethod
    def _raise_structured_status(response: httpx.Response) -> None:
        if response.status_code in {403, 429}:
            raise SECRateLimitError("SEC data is temporarily unavailable.")
        if response.status_code in {408, 425} or response.status_code >= 500:
            raise SECUnavailableError("SEC data is temporarily unavailable.")
        if response.status_code >= 400:
            raise SECError("SEC request failed.")

    @staticmethod
    def _json_object(response: httpx.Response) -> Dict[str, Any]:
        payload: Any = None
        invalid = False
        try:
            payload = response.json()
        except ValueError:
            invalid = True
        if invalid or not isinstance(payload, dict):
            raise SECInvalidResponseError("SEC returned an invalid response.")
        return payload

    async def _get_response(self, url: str) -> httpx.Response:
        response: Optional[httpx.Response] = None
        try:
            response = await self._client.get(url)
        except httpx.RequestError:
            pass
        if response is None:
            raise SECUnavailableError("SEC data is temporarily unavailable.")
        return response

    def _resolve_metric(
        self,
        payload: Dict[str, Any],
        kind: str,
        unit: str,
        aliases: List[str],
        as_of: date,
    ) -> tuple[List[FundamentalValue], bool]:
        facts = payload.get("facts", {}).get("us-gaap", {})
        candidates: Dict[str, List[Dict[str, Any]]] = {}
        for concept in aliases:
            concept_rows = facts.get(concept, {}).get("units", {}).get(unit, [])
            winners: Dict[str, Dict[str, Any]] = {}
            for row in concept_rows:
                if not self._is_annual(row, kind, as_of):
                    continue
                key = row["end"]
                rank = (
                    row.get("filed", ""),
                    1 if row.get("form") == "10-K/A" else 0,
                    row.get("accn", ""),
                )
                current = winners.get(key)
                if current is None or rank > current["rank"]:
                    winners[key] = {
                        "rows": [row],
                        "rank": rank,
                        "concept": concept,
                    }
                elif rank == current["rank"]:
                    current["rows"].append(row)
            for key, winner in winners.items():
                for row in winner["rows"]:
                    candidates.setdefault(key, []).append({
                        "row": row,
                        "rank": winner["rank"],
                        "concept": winner["concept"],
                    })

        values = []
        conflict_found = False
        newest_period = max(
            candidates,
            key=lambda period: period.split("/")[-1],
            default=None,
        )
        for period, period_candidates in candidates.items():
            winning_rank = max(candidate["rank"] for candidate in period_candidates)
            winning_candidates = [
                candidate for candidate in period_candidates
                if candidate["rank"] == winning_rank
            ]
            reported_values = {
                candidate["row"]["val"] for candidate in winning_candidates
            }
            winning_contexts = {
                (
                    candidate["row"].get("start"),
                    candidate["row"].get("end"),
                    candidate["row"].get("accn"),
                    candidate["row"].get("form"),
                    candidate["row"].get("filed"),
                    candidate["row"].get("fy"),
                )
                for candidate in winning_candidates
            }
            if len(reported_values) > 1 or len(winning_contexts) > 1:
                if period == newest_period:
                    conflict_found = True
                continue
            winner = winning_candidates[0]
            row = winner["row"]
            provenance = FactProvenance(
                namespace="us-gaap",
                concept=winner["concept"],
                unit=unit,
                start=row.get("start"),
                end=row["end"],
                accession=row["accn"],
                form=row["form"],
                filed=date.fromisoformat(row["filed"]),
                fiscal_year=row.get("fy"),
            )
            values.append(FundamentalValue(value=row["val"], provenance=provenance))
        return (
            sorted(
                values,
                key=lambda item: item.provenance.end if item.provenance else "",
                reverse=True,
            ),
            conflict_found,
        )

    def _derive_liabilities(
        self, payload: Dict[str, Any], as_of: date
    ) -> tuple[List[FundamentalValue], bool]:
        facts = payload.get("facts", {}).get("us-gaap", {})
        total_rows = facts.get("LiabilitiesAndStockholdersEquity", {}).get(
            "units", {}
        ).get("USD", [])
        equity_rows = facts.get("StockholdersEquity", {}).get("units", {}).get(
            "USD", [])
        def grouped(rows: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
            result: Dict[tuple, List[Dict[str, Any]]] = {}
            for row in rows:
                if not self._is_annual(row, "instant", as_of):
                    continue
                key = (
                    row.get("end"), row.get("accn"), row.get("filed"),
                    row.get("form"), row.get("fy"),
                )
                result.setdefault(key, []).append(row)
            return result

        totals = grouped(total_rows)
        equities = grouped(equity_rows)
        period_candidates: Dict[
            str,
            List[tuple[tuple[str, int, str], Optional[FundamentalValue], bool]],
        ] = {}
        for period_key in totals.keys() & equities.keys():
            end, accession, filed, form, fiscal_year = period_key
            total_candidates = totals[period_key]
            equity_candidates = equities[period_key]
            rank = (filed, 1 if form == "10-K/A" else 0, accession)
            if (
                len({row["val"] for row in total_candidates}) != 1
                or len({row["val"] for row in equity_candidates}) != 1
            ):
                period_candidates.setdefault(end, []).append((rank, None, True))
                continue
            total = total_candidates[0]
            equity = equity_candidates[0]

            def provenance(row: Dict[str, Any], concept: str) -> FactProvenance:
                return FactProvenance(
                    namespace="us-gaap",
                    concept=concept,
                    unit="USD",
                    start=None,
                    end=end,
                    accession=accession,
                    form=row["form"],
                    filed=date.fromisoformat(row["filed"]),
                    fiscal_year=row.get("fy"),
                )

            total_source = provenance(total, "LiabilitiesAndStockholdersEquity")
            equity_source = provenance(equity, "StockholdersEquity")
            derived_source = FactProvenance(
                namespace="us-gaap",
                concept=(
                    "derived:LiabilitiesAndStockholdersEquity-StockholdersEquity"
                ),
                unit="USD",
                start=None,
                end=end,
                accession=accession,
                form=total["form"],
                filed=date.fromisoformat(total["filed"]),
                fiscal_year=total.get("fy"),
                derived=True,
                inputs=[total_source, equity_source],
            )
            value = FundamentalValue(
                value=total["val"] - equity["val"],
                provenance=derived_source,
            )
            period_candidates.setdefault(end, []).append((rank, value, False))

        winners: Dict[str, FundamentalValue] = {}
        conflicting_ends = set()
        for end, candidates in period_candidates.items():
            winning_rank = max(candidate[0] for candidate in candidates)
            winning = [candidate for candidate in candidates if candidate[0] == winning_rank]
            winning_values = {
                candidate[1].value
                for candidate in winning
                if candidate[1] is not None
            }
            if any(candidate[2] for candidate in winning) or len(winning_values) != 1:
                conflicting_ends.add(end)
                continue
            winners[end] = next(
                candidate[1] for candidate in winning if candidate[1] is not None
            )
        values = [
            item
            for item in sorted(
                winners.values(),
                key=lambda item: (
                    item.provenance.end if item.provenance else ""
                ),
                reverse=True,
            )
        ]
        newest_end = max(period_candidates, default=None)
        return values, newest_end in conflicting_ends

    @staticmethod
    def _is_annual(row: Dict[str, Any], kind: str, as_of: date) -> bool:
        value = row.get("val")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise SECInvalidResponseError("SEC returned an invalid response.")
        if row.get("form") not in {"10-K", "10-K/A"} or row.get("fp") != "FY":
            return False
        filed = row.get("filed")
        if not filed or date.fromisoformat(filed) > as_of:
            return False
        if kind == "instant":
            return bool(row.get("end")) and not row.get("start")
        if not row.get("start") or not row.get("end"):
            return False
        days = (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days
        return 300 <= days <= 400
