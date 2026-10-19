"""Authoritative annual company fundamentals from SEC EDGAR Company Facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
import re
from typing import Any, Dict, List, Optional

import httpx

from data_sources.symbols import canonicalize_class_symbol


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
    is_financial_institution: bool = False


@dataclass(frozen=True)
class MetricFactInput:
    role: str
    value: float
    provenance: FactProvenance


@dataclass(frozen=True)
class NormalizedMetric:
    value: Optional[float]
    formula: str
    formula_version: str = "sec_companyfacts_v1"
    inputs: tuple[MetricFactInput, ...] = ()
    availability_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return (
            self.value is not None
            and math.isfinite(float(self.value))
            and bool(self.inputs)
            and self.availability_reason is None
        )


class SECNormalization(dict):
    """Analysis values plus the exact SEC facts used to derive them."""

    def __init__(self, values: Dict[str, Any], metrics: Dict[str, NormalizedMetric]):
        super().__init__(values)
        self.metrics = metrics


def normalize_sec_fundamentals(data: SECFundamentals) -> SECNormalization:
    """Build the analysis contract without inventing values for missing facts."""

    reference = data.latest["revenue"].provenance
    metric_inputs: Dict[str, tuple[MetricFactInput, ...]] = {}

    def evidence_input(role: str, fact: Optional[FundamentalValue]):
        if fact is None or fact.value is None or fact.provenance is None:
            return None
        if not math.isfinite(float(fact.value)):
            return None
        return MetricFactInput(role, float(fact.value), fact.provenance)

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

    def fact_for_reference(
        metric: str, reference_provenance: Optional[FactProvenance]
    ) -> Optional[FundamentalValue]:
        if reference_provenance is None:
            return None
        declared_latest = data.latest[metric]
        if declared_latest.availability_reason is not None:
            return None
        if (
            declared_latest.value is not None
            and compatible_with(declared_latest.provenance, reference_provenance)
        ):
            return declared_latest
        return next(
            (
                item
                for item in data.history[metric]
                if item.value is not None
                and compatible_with(item.provenance, reference_provenance)
            ),
            None,
        )

    def latest(metric: str) -> Optional[float]:
        selected = fact_for_reference(metric, reference)
        return selected.value if selected is not None else None

    def ratio(numerator: Optional[float], denominator: Optional[float]):
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator

    def growth(metric: str, output_metric: str, years: int = 1):
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
        prior_reference = reference_years_ago(years)
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
        role_prefix = "income" if metric == "net_income" else metric
        inputs = (
            evidence_input(f"{role_prefix}.begin", prior),
            evidence_input(f"{role_prefix}.current", current),
        )
        valid_inputs = tuple(item for item in inputs if item is not None)
        if len(valid_inputs) == 2:
            metric_inputs[output_metric] = valid_inputs
        return (current.value / prior.value) ** (1 / years) - 1

    def earnings_turnaround() -> Optional[float]:
        if reference is None:
            return None
        current = fact_for_reference("net_income", reference)
        prior_reference = reference_years_ago(3)
        prior = fact_for_reference("net_income", prior_reference)
        if (
            current is None
            or current.value is None
            or prior is None
            or prior.value is None
        ):
            return None
        inputs = (
            evidence_input("income.begin", prior),
            evidence_input("income.current", current),
        )
        valid_inputs = tuple(item for item in inputs if item is not None)
        if len(valid_inputs) == 2:
            metric_inputs["earnings_turnaround"] = valid_inputs
        return 1.0 if prior.value <= 0 < current.value else 0.0

    def reference_years_ago(years: int) -> Optional[FactProvenance]:
        if reference is None:
            return None
        references = sorted(
            (
                item.provenance
                for item in data.history["revenue"]
                if item.provenance is not None
            ),
            key=lambda item: item.end,
        )
        try:
            current_index = references.index(reference)
        except ValueError:
            return None
        prior_index = current_index - years
        if prior_index < 0:
            return None
        span = references[prior_index : current_index + 1]
        end_dates = [date.fromisoformat(item.end) for item in span]
        if any(
            not 300 <= (later - earlier).days <= 430
            for earlier, later in zip(end_dates, end_dates[1:])
        ):
            return None
        return references[prior_index]

    def value_for_reference(
        metric: str, reference_provenance: Optional[FactProvenance]
    ) -> Optional[float]:
        selected = fact_for_reference(metric, reference_provenance)
        return selected.value if selected is not None else None

    def debt_for_reference(
        reference_provenance: Optional[FactProvenance],
    ) -> Optional[float]:
        components = debt_facts_for_reference(reference_provenance)
        if components is None:
            return None
        return sum(item.value for item in components if item.value is not None)

    def debt_facts_for_reference(
        reference_provenance: Optional[FactProvenance],
    ) -> Optional[List[FundamentalValue]]:
        components = [
            fact_for_reference("long_term_debt_current", reference_provenance),
            fact_for_reference("long_term_debt_noncurrent", reference_provenance),
            fact_for_reference("short_term_borrowings", reference_provenance),
        ]
        if any(component is None for component in components):
            return None
        return [component for component in components if component is not None]

    def debt_facts_for_period_end(
        period_end: Optional[str],
        metrics: tuple[str, ...] = (
            "long_term_debt_current",
            "long_term_debt_noncurrent",
            "short_term_borrowings",
        ),
    ) -> Optional[List[FundamentalValue]]:
        if period_end is None:
            return None
        components = []
        for metric in metrics:
            if data.latest[metric].availability_reason is not None:
                return None
            component = next(
                (
                    item
                    for item in data.history[metric]
                    if item.value is not None
                    and item.provenance is not None
                    and item.provenance.end == period_end
                ),
                None,
            )
            if component is None or component.provenance is None:
                return None
            components.append(component)
        contexts = {
            (
                item.provenance.end,
                item.provenance.accession,
                item.provenance.form,
                item.provenance.filed,
                item.provenance.fiscal_year,
            )
            for item in components
            if item.provenance is not None
        }
        if len(contexts) != 1:
            return None
        return components

    def debt_facts_periods_ago(
        years: int,
    ) -> tuple[Optional[List[FundamentalValue]], Optional[List[FundamentalValue]]]:
        if reference is None:
            return None, None
        period_ends = sorted(
            {
                item.provenance.end
                for item in data.history["long_term_debt_current"]
                if item.provenance is not None
            }
        )
        try:
            current_index = period_ends.index(reference.end)
        except ValueError:
            return None, None
        prior_index = current_index - years
        if prior_index < 0:
            return None, None
        span = [date.fromisoformat(item) for item in period_ends[prior_index : current_index + 1]]
        if any(
            not 300 <= (later - earlier).days <= 430
            for earlier, later in zip(span, span[1:])
        ):
            return None, None
        current_end = reference.end
        prior_end = period_ends[prior_index]
        full_metrics = (
            "long_term_debt_current",
            "long_term_debt_noncurrent",
            "short_term_borrowings",
        )

        def has_observation(metric: str, period_end: str) -> bool:
            return any(
                item.value is not None
                and item.provenance is not None
                and item.provenance.end == period_end
                for item in data.history[metric]
            )

        if all(
            has_observation(metric, period_end)
            for metric in full_metrics
            for period_end in (current_end, prior_end)
        ):
            current = debt_facts_for_period_end(current_end, full_metrics)
            prior = debt_facts_for_period_end(prior_end, full_metrics)
            return (current, prior) if current is not None and prior is not None else (None, None)

        return None, None

    def operating_margin_trend() -> Optional[float]:
        references = [
            item.provenance
            for item in data.history["revenue"]
            if item.provenance is not None
        ][:5]
        if len(references) != 5:
            return None
        references = sorted(references, key=lambda item: item.end)
        end_years = [date.fromisoformat(item.end).year for item in references]
        if any(
            later - earlier != 1
            for earlier, later in zip(end_years, end_years[1:])
        ):
            return None
        margins = []
        trend_inputs = []
        for period_reference in references:
            revenue_fact = fact_for_reference("revenue", period_reference)
            operating_income_fact = fact_for_reference(
                "operating_income", period_reference
            )
            if (
                revenue_fact is None
                or revenue_fact.value is None
                or revenue_fact.value <= 0
                or operating_income_fact is None
                or operating_income_fact.value is None
            ):
                return None
            period_year = date.fromisoformat(period_reference.end).year
            trend_inputs.extend(
                [
                    evidence_input(f"margin.{period_year}.revenue", revenue_fact),
                    evidence_input(
                        f"margin.{period_year}.operating_income",
                        operating_income_fact,
                    ),
                ]
            )
            margins.append(operating_income_fact.value / revenue_fact.value)
        x_mean = 2.0
        margin_mean = sum(margins) / len(margins)
        result = sum(
            (index - x_mean) * (margin - margin_mean)
            for index, margin in enumerate(margins)
        ) / sum((index - x_mean) ** 2 for index in range(5))
        valid_inputs = tuple(item for item in trend_inputs if item is not None)
        if len(valid_inputs) == 10:
            metric_inputs["margin_trend"] = valid_inputs
        return result

    def dividend_streak() -> Optional[int]:
        if reference is None:
            return None
        observations = [
            item
            for item in data.history["dividend_per_share"]
            if item.provenance is not None and item.value is not None
        ]
        observations.sort(
            key=lambda item: item.provenance.end if item.provenance else "",
            reverse=True,
        )
        current = [
            item
            for item in observations
            if item.provenance is not None
            and item.provenance.end == reference.end
        ]
        if len(current) != 1 or not compatible_with(
            current[0].provenance, reference
        ):
            return None
        current_index = observations.index(current[0])
        observations = observations[current_index:]
        selected = []
        previous_end = None
        streak = 0
        for item in observations:
            provenance = item.provenance
            if provenance is None or provenance.start is None:
                return None
            start = date.fromisoformat(provenance.start)
            end = date.fromisoformat(provenance.end)
            if not 300 <= (end - start).days <= 400:
                return None
            if previous_end is not None and not 300 <= (previous_end - end).days <= 430:
                return None
            period_year = end.year
            value = item.value
            if value is None:
                return None
            if value <= 0:
                selected.append(
                    evidence_input(f"streak.{period_year}.terminator", item)
                )
                valid_inputs = tuple(
                    evidence for evidence in selected if evidence is not None
                )
                if len(valid_inputs) == streak + 1:
                    metric_inputs["consecutive_dividend_years"] = valid_inputs
                    return streak
                return None
            selected.append(
                evidence_input(f"streak.{period_year}.dividend", item)
            )
            streak += 1
            previous_end = end
        return None

    revenue = latest("revenue")
    net_income = latest("net_income")
    equity = latest("equity")
    operating_cash_flow = latest("operating_cash_flow")
    capex = latest("capex_payments")
    shares = latest("shares_outstanding")
    debt_facts = {
        metric: fact_for_reference(metric, reference)
        for metric in (
            "long_term_debt_current",
            "long_term_debt_noncurrent",
            "short_term_borrowings",
            "debt_current",
            "long_term_debt",
        )
    }

    def legitimately_unreported(metric: str) -> bool:
        return (
            debt_facts[metric] is None
            and data.latest[metric].availability_reason
            != "conflicting_standard_concepts"
            and reference is not None
            and not any(
                item.value is not None
                and item.provenance is not None
                and item.provenance.end == reference.end
                for item in data.history[metric]
            )
        )

    reported_debt_roles: List[tuple[str, FundamentalValue]] = []
    debt_to_equity_formula = "reviewed_funded_debt_branch_divided_by_equity"
    if (
        debt_facts["long_term_debt_current"] is not None
        and debt_facts["long_term_debt_noncurrent"] is not None
        and (
            debt_facts["short_term_borrowings"] is not None
            or legitimately_unreported("short_term_borrowings")
        )
    ):
        reported_debt_roles = [
            ("long_term_current", debt_facts["long_term_debt_current"]),
            ("long_term_noncurrent", debt_facts["long_term_debt_noncurrent"]),
        ]
        if debt_facts["short_term_borrowings"] is not None:
            reported_debt_roles.append(
                ("short_term_borrowings", debt_facts["short_term_borrowings"])
            )
        debt_to_equity_formula = (
            "long_term_debt_components_plus_optional_short_term_borrowings_"
            "divided_by_equity"
        )
    elif (
        debt_facts["debt_current"] is not None
        and debt_facts["long_term_debt"] is not None
        and all(
            legitimately_unreported(metric)
            for metric in (
                "long_term_debt_current",
                "long_term_debt_noncurrent",
                "short_term_borrowings",
            )
        )
    ):
        reported_debt_roles = [
            ("debt_current", debt_facts["debt_current"]),
            ("long_term_debt", debt_facts["long_term_debt"]),
        ]
        debt_to_equity_formula = (
            "debt_current_plus_long_term_debt_divided_by_equity"
        )
    elif (
        debt_facts["short_term_borrowings"] is not None
        and debt_facts["long_term_debt_noncurrent"] is not None
        and all(
            legitimately_unreported(metric)
            for metric in (
                "long_term_debt_current",
                "debt_current",
                "long_term_debt",
            )
        )
    ):
        reported_debt_roles = [
            ("short_term_borrowings", debt_facts["short_term_borrowings"]),
            ("long_term_noncurrent", debt_facts["long_term_debt_noncurrent"]),
        ]
        debt_to_equity_formula = (
            "short_term_borrowings_plus_long_term_debt_noncurrent_"
            "divided_by_equity"
        )
    reported_debt_facts = [fact for _, fact in reported_debt_roles]
    total_debt = (
        sum(item.value for item in reported_debt_facts if item.value is not None)
        if reported_debt_facts
        else None
    )
    current_assets = latest("current_assets")
    current_liabilities = latest("current_liabilities")
    gross_profit = latest("gross_profit")
    cost_of_revenue = value_for_reference("cost_of_revenue", reference)
    operating_income = latest("operating_income")
    income_tax = latest("income_tax")
    pretax_income = latest("pretax_income")
    interest_expense = latest("interest_expense")
    dividend_per_share = latest("dividend_per_share")
    revenue_source = data.latest["revenue"].provenance
    is_financial_institution = data.is_financial_institution or bool(
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

    prior_reference = reference_years_ago(1)
    current_invested_capital = None
    prior_invested_capital = None
    if not is_financial_institution:
        current_debt = debt_for_reference(reference)
        prior_debt = debt_for_reference(prior_reference)
        current_equity = value_for_reference("equity", reference)
        prior_equity = value_for_reference("equity", prior_reference)
        current_cash = value_for_reference("cash_and_equivalents", reference)
        prior_cash = value_for_reference("cash_and_equivalents", prior_reference)
        if (
            current_debt is not None
            and current_equity is not None
            and current_cash is not None
        ):
            current_invested_capital = current_equity + current_debt - current_cash
        if (
            prior_debt is not None
            and prior_equity is not None
            and prior_cash is not None
        ):
            prior_invested_capital = prior_equity + prior_debt - prior_cash

    roic = None
    if (
        not is_financial_institution
        and operating_income is not None
        and income_tax is not None
        and pretax_income is not None
        and pretax_income > 0
        and current_invested_capital is not None
        and prior_invested_capital is not None
    ):
        tax_rate = income_tax / pretax_income
        average_invested_capital = (
            current_invested_capital + prior_invested_capital
        ) / 2
        if 0 <= tax_rate <= 1 and average_invested_capital > 0:
            roic = operating_income * (1 - tax_rate) / average_invested_capital
            current_debt_facts = debt_facts_for_reference(reference) or []
            prior_debt_facts = debt_facts_for_reference(prior_reference) or []
            roic_inputs = [
                evidence_input(
                    "income.current.operating_income",
                    fact_for_reference("operating_income", reference),
                ),
                evidence_input(
                    "income.current.tax",
                    fact_for_reference("income_tax", reference),
                ),
                evidence_input(
                    "income.current.pretax_income",
                    fact_for_reference("pretax_income", reference),
                ),
                evidence_input(
                    "capital.begin.equity",
                    fact_for_reference("equity", prior_reference),
                ),
                evidence_input(
                    "capital.end.equity", fact_for_reference("equity", reference)
                ),
                evidence_input(
                    "capital.begin.cash",
                    fact_for_reference("cash_and_equivalents", prior_reference),
                ),
                evidence_input(
                    "capital.end.cash",
                    fact_for_reference("cash_and_equivalents", reference),
                ),
            ]
            component_roles = (
                "long_term_current",
                "long_term_noncurrent",
                "short_term_borrowings",
            )
            roic_inputs.extend(
                evidence_input(f"capital.begin.debt.{role}", fact)
                for role, fact in zip(component_roles, prior_debt_facts)
            )
            roic_inputs.extend(
                evidence_input(f"capital.end.debt.{role}", fact)
                for role, fact in zip(component_roles, current_debt_facts)
            )
            valid_inputs = tuple(item for item in roic_inputs if item is not None)
            if len(valid_inputs) == 13:
                metric_inputs["roic"] = valid_inputs

    interest_coverage = None
    if (
        not is_financial_institution
        and operating_income is not None
        and interest_expense is not None
        and interest_expense > 0
    ):
        interest_coverage = operating_income / interest_expense

    debt_growth_3y = None
    current_debt_facts, prior_debt_facts = debt_facts_periods_ago(3)
    debt_three_years_ago = (
        sum(item.value for item in prior_debt_facts if item.value is not None)
        if prior_debt_facts is not None
        else None
    )
    current_debt_for_growth = (
        sum(item.value for item in current_debt_facts if item.value is not None)
        if current_debt_facts is not None
        else None
    )
    if (
        not is_financial_institution
        and current_debt_for_growth is not None
        and debt_three_years_ago is not None
        and current_debt_for_growth >= 0
        and debt_three_years_ago > 0
    ):
        debt_growth_3y = (
            current_debt_for_growth / debt_three_years_ago
        ) ** (1 / 3) - 1
        component_roles = (
            "long_term_current",
            "long_term_noncurrent",
            "short_term_borrowings",
        )[: len(current_debt_facts or [])]
        raw_inputs = [
            evidence_input(f"debt.current.{role}", fact)
            for role, fact in zip(component_roles, current_debt_facts or [])
        ] + [
            evidence_input(f"debt.begin.{role}", fact)
            for role, fact in zip(component_roles, prior_debt_facts or [])
        ]
        valid_inputs = tuple(item for item in raw_inputs if item is not None)
        if len(valid_inputs) == 2 * len(component_roles):
            metric_inputs["debt_growth_3y"] = valid_inputs
    gross_margin = ratio(gross_profit, revenue)
    gross_margin_formula = "gross_profit_divided_by_revenue"
    if (
        gross_margin is None
        and revenue is not None
        and revenue > 0
        and cost_of_revenue is not None
        and 0 <= cost_of_revenue <= revenue
    ):
        gross_margin = (revenue - cost_of_revenue) / revenue
        fallback_inputs = tuple(
            item
            for item in (
                evidence_input(
                    "income.current.revenue",
                    fact_for_reference("revenue", reference),
                ),
                evidence_input(
                    "income.current.cost_of_revenue",
                    fact_for_reference("cost_of_revenue", reference),
                ),
            )
            if item is not None
        )
        if len(fallback_inputs) == 2:
            metric_inputs["gross_margin"] = fallback_inputs
            gross_margin_formula = (
                "revenue_minus_cost_of_revenue_divided_by_revenue"
            )

    values = {
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
        "roe": ratio(net_income, equity) if equity is not None and equity > 0 else None,
        "revenue_growth": growth("revenue", "revenue_growth"),
        "earnings_growth": growth("net_income", "earnings_growth", years=3),
        "earnings_turnaround": earnings_turnaround(),
        "revenue_growth_5y": growth("revenue", "revenue_growth_5y", years=5),
        "free_cash_flow_per_share": ratio(free_cash_flow, shares),
        "debt_to_equity": (
            ratio(total_debt, equity) if equity is not None and equity > 0 else None
        ),
        "gross_margin": gross_margin,
        "operating_margin": ratio(operating_income, revenue),
        "roa": None,
        "roic": roic,
        "quick_ratio": None,
        "interest_coverage": interest_coverage,
        "cash_and_equivalents": latest("cash_and_equivalents"),
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "current_ratio": ratio(current_assets, current_liabilities),
        "capex_to_revenue": ratio(capex, revenue),
        "margin_trend": (
            None if is_financial_institution else operating_margin_trend()
        ),
        "dividend_per_share": dividend_per_share,
        "consecutive_dividend_years": dividend_streak(),
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
        "debt_growth_3y": debt_growth_3y,
        "expected_return": None,
    }

    def set_inputs(
        output_metric: str,
        specifications: List[tuple[str, str, Optional[FactProvenance]]],
    ) -> None:
        selected = tuple(
            item
            for role, source_metric, source_reference in specifications
            if (item := evidence_input(
                role, fact_for_reference(source_metric, source_reference)
            )) is not None
        )
        if len(selected) == len(specifications):
            metric_inputs[output_metric] = selected

    current_specs = {
        "revenue": [("income.current.revenue", "revenue", reference)],
        "roe": [
            ("income.current.net_income", "net_income", reference),
            ("capital.end.equity", "equity", reference),
        ],
        "profit_margin": [
            ("income.current.net_income", "net_income", reference),
            ("income.current.revenue", "revenue", reference),
        ],

        "free_cash_flow": [
            ("cashflow.current.operating", "operating_cash_flow", reference),
            ("cashflow.current.capex", "capex_payments", reference),
        ],
        "cash_and_equivalents": [
            ("cash.current", "cash_and_equivalents", reference)
        ],
        "current_ratio": [
            ("balance.current.current_assets", "current_assets", reference),
            ("balance.current.current_liabilities", "current_liabilities", reference),
        ],
        "current_assets": [
            ("balance.current.current_assets", "current_assets", reference)
        ],
        "total_liabilities": [
            ("balance.current.liabilities", "liabilities", reference)
        ],
        "shares_outstanding": [
            ("shares.period_end", "shares_outstanding", reference)
        ],
        "eps": [("income.current.diluted_eps", "diluted_eps", reference)],
        "book_value_per_share": [
            ("capital.end.equity", "equity", reference),
            ("shares.period_end", "shares_outstanding", reference),
        ],
        "gross_margin": [
            ("income.current.gross_profit", "gross_profit", reference),
            ("income.current.revenue", "revenue", reference),
        ],
        "operating_margin": [
            ("income.current.operating_income", "operating_income", reference),
            ("income.current.revenue", "revenue", reference),
        ],
        "capex_to_revenue": [
            ("cashflow.current.capex", "capex_payments", reference),
            ("income.current.revenue", "revenue", reference),
        ],
        "interest_coverage": [
            ("income.current.operating_income", "operating_income", reference),
            ("income.current.interest_expense", "interest_expense", reference),
        ],
        "dividend_per_share": [
            ("dividend.current", "dividend_per_share", reference)
        ],
    }
    for output_metric, specifications in current_specs.items():
        set_inputs(output_metric, specifications)
    debt_to_equity_inputs = tuple(
        item
        for item in (
            *(
                evidence_input(f"debt.current.{role}", fact)
                for role, fact in reported_debt_roles
            ),
            evidence_input(
                "capital.end.equity", fact_for_reference("equity", reference)
            ),
        )
        if item is not None
    )
    if (
        reported_debt_facts
        and len(debt_to_equity_inputs) == len(reported_debt_facts) + 1
    ):
        metric_inputs["debt_to_equity"] = debt_to_equity_inputs

    formulas = {
        "revenue": "reported",
        "roe": "net_income_divided_by_equity",
        "profit_margin": "net_income_divided_by_revenue",
        "debt_to_equity": debt_to_equity_formula,
        "free_cash_flow": "operating_cash_flow_minus_capex",
        "earnings_growth": "three_year_net_income_cagr",
        "earnings_turnaround": "negative_or_zero_to_positive_three_year_net_income_transition",
        "cash_and_equivalents": "reported",
        "current_ratio": "current_assets_divided_by_current_liabilities",
        "current_assets": "reported",
        "total_liabilities": "reported_or_compatibly_derived",
        "shares_outstanding": "reported_period_end",
        "eps": "reported_diluted_eps",
        "book_value_per_share": "equity_divided_by_period_end_shares",
        "consecutive_dividend_years": "consecutive_positive_common_dividend_years",
        "gross_margin": gross_margin_formula,
        "operating_margin": "operating_income_divided_by_revenue",
        "roic": "nopat_divided_by_average_invested_capital",
        "revenue_growth_5y": "five_year_revenue_cagr",
        "capex_to_revenue": "capex_divided_by_revenue",
        "interest_coverage": "operating_income_divided_by_interest_expense",
        "margin_trend": "five_year_operating_margin_ols_slope",
        "debt_growth_3y": "three_year_funded_debt_cagr",
        "dividend_per_share": "reported_annual_common_dividend_per_share",
    }
    metrics = {}
    for name, formula in formulas.items():
        value = values.get(name)
        inputs = metric_inputs.get(name, ())
        reason = None
        if value is None:
            if name in {"roe", "debt_to_equity"} and equity is not None and equity <= 0:
                reason = "nonpositive_equity_not_meaningful"
            else:
                reason = (
                    free_cash_flow_reason
                    if name == "free_cash_flow" and free_cash_flow_reason
                    else "required_inputs_unavailable"
                )
        elif not inputs:
            value = None
            values[name] = None
            reason = "required_provenance_unavailable"
        if name == "debt_to_equity" and value is None:
            formula = "reviewed_funded_debt_branch_divided_by_equity"
            inputs = ()
        metrics[name] = NormalizedMetric(
            value=float(value) if value is not None else None,
            formula=formula,
            formula_version=(
                "sec_companyfacts_debt_to_equity_v3"
                if name == "debt_to_equity"
                else "sec_companyfacts_v1"
            ),
            inputs=inputs,
            availability_reason=reason,
        )

    return SECNormalization(values, metrics)


_METRICS = {
    "revenue": ("duration", "USD", [
        "Revenues",
        "RevenuesNetOfInterestExpense",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
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
    "debt_current": ("instant", "USD", ["DebtCurrent"]),
    "long_term_debt": ("instant", "USD", ["LongTermDebt"]),
    "current_assets": ("instant", "USD", ["AssetsCurrent"]),
    "current_liabilities": ("instant", "USD", ["LiabilitiesCurrent"]),
    "cash_and_equivalents": (
        "instant",
        "USD",
        ["CashAndCashEquivalentsAtCarryingValue"],
    ),
    "gross_profit": ("duration", "USD", ["GrossProfit"]),
    "cost_of_revenue": (
        "duration",
        "USD",
        ["CostOfGoodsAndServicesSold", "CostOfRevenue"],
    ),
    "operating_income": ("duration", "USD", ["OperatingIncomeLoss"]),
    "shares_outstanding": ("instant", "shares", ["CommonStockSharesOutstanding"]),
    "income_tax": ("duration", "USD", ["IncomeTaxExpenseBenefit"]),
    "pretax_income": ("duration", "USD", [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ]),
    "interest_expense": ("duration", "USD", [
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
        "InterestExpense",
        "InterestAndDebtExpense",
    ]),
    "dividend_per_share": (
        "duration",
        "USD/shares",
        ["CommonStockDividendsPerShareDeclared"],
    ),
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
        provider_ticker = canonicalize_class_symbol(ticker)
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
                if str(item.get("ticker", "")).upper() == provider_ticker
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
        us_gaap = payload["facts"].get("us-gaap")
        if not isinstance(us_gaap, dict):
            raise SECInvalidResponseError("SEC returned an invalid response.")
        financial_institution_markers = {
            "Deposits",
            "LoansAndLeasesReceivableNetReportedAmount",
            "FederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell",
            "InsurancePremiumsRevenue",
            "PolicyholderBenefitsAndClaimsIncurredNet",
            "PolicyholderBenefitsAndClaimsPayableCurrent",
            "PolicyholderBenefitsAndClaimsPayableNoncurrent",
        }
        is_financial_institution = bool(
            financial_institution_markers.intersection(us_gaap)
            or "RevenuesNetOfInterestExpense" in us_gaap
        )

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
                if metric != "revenue"
            }
            revenue_kind, revenue_unit, revenue_concepts = _METRICS["revenue"]
            preferred_revenue_concept = None
            if (
                is_financial_institution
                and "RevenuesNetOfInterestExpense" in us_gaap
            ):
                preferred_revenue_concept = "RevenuesNetOfInterestExpense"
                revenue_concepts = [
                    preferred_revenue_concept,
                    *(
                        concept
                        for concept in revenue_concepts
                        if concept != preferred_revenue_concept
                    ),
                ]
            resolved["revenue"] = self._resolve_ordered_concepts(
                payload,
                kind=revenue_kind,
                unit=revenue_unit,
                concepts=revenue_concepts,
                as_of=as_of,
                preferred_concept=preferred_revenue_concept,
            )
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
            dividend_values, dividend_conflict = resolved["dividend_per_share"]
            if not dividend_values and not dividend_conflict:
                resolved["dividend_per_share"] = self._resolve_metric(
                    payload,
                    kind="duration",
                    unit="USD/shares",
                    aliases=["CommonStockDividendsPerShareCashPaid"],
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
            is_financial_institution=is_financial_institution,
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

    def _resolve_ordered_concepts(
        self,
        payload: Dict[str, Any],
        kind: str,
        unit: str,
        concepts: List[str],
        as_of: date,
        preferred_concept: Optional[str] = None,
    ) -> tuple[List[FundamentalValue], bool]:
        """Select one revenue concept family; never pool narrower concepts."""

        facts = payload.get("facts", {}).get("us-gaap", {})
        newest_by_concept = {}
        for concept in concepts:
            rows = facts.get(concept, {}).get("units", {}).get(unit, [])
            annual_ends = [
                row["end"]
                for row in rows
                if self._is_annual(row, kind, as_of)
            ]
            if annual_ends:
                newest_by_concept[concept] = max(annual_ends)
        if not newest_by_concept:
            return [], False
        newest_period = max(newest_by_concept.values())
        if (
            preferred_concept is not None
            and preferred_concept in newest_by_concept
            and newest_by_concept[preferred_concept] != newest_period
        ):
            return [], False
        selected_concept = next(
            concept
            for concept in concepts
            if newest_by_concept.get(concept) == newest_period
        )
        return self._resolve_metric(
            payload,
            kind=kind,
            unit=unit,
            aliases=[selected_concept],
            as_of=as_of,
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
