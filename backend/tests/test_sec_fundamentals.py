"""Behavioral tests for authoritative SEC fundamentals."""

import json
import traceback
from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
import pytest

from data_sources.sec import (
    SECFundamentalsClient,
    SECIssuerNotFoundError,
    SECRateLimitError,
    SECInvalidResponseError,
    SECUnavailableError,
    SECConfigurationError,
    FundamentalValue,
    normalize_sec_fundamentals,
)


FIXTURES = Path(__file__).parent / "fixtures" / "sec"


@pytest.mark.parametrize("user_agent", ["", "contact@example.com", "GOATlens", "x @"])
def test_sec_requires_identifying_application_and_contact(user_agent):
    with pytest.raises(SECConfigurationError):
        SECFundamentalsClient(user_agent=user_agent)


def load_json(name):
    return json.loads((FIXTURES / name).read_text())


def sec_transport(company_file):
    ticker_map = load_json("company_tickers.json")
    companyfacts = load_json(company_file)
    return sec_payload_transport(ticker_map, companyfacts)


def sec_payload_transport(ticker_map, companyfacts):
    async def handler(request):
        if request.url.path.endswith("/files/company_tickers.json"):
            return httpx.Response(200, json=ticker_map)
        if "/api/xbrl/companyfacts/" in request.url.path:
            return httpx.Response(200, json=companyfacts)
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_aapl_fundamentals_match_literal_10k_values_with_provenance():
    expected = load_json("expected_fundamentals.json")["companies"]["AAPL"]

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_transport("companyfacts_aapl.json"),
    ) as client:
        result = await client.get_fundamentals(
            "AAPL",
            as_of=date.fromisoformat("2026-10-14"),
        )

    assert result.ticker == "AAPL"
    assert result.cik == expected["cik"]
    assert result.company_name == "Apple Inc."

    for metric, literal in expected["values"].items():
        actual = result.latest[metric]
        assert actual.value == literal["value"]
        if literal["value"] is None:
            assert actual.provenance is None
            assert actual.availability_reason == (
                "issuer_does_not_report_a_standard_entity_wide_annual_fact"
            )
            continue
        assert actual.availability_reason is None
        assert actual.provenance is not None
        assert actual.provenance.unit == literal["unit"]
        assert actual.provenance.end == expected["period_end"]
        assert actual.provenance.accession == expected["accession"]
        assert actual.provenance.form == "10-K"
        assert actual.provenance.filed <= date.fromisoformat("2026-10-14")

    normalized = normalize_sec_fundamentals(result)
    assert normalized["debt_to_equity"] == pytest.approx(
        1.3380304612588665, rel=1e-12
    )
    assert len(result.history["revenue"]) >= 10
    assert len(result.history["net_income"]) >= 10
    assert len(result.history["assets"]) >= 10


@pytest.mark.asyncio
async def test_sec_derives_multi_year_agent_metrics_without_defaults():
    def duration(year, value):
        return {
            "start": f"{year}-01-01",
            "end": f"{year}-12-31",
            "val": value,
            "accn": f"filing-{year}",
            "fy": year,
            "fp": "FY",
            "form": "10-K",
            "filed": f"{year + 1}-02-01",
        }

    def instant(year, value):
        return {**duration(year, value), "start": None}

    years = [2019, 2020, 2021, 2022, 2023, 2024]
    revenue = [90, 100, 110, 120, 130, 140]
    operating_income = [8, 10, 12, 15, 18, 21]
    net_income = [5, 6, 7, 8, 10, 12]
    equity = [48, 50, 55, 60, 65, 70]
    cash = [9, 10, 10, 12, 13, 14]
    noncurrent_debt = [32, 30, 28, 26, 24, 22]
    dividends = [0, 1, 1, 1, 1, 1]
    facts = {
        "Revenues": {"units": {"USD": [duration(y, v) for y, v in zip(years, revenue)]}},
        "OperatingIncomeLoss": {"units": {"USD": [duration(y, v) for y, v in zip(years, operating_income)]}},
        "NetIncomeLoss": {"units": {"USD": [duration(y, v) for y, v in zip(years, net_income)]}},
        "StockholdersEquity": {"units": {"USD": [instant(y, v) for y, v in zip(years, equity)]}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [instant(y, v) for y, v in zip(years, cash)]}},
        "LongTermDebtCurrent": {"units": {"USD": [{**instant(y, 5), "accn": f"debt-{y}" if y <= 2021 else f"filing-{y}"} for y in years]}},
        "LongTermDebtNoncurrent": {"units": {"USD": [{**instant(y, v), "accn": f"debt-{y}" if y <= 2021 else f"filing-{y}"} for y, v in zip(years, noncurrent_debt)]}},
        "ShortTermBorrowings": {"units": {"USD": [{**instant(y, 2), "accn": f"debt-{y}" if y <= 2021 else f"filing-{y}"} for y in years]}},
        "IncomeTaxExpenseBenefit": {"units": {"USD": [duration(2024, 5)]}},
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": {"units": {"USD": [duration(2024, 25)]}},
        "InterestExpenseNonoperating": {"units": {"USD": [duration(2024, 3)]}},
        "CommonStockDividendsPerShareDeclared": {"units": {"USD/shares": [duration(y, v) for y, v in zip(years, dividends)]}},
    }
    payload = {"entityName": "Metric Corp", "facts": {"us-gaap": facts}}
    ticker_map = {"0": {"cik_str": 1, "ticker": "METR", "title": "Metric Corp"}}

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        result = await client.get_fundamentals("METR", as_of=date(2026, 10, 14))

    normalized = normalize_sec_fundamentals(result)
    assert normalized["dividend_per_share"] == 1
    assert normalized["consecutive_dividend_years"] == 5
    assert normalized["roic"] == pytest.approx(0.2)
    assert normalized["margin_trend"] == pytest.approx(0.012937062937062937)
    assert normalized["interest_coverage"] == 7
    assert normalized["debt_growth_3y"] == pytest.approx(-0.06075984576762794)
    assert normalized["earnings_growth"] == pytest.approx(0.19681696117715086)

    earnings_inputs = {item.role: item for item in normalized.metrics["earnings_growth"].inputs}
    assert earnings_inputs["income.begin"].provenance.end == "2021-12-31"
    assert earnings_inputs["income.begin"].provenance.accession == "filing-2021"
    assert earnings_inputs["income.current"].provenance.end == "2024-12-31"
    assert len(normalized.metrics["margin_trend"].inputs) == 10
    assert len(normalized.metrics["debt_growth_3y"].inputs) == 6
    assert any(
        item.role == "streak.2019.terminator" and item.value == 0
        for item in normalized.metrics["consecutive_dividend_years"].inputs
    )
    roic_roles = {item.role for item in normalized.metrics["roic"].inputs}
    assert {
        "income.current.operating_income",
        "income.current.tax",
        "income.current.pretax_income",
        "capital.begin.equity",
        "capital.end.equity",
        "capital.begin.cash",
        "capital.end.cash",
    } <= roic_roles

    incompatible_history = {key: list(items) for key, items in result.history.items()}
    incompatible_history["short_term_borrowings"] = [
        replace(
            item,
            provenance=replace(item.provenance, accession="incompatible-2021"),
        )
        if item.provenance is not None and item.provenance.end == "2021-12-31"
        else item
        for item in incompatible_history["short_term_borrowings"]
    ]
    incompatible = normalize_sec_fundamentals(
        replace(result, history=incompatible_history)
    )
    assert incompatible["debt_growth_3y"] is None
    assert not incompatible.metrics["debt_growth_3y"].available

    shifted_history = {key: list(items) for key, items in result.history.items()}
    for metric in ("revenue", "net_income"):
        shifted_history[metric] = [
            replace(item, provenance=replace(item.provenance, end="2022-01-02"))
            if item.provenance is not None and item.provenance.end == "2021-12-31"
            else item
            for item in shifted_history[metric]
        ]
    shifted = normalize_sec_fundamentals(replace(result, history=shifted_history))
    assert shifted["earnings_growth"] == pytest.approx(0.19681696117715086)
    assert shifted.metrics["earnings_growth"].inputs[0].provenance.end == "2022-01-02"

    mismatched_dividend_history = {
        key: list(items) for key, items in result.history.items()
    }
    mismatched_dividend_history["dividend_per_share"] = [
        replace(
            item,
            provenance=replace(item.provenance, accession="wrong-current-filing"),
        )
        if item.provenance is not None and item.provenance.end == "2024-12-31"
        else item
        for item in mismatched_dividend_history["dividend_per_share"]
    ]
    mismatched_dividend = normalize_sec_fundamentals(
        replace(result, history=mismatched_dividend_history)
    )
    assert mismatched_dividend["consecutive_dividend_years"] is None

    malformed_dividend_history = {
        key: list(items) for key, items in result.history.items()
    }
    malformed_dividend_history["dividend_per_share"] = [
        replace(
            item,
            provenance=replace(item.provenance, start="2022-01-01"),
        )
        if item.provenance is not None and item.provenance.end == "2023-12-31"
        else item
        for item in malformed_dividend_history["dividend_per_share"]
    ]
    malformed_dividend = normalize_sec_fundamentals(
        replace(result, history=malformed_dividend_history)
    )
    assert malformed_dividend["consecutive_dividend_years"] is None


@pytest.mark.asyncio
async def test_broader_short_term_borrowings_supersedes_commercial_paper():
    duration = {
        "start": "2024-01-01", "end": "2024-12-31", "val": 100,
        "accn": "filing", "fy": 2024, "fp": "FY", "form": "10-K",
        "filed": "2025-02-01",
    }
    instant = {**duration, "start": None}
    payload = {"entityName": "Debt Corp", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [duration]}},
        "LongTermDebtCurrent": {"units": {"USD": [{**instant, "val": 10}]}},
        "LongTermDebtNoncurrent": {"units": {"USD": [{**instant, "val": 20}]}},
        "CommercialPaper": {"units": {"USD": [{**instant, "val": 5}]}},
        "ShortTermBorrowings": {"units": {"USD": [{**instant, "val": 15}]}},
        "StockholdersEquity": {"units": {"USD": [{**instant, "val": 100}]}},
    }}}
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "DEBT", "title": "Debt Corp"}
    }
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        result = await client.get_fundamentals("DEBT", as_of=date(2026, 10, 14))

    short_term = result.latest["short_term_borrowings"]
    assert short_term.value == 15
    assert short_term.provenance is not None
    assert short_term.provenance.concept == "ShortTermBorrowings"
    assert normalize_sec_fundamentals(result)["debt_to_equity"] == 0.45


@pytest.mark.asyncio
async def test_disagreeing_revenue_aliases_are_unavailable_not_silently_preferred():
    annual = {
        "start": "2024-01-01",
        "end": "2024-12-31",
        "accn": "0000000000-25-000001",
        "fy": 2024,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-02-01",
    }
    payload = {
        "cik": 1,
        "entityName": "Conflict Corp",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [
                        {**annual, "val": 100},
                        {
                            **annual,
                            "start": "2023-01-01",
                            "end": "2023-12-31",
                            "accn": "0000000000-24-000001",
                            "filed": "2024-02-01",
                            "fy": 2023,
                            "val": 90,
                        },
                    ]}
                },
                "Revenues": {"units": {"USD": [{**annual, "val": 120}]}},
            }
        },
    }
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "CONF", "title": "Conflict Corp"}
    }

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError):
            await client.get_fundamentals(
                "CONF", as_of=date.fromisoformat("2026-10-14")
            )


@pytest.mark.asyncio
async def test_duration_aliases_with_same_end_and_different_starts_conflict():
    annual = {
        "end": "2024-12-31",
        "accn": "0000000000-25-000001",
        "fy": 2024,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-02-01",
    }
    payload = {
        "cik": 1,
        "entityName": "Conflict Corp",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {**annual, "start": "2024-01-01", "val": 100},
                            {
                                **annual,
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "accn": "0000000000-24-000001",
                                "filed": "2024-02-01",
                                "fy": 2023,
                                "val": 90,
                            },
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [{**annual, "start": "2024-01-02", "val": 120}]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                **annual,
                                "start": None,
                                "val": 500,
                            }
                        ]
                    }
                },
            }
        },
    }
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "CONF", "title": "Conflict Corp"}
    }
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError):
            await client.get_fundamentals("CONF", as_of=date(2026, 10, 14))


@pytest.mark.asyncio
async def test_same_concept_same_filing_context_with_two_values_conflicts():
    annual = {
        "start": "2024-01-01",
        "end": "2024-12-31",
        "accn": "0000000000-25-000001",
        "fy": 2024,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-02-01",
    }
    payload = {
        "entityName": "Conflict Corp",
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                {**annual, "val": 100},
                {**annual, "val": 120},
                {
                    **annual,
                    "start": "2023-01-01",
                    "end": "2023-12-31",
                    "accn": "0000000000-24-000001",
                    "filed": "2024-02-01",
                    "fy": 2023,
                    "val": 90,
                },
            ]}},
            "Assets": {"units": {"USD": [{
                **annual, "start": None, "val": 500,
            }]}},
        }},
    }
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "CONF", "title": "Conflict Corp"}
    }
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError):
            await client.get_fundamentals("CONF", as_of=date(2026, 10, 14))


@pytest.mark.asyncio
async def test_equal_values_in_different_winning_contexts_are_invalid():
    annual = {
        "end": "2024-12-31", "accn": "filing", "fy": 2024, "fp": "FY",
        "form": "10-K", "filed": "2025-02-01", "val": 100,
    }
    payload = {"entityName": "Ambiguous Corp", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {**annual, "start": "2024-01-01"},
            {**annual, "start": "2024-01-02"},
        ]}},
    }}}
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "AMB", "title": "Ambiguous Corp"}
    }
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError):
            await client.get_fundamentals("AMB", as_of=date(2026, 10, 14))

    payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"].reverse()
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError):
            await client.get_fundamentals("AMB", as_of=date(2026, 10, 14))


@pytest.mark.asyncio
async def test_newer_alias_filing_supersedes_older_same_period_value():
    base = {
        "start": "2024-01-01", "end": "2024-12-31", "fy": 2024,
        "fp": "FY", "form": "10-K",
    }
    payload = {"entityName": "Restated Corp", "facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {"USD": [{
                **base, "accn": "new", "filed": "2025-03-01", "val": 100,
            }]}
        },
        "Revenues": {"units": {"USD": [{
            **base, "accn": "old", "filed": "2025-02-01", "val": 90,
        }]}},
    }}}
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "REST", "title": "Restated Corp"}
    }
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        result = await client.get_fundamentals("REST", as_of=date(2026, 10, 14))

    assert result.latest["revenue"].value == 100
    assert result.latest["revenue"].provenance.accession == "new"


@pytest.mark.asyncio
async def test_latest_ratios_do_not_mix_fiscal_periods_or_filings():
    payload = load_json("companyfacts_aapl.json")
    net_income = payload["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    payload["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"] = [
        row for row in net_income if row.get("end") != "2025-09-27"
    ]

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(load_json("company_tickers.json"), payload),
    ) as client:
        result = await client.get_fundamentals(
            "AAPL", as_of=date.fromisoformat("2026-10-14")
        )

    normalized = normalize_sec_fundamentals(result)
    assert result.latest["revenue"].provenance.end == "2025-09-27"
    assert result.latest["net_income"].provenance.end != "2025-09-27"
    assert normalized["profit_margin"] is None
    assert normalized["roe"] is None


@pytest.mark.asyncio
async def test_growth_requires_current_and_prior_reference_compatible_contexts():
    revenue_current = {
        "start": "2024-01-01", "end": "2024-12-31", "val": 100,
        "accn": "revenue-current", "fy": 2024, "fp": "FY",
        "form": "10-K", "filed": "2025-02-01",
    }
    revenue_prior = {
        **revenue_current, "start": "2023-01-01", "end": "2023-12-31",
        "val": 80, "accn": "revenue-prior", "fy": 2023,
        "filed": "2024-02-01",
    }
    incompatible_income_current = {
        **revenue_current, "start": "2025-01-01", "end": "2025-12-31",
        "val": 150, "accn": "income-current", "fy": 2025,
        "filed": "2026-02-01",
    }
    incompatible_income_prior = {
        **revenue_current, "val": 100, "accn": "income-prior",
    }
    payload = {"entityName": "Growth Corp", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [revenue_current, revenue_prior]}},
        "NetIncomeLoss": {"units": {"USD": [
            incompatible_income_current, incompatible_income_prior,
        ]}},
    }}}
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "GROW", "title": "Growth Corp"}
    }
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        result = await client.get_fundamentals("GROW", as_of=date(2026, 10, 14))

    normalized = normalize_sec_fundamentals(result)
    assert normalized["net_income"] is None
    assert normalized["earnings_growth"] is None


@pytest.mark.asyncio
async def test_duration_ratios_require_the_same_fiscal_period_start():
    payload = load_json("companyfacts_aapl.json")
    rows = payload["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    for row in rows:
        if row.get("end") == "2025-09-27" and row.get("start") == "2024-09-29":
            row["start"] = "2024-09-30"

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(load_json("company_tickers.json"), payload),
    ) as client:
        result = await client.get_fundamentals("AAPL", as_of=date(2026, 10, 14))

    normalized = normalize_sec_fundamentals(result)
    assert normalized["profit_margin"] is None


@pytest.mark.asyncio
async def test_period_end_shares_select_reference_compatible_history_fact():
    payload = load_json("companyfacts_aapl.json")
    rows = payload["facts"]["us-gaap"]["CommonStockSharesOutstanding"][
        "units"
    ]["shares"]
    rows.append({
        "end": "2025-10-17",
        "val": 14776353000,
        "accn": "0000320193-25-000079",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-10-31",
        "frame": "CY2025Q3I",
    })

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(load_json("company_tickers.json"), payload),
    ) as client:
        result = await client.get_fundamentals("AAPL", as_of=date(2026, 10, 14))

    normalized = normalize_sec_fundamentals(result)
    assert result.history["shares_outstanding"][0].value == 14776353000
    assert normalized["shares_outstanding"] == 14773260000
    assert normalized["book_value_per_share"] == pytest.approx(
        73733000000 / 14773260000, rel=1e-12
    )


@pytest.mark.asyncio
async def test_aapl_five_year_revenue_growth_is_annualized_cagr():
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_transport("companyfacts_aapl.json"),
    ) as client:
        result = await client.get_fundamentals(
            "AAPL", as_of=date.fromisoformat("2026-10-14")
        )
    normalized = normalize_sec_fundamentals(result)
    assert normalized["revenue_growth_5y"] == pytest.approx(
        0.08677354924090741, rel=1e-9
    )


@pytest.mark.asyncio
async def test_amzn_liabilities_use_same_accession_total_and_equity():
    expected = load_json("expected_fundamentals.json")["companies"]["AMZN"]

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_transport("companyfacts_amzn.json"),
    ) as client:
        result = await client.get_fundamentals(
            "AMZN", as_of=date.fromisoformat("2026-10-14")
        )

    liabilities = result.latest["liabilities"]
    assert liabilities.value == expected["values"]["liabilities"]["value"]
    assert liabilities.availability_reason is None
    assert liabilities.provenance is not None
    assert liabilities.provenance.derived is True
    assert liabilities.provenance.end == expected["period_end"]
    assert liabilities.provenance.accession == expected["accession"]
    assert len(liabilities.provenance.inputs) == 2
    assert {
        item.concept for item in liabilities.provenance.inputs
    } == {"LiabilitiesAndStockholdersEquity", "StockholdersEquity"}
    assert {
        item.accession for item in liabilities.provenance.inputs
    } == {expected["accession"]}
    assert len(result.history["liabilities"]) >= 10


@pytest.mark.asyncio
async def test_conflicting_same_context_liability_inputs_are_unavailable():
    annual_duration = {
        "start": "2024-01-01", "end": "2024-12-31", "val": 100,
        "accn": "0000000000-25-000001", "fy": 2024, "fp": "FY",
        "form": "10-K", "filed": "2025-02-01",
    }
    annual_instant = {**annual_duration, "start": None}
    payload = {"entityName": "Conflict Corp", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [annual_duration]}},
        "LiabilitiesAndStockholdersEquity": {"units": {"USD": [
            {**annual_instant, "val": 500},
            {**annual_instant, "val": 550},
        ]}},
        "StockholdersEquity": {"units": {"USD": [
            {**annual_instant, "val": 200},
        ]}},
    }}}
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "CONF", "title": "Conflict Corp"}
    }
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        result = await client.get_fundamentals("CONF", as_of=date(2026, 10, 14))

    assert result.latest["liabilities"].value is None
    assert result.latest["liabilities"].availability_reason == (
        "conflicting_standard_concepts"
    )


@pytest.mark.asyncio
async def test_newer_clean_liability_restatement_supersedes_older_conflict():
    duration = {
        "start": "2024-01-01", "end": "2024-12-31", "val": 100,
        "accn": "revenue", "fy": 2024, "fp": "FY", "form": "10-K",
        "filed": "2025-03-01",
    }
    old = {
        "start": None, "end": "2024-12-31", "accn": "old", "fy": 2024,
        "fp": "FY", "form": "10-K", "filed": "2025-02-01",
    }
    new = {**old, "accn": "new", "filed": "2025-03-01"}
    payload = {"entityName": "Restated Corp", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [duration]}},
        "LiabilitiesAndStockholdersEquity": {"units": {"USD": [
            {**old, "val": 500}, {**old, "val": 550}, {**new, "val": 600},
        ]}},
        "StockholdersEquity": {"units": {"USD": [
            {**old, "val": 200}, {**new, "val": 250},
        ]}},
    }}}
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "REST", "title": "Restated Corp"}
    }
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        result = await client.get_fundamentals("REST", as_of=date(2026, 10, 14))

    liabilities = result.latest["liabilities"]
    assert liabilities.value == 350
    assert liabilities.availability_reason is None
    assert liabilities.provenance is not None
    assert liabilities.provenance.accession == "new"


@pytest.mark.parametrize(
    ("ticker", "fixture_name"),
    [
        ("AMZN", "companyfacts_amzn.json"),
        ("JPM", "companyfacts_jpm.json"),
    ],
)
@pytest.mark.asyncio
async def test_latest_values_match_independent_filing_literals(ticker, fixture_name):
    expected = load_json("expected_fundamentals.json")["companies"][ticker]

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_transport(fixture_name),
    ) as client:
        result = await client.get_fundamentals(
            ticker, as_of=date.fromisoformat("2026-10-14")
        )

    for metric, literal in expected["values"].items():
        assert result.latest[metric].value == literal["value"]


@pytest.mark.asyncio
async def test_jpm_does_not_apply_industrial_free_cash_flow_semantics():
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_transport("companyfacts_jpm.json"),
    ) as client:
        result = await client.get_fundamentals(
            "JPM", as_of=date.fromisoformat("2026-10-14")
        )

    normalized = normalize_sec_fundamentals(result)
    assert result.latest["capex_payments"].value is None
    assert normalized["free_cash_flow"] is None
    assert normalized["free_cash_flow_availability_reason"] == (
        "industrial_free_cash_flow_not_applicable_to_financial_institution"
    )
    assert normalized["operating_cash_flow_comparability"] == (
        "not_comparable_to_industrial_companies"
    )
    assert normalized["roic"] is None
    assert normalized["margin_trend"] is None
    assert normalized["interest_coverage"] is None
    assert normalized["debt_growth_3y"] is None


@pytest.mark.asyncio
async def test_financial_institution_markers_fail_closed_when_revenue_is_generic():
    payload = load_json("companyfacts_jpm.json")
    us_gaap = payload["facts"]["us-gaap"]
    us_gaap["Revenues"] = us_gaap.pop("RevenuesNetOfInterestExpense")
    us_gaap["Deposits"] = {"units": {}}
    ticker_map = {"0": {"cik_str": 19617, "ticker": "JPM", "title": "JPM"}}

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        result = await client.get_fundamentals(
            "JPM", as_of=date.fromisoformat("2026-10-14")
        )

    normalized = normalize_sec_fundamentals(result)
    assert normalized["free_cash_flow"] is None
    assert normalized["free_cash_flow_availability_reason"] == (
        "industrial_free_cash_flow_not_applicable_to_financial_institution"
    )
    assert normalized["operating_cash_flow_comparability"] == (
        "not_comparable_to_industrial_companies"
    )
    assert normalized["roic"] is None
    assert normalized["margin_trend"] is None
    assert normalized["interest_coverage"] is None
    assert normalized["debt_growth_3y"] is None


@pytest.mark.asyncio
async def test_dividend_streak_uses_completed_period_year_not_filing_fiscal_year():
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_transport("companyfacts_amzn.json"),
    ) as client:
        result = await client.get_fundamentals(
            "AMZN", as_of=date.fromisoformat("2026-10-14")
        )

    base = result.latest["revenue"].provenance
    assert base is not None
    result.history["dividend_per_share"] = [
        FundamentalValue(
            value=0.0 if year == 2011 else 1.0,
            provenance=replace(
                base,
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                fiscal_year=2025,
            ),
        )
        for year in range(2011, 2026)
    ]

    assert normalize_sec_fundamentals(result)["consecutive_dividend_years"] == 14


@pytest.mark.parametrize("status_code", [403, 429])
@pytest.mark.asyncio
async def test_sec_rate_limit_is_structured_and_retryable(status_code):
    ticker_map = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    }

    async def handler(request):
        if request.url.path.endswith("/files/company_tickers.json"):
            return httpx.Response(200, json=ticker_map)
        return httpx.Response(status_code, json={"provider_detail": "do not expose"})

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(SECRateLimitError) as exc_info:
            await client.get_fundamentals("AAPL")

    assert exc_info.value.retryable is True
    assert str(exc_info.value) == "SEC data is temporarily unavailable."
    assert "provider_detail" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_ticker_absent_from_sec_company_map_is_a_distinct_unknown_issuer():
    async def handler(request):
        if request.url.path.endswith("/files/company_tickers.json"):
            return httpx.Response(200, json={})
        raise AssertionError("Company Facts must not be requested for an unknown issuer")

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(SECIssuerNotFoundError) as exc_info:
            await client.get_fundamentals("NOPE")

    assert exc_info.value.retryable is False
    assert str(exc_info.value) == "Ticker is not present in the SEC company map."


@pytest.mark.asyncio
async def test_sec_transport_failure_is_sanitized_without_exception_chain():
    secret = "private-sec-transport-detail"

    async def handler(request):
        raise httpx.ConnectError(secret, request=request)

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(SECUnavailableError) as exc_info:
            await client.get_fundamentals("AAPL")

    formatted = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert exc_info.value.retryable is True
    assert str(exc_info.value) == "SEC data is temporarily unavailable."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in formatted


@pytest.mark.asyncio
async def test_sec_server_error_is_retryable_unavailable():
    ticker_map = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    }

    async def handler(request):
        if request.url.path.endswith("/files/company_tickers.json"):
            return httpx.Response(200, json=ticker_map)
        return httpx.Response(503, json={"provider_detail": "private"})

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(SECUnavailableError) as exc_info:
            await client.get_fundamentals("AAPL")

    assert exc_info.value.retryable is True
    assert str(exc_info.value) == "SEC data is temporarily unavailable."


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (b"not-json private-provider-body", "application/json"),
        (b"[]", "application/json"),
    ],
)
@pytest.mark.asyncio
async def test_malformed_sec_ticker_map_is_a_sanitized_invalid_response(
    content, content_type
):
    async def handler(request):
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": content_type},
        )

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(SECInvalidResponseError) as exc_info:
            await client.get_fundamentals("AAPL")

    formatted = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert exc_info.value.retryable is False
    assert str(exc_info.value) == "SEC returned an invalid response."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "private-provider-body" not in formatted


@pytest.mark.asyncio
async def test_malformed_nested_ticker_map_is_invalid_not_unknown_issuer():
    async def handler(request):
        return httpx.Response(200, json={"0": ["private nested body"]})

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(SECInvalidResponseError) as exc_info:
            await client.get_fundamentals("AAPL")

    assert str(exc_info.value) == "SEC returned an invalid response."
    assert "private nested body" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_company_facts_without_facts_object_is_invalid_response():
    ticker_map = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    }

    async def handler(request):
        if request.url.path.endswith("/files/company_tickers.json"):
            return httpx.Response(200, json=ticker_map)
        return httpx.Response(200, json={"entityName": "Unexpected shape"})

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(SECInvalidResponseError) as exc_info:
            await client.get_fundamentals("AAPL")

    assert str(exc_info.value) == "SEC returned an invalid response."


@pytest.mark.asyncio
async def test_empty_facts_object_is_an_invalid_unusable_response():
    ticker_map = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    }
    payload = {"entityName": "Apple Inc.", "facts": {}}
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError):
            await client.get_fundamentals("AAPL")


@pytest.mark.asyncio
async def test_conflict_only_revenue_payload_is_invalid_without_usable_reference():
    annual = {
        "start": "2024-01-01", "end": "2024-12-31",
        "accn": "0000000000-25-000001", "fy": 2024, "fp": "FY",
        "form": "10-K", "filed": "2025-02-01",
    }
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "BAD", "title": "Bad Corp"}
    }
    payload = {"entityName": "Bad Corp", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {**annual, "val": 100}, {**annual, "val": 120},
        ]}},
        "Assets": {"units": {"USD": [{
            **annual, "start": None, "val": 500,
        }]}},
    }}}
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError):
            await client.get_fundamentals("BAD", as_of=date(2026, 10, 14))


@pytest.mark.asyncio
async def test_assets_only_payload_is_invalid_without_revenue_reference():
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "BAD", "title": "Bad Corp"}
    }
    payload = {"entityName": "Bad Corp", "facts": {"us-gaap": {
        "Assets": {"units": {"USD": [{
            "end": "2024-12-31", "val": 500,
            "accn": "0000000000-25-000001", "fy": 2024, "fp": "FY",
            "form": "10-K", "filed": "2025-02-01",
        }]}},
    }}}
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError):
            await client.get_fundamentals("BAD", as_of=date(2026, 10, 14))


@pytest.mark.asyncio
async def test_non_numeric_sec_fact_value_is_invalid():
    ticker_map = {
        "0": {"cik_str": 1, "ticker": "BAD", "title": "Bad Corp"}
    }
    payload = {"entityName": "Bad Corp", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [{
            "start": "2024-01-01", "end": "2024-12-31",
            "val": "100", "accn": "0000000000-25-000001",
            "fy": 2024, "fp": "FY", "form": "10-K",
            "filed": "2025-02-01",
        }]}},
    }}}
    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(ticker_map, payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError):
            await client.get_fundamentals("BAD", as_of=date(2026, 10, 14))


@pytest.mark.parametrize("mutation", ["nested_shape", "invalid_date"])
@pytest.mark.asyncio
async def test_malformed_nested_company_facts_are_sanitized(mutation):
    payload = load_json("companyfacts_aapl.json")
    revenue = payload["facts"]["us-gaap"][
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    ]
    if mutation == "nested_shape":
        revenue["units"] = []
    else:
        revenue["units"]["USD"][0]["filed"] = "not-a-date private-body"

    async with SECFundamentalsClient(
        user_agent="GOATlens tests tests@example.com",
        transport=sec_payload_transport(load_json("company_tickers.json"), payload),
    ) as client:
        with pytest.raises(SECInvalidResponseError) as exc_info:
            await client.get_fundamentals("AAPL")

    assert exc_info.value.retryable is False
    assert str(exc_info.value) == "SEC returned an invalid response."
    assert exc_info.value.__cause__ is None
    assert "private-body" not in str(exc_info.value)
