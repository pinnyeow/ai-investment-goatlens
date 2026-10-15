"""Repository-level architecture constraints for external data providers."""

import base64
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

import main


ROOT = Path(__file__).parents[2]


def test_fmp_is_absent_from_runtime_configuration_and_documentation():
    assert not (ROOT / "backend" / "data_sources" / "fmp.py").exists()
    skipped_parts = {".git", "venv", ".pytest_cache", "__pycache__"}
    text_suffixes = {".py", ".md", ".html", ".yaml", ".yml", ".toml"}
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path == Path(__file__)
            or path.suffix not in text_suffixes
            or skipped_parts.intersection(path.relative_to(ROOT).parts)
        ):
            continue
        content = path.read_text(errors="replace").lower()
        assert "fmp" not in content, path
        assert "financial modeling prep" not in content, path


def test_release_smoke_checks_sec_analysis_and_reports_market_status_once():
    workflow = (
        ROOT / ".github" / "workflows" / "smoke-tests.yml"
    ).read_text()

    assert workflow.count("/api/fundamentals-health") == 1
    assert workflow.count("/api/market-data-health") == 1
    assert workflow.count("/api/analyze") == 1
    assert '"source":"sec"' in workflow
    assert '"fundamentals":{"available":true,"source":"sec"' in workflow
    assert '"agents":["buffett","lynch","graham","munger","dalio"]' in workflow
    assert "assert body['completed_analyses'] == 5" in workflow
    assert "assert body['requested_analyses'] == 5" in workflow
    assert "assert body['agreement_score'] is not None" in workflow
    assert "assert all(item['status'] == 'complete'" in workflow
    assert 'RELEASE=$(python3 -c' in workflow
    assert '[[ "$RELEASE" == "$EXPECTED_SHA" ]]' in workflow
    assert '[[ "$BODY" == *' not in workflow


def test_no_unproven_synthetic_demo_claims_remain():
    backend = (ROOT / "backend" / "main.py").read_text()
    frontend = (ROOT / "frontend" / "index.html").read_text()
    workflow = (ROOT / ".github" / "workflows" / "smoke-tests.yml").read_text()
    assert "/api/demo" not in backend
    assert "/api/demo" not in frontend
    assert "Demo API returns valid data" not in workflow


def test_core_runtime_does_not_call_broad_yfinance_adapter():
    backend = (ROOT / "backend" / "main.py").read_text()
    assert "YahooFinanceClient" not in backend
    assert "YahooFinanceError" not in backend


def test_removed_broad_yfinance_provider_has_no_runtime_footprint():
    requirements = (ROOT / "backend" / "requirements.txt").read_text().lower()
    assert "yfinance" not in requirements
    assert "lxml" not in requirements
    assert not (ROOT / "backend" / "data_sources" / "yahoo.py").exists()


def test_frontend_third_party_scripts_are_integrity_pinned_and_csp_limited():
    frontend = (ROOT / "frontend" / "index.html").read_text()
    assert 'http-equiv="Content-Security-Policy"' in frontend
    assert "object-src 'none'" in frontend
    assert "base-uri 'none'" in frontend
    assert "script-src 'self' 'sha256-" in frontend
    script_policy = frontend.split("script-src", 1)[1].split(";", 1)[0]
    assert "'unsafe-inline'" not in script_policy
    assert "onclick=" not in frontend
    assert 'data-action="open-scoring"' in frontend
    assert "document.addEventListener('click'" in frontend
    script_lines = [
        line for line in frontend.splitlines()
        if "<script" in line and "https://" in line
    ]
    assert len(script_lines) == 2
    assert all('integrity="sha384-' in line for line in script_lines)
    assert all('crossorigin="anonymous"' in line for line in script_lines)


def test_csp_authorizes_the_exact_current_inline_application_script():
    frontend = (ROOT / "frontend" / "index.html").read_text()
    inline_scripts = [
        body
        for body in re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", frontend, re.S)
        if body.strip()
    ]
    assert len(inline_scripts) == 1
    digest = base64.b64encode(
        hashlib.sha256(inline_scripts[0].encode()).digest()
    ).decode()
    expected = f"'sha256-{digest}'"

    assert expected in frontend
    with TestClient(main.app) as client:
        response = client.get("/")
    assert expected in response.headers["Content-Security-Policy"]


def test_public_models_reject_extra_fields_and_numeric_coercion():
    with pytest.raises(ValidationError):
        main.MetricEvidenceResponse.model_validate(
            {
                "value": 1.0,
                "available": True,
                "source": "sec",
                "as_of": "2026-10-14",
                "formula": "reported_fact",
                "formula_version": "sec_v1",
                "secret": "do-not-ignore",
            }
        )
    with pytest.raises(ValidationError):
        main.MetricEvidenceResponse.model_validate(
            {
                "value": 1.0,
                "available": True,
                "source": "sec",
                "as_of": "2026-10-14",
                "formula": "reported_fact",
                "formula_version": "sec_v1",
                "inputs": [],
            }
        )
    with pytest.raises(ValidationError):
        main.MetricEvidenceResponse.model_validate(
            {
                "value": 1.0,
                "available": True,
                "source": "sec",
                "as_of": "2026-10-14",
                "formula": "reported_fact",
                "formula_version": "sec_v1",
                "inputs": ["revenue_history"],
            }
        )
    with pytest.raises(ValidationError):
        main.MetricEvidenceResponse.model_validate(
            {
                "value": 1.0,
                "available": False,
                "source": "sec",
                "as_of": None,
                "formula": "reported_fact",
                "formula_version": "sec_v1",
                "inputs": [],
                "availability_reason": "required_inputs_unavailable",
            }
        )
    with pytest.raises(ValidationError):
        main.MetricEvidenceResponse.model_validate(
            {
                "value": 0.2,
                "available": True,
                "source": "yahoo",
                "as_of": "2026-10-14",
                "formula": "sample_volatility",
                "formula_version": "yahoo_chart_v1",
                "inputs": [
                    {
                        "kind": "market_series",
                        "role": "asset.adjusted_close",
                        "source": "yahoo",
                        "start": "2025-10-14",
                        "end": "2026-10-14",
                        "observation_count": 252,
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        main.PricePoint.model_validate(
            {
                "date": "2026-10-14",
                "open": "1.0",
                "high": 2.0,
                "low": 0.5,
                "close": True,
                "volume": 1,
                "junk": "do-not-ignore",
            }
        )


def test_every_strategy_required_metric_has_one_public_evidence_contract():
    expected = {
        "buffett": ("roe", "profit_margin", "debt_to_equity", "free_cash_flow"),
        "lynch": (
            "pe_ratio",
            "earnings_growth",
            "debt_to_equity",
        ),
        "graham": (
            "pe_ratio",
            "pb_ratio",
            "current_ratio",
            "current_price",
            "current_assets",
            "total_liabilities",
            "shares_outstanding",
            "eps",
            "book_value_per_share",
            "dividend_yield",
            "consecutive_dividend_years",
        ),
        "munger": (
            "gross_margin",
            "operating_margin",
            "roic",
            "revenue_growth_5y",
            "capex_to_revenue",
            "debt_to_equity",
            "margin_trend",
        ),
        "dalio": (
            "beta",
            "volatility_252d",
            "debt_to_equity",
            "correlation_sp500",
            "debt_growth_3y",
            "realized_total_return_252d",
        ),
    }

    assert main._AGENT_REQUIRED_METRICS == expected
    required = {metric for metrics in expected.values() for metric in metrics}
    assert set(main._METRIC_EVIDENCE_CONTRACTS) == required | {
        "cash_and_equivalents"
    }


def test_aligned_market_metrics_use_the_aligned_window_end_as_as_of():
    normalized = {
        "ticker": "AAPL",
        "beta": 1.2,
        "correlation_sp500": 0.8,
        "volatility_252d": 0.2,
        "realized_total_return_252d": 0.1,
        "current_price": 200.0,
    }
    snapshot = SimpleNamespace(
        current_price_as_of="2026-10-14T20:00:00+00:00",
        metric_session_start="2025-10-27",
        metric_session_end="2026-10-14",
        aligned_metric_session_start="2025-10-27",
        aligned_metric_session_end="2026-10-13",
        asset_return_observations=252,
        aligned_return_observations=251,
    )

    evidence = main._build_metric_evidence(normalized, snapshot)

    assert evidence["beta"]["as_of"] == "2026-10-13"
    assert evidence["correlation_sp500"]["as_of"] == "2026-10-13"
    assert evidence["volatility_252d"]["as_of"] == "2026-10-14"


def test_analysis_request_rejects_unknown_fields():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze",
            json={"ticker": "AAPL", "agents": ["buffett"], "admin": True},
        )
    assert response.status_code == 422


def test_no_public_tracing_debug_or_credential_metadata_logging():
    backend = (ROOT / "backend" / "main.py").read_text()
    assert "/api/debug/tracing" not in backend
    assert "Space ID:" not in backend
    assert "API Key:" not in backend

    runtime_root = ROOT / "backend"
    for path in runtime_root.rglob("*.py"):
        relative_parts = path.relative_to(runtime_root).parts
        if relative_parts[0] in {"tests", "evals", "venv"}:
            continue
        content = path.read_text(errors="replace")
        assert "traceback.print_exc()" not in content, path
        for line in content.splitlines():
            if "print(" in line or "logger." in line:
                assert "{e}" not in line, path
                assert "{error}" not in line, path
                assert "str(e)" not in line, path
                assert "str(error)" not in line, path
