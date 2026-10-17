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


def _provider_isolation_paths(root):
    text_suffixes = {".py", ".md", ".html", ".yaml", ".yml", ".toml"}
    skipped_parts = {"venv", ".pytest_cache", "__pycache__"}
    candidates = [path for path in root.iterdir() if path.is_file()]
    for shipped_root_name in ("backend", "frontend", "docs", ".github"):
        shipped_root = root / shipped_root_name
        if shipped_root.exists():
            candidates.extend(shipped_root.rglob("*"))
    return [
        path
        for path in candidates
        if (
            path.is_file()
            and path.suffix in text_suffixes
            and path != Path(__file__)
            and not skipped_parts.intersection(path.relative_to(root).parts)
        )
    ]


def _assert_fmp_absent(root):
    for path in _provider_isolation_paths(root):
        content = path.read_text(errors="replace").lower()
        assert "fmp" not in content, path
        assert "financial modeling prep" not in content, path


def test_provider_scan_excludes_internal_plans_but_still_scans_shipped_docs(
    tmp_path,
):
    (tmp_path / ".hermes" / "plans").mkdir(parents=True)
    (tmp_path / ".hermes" / "plans" / "internal.md").write_text("FMP plan")
    (tmp_path / "docs").mkdir()
    shipped_doc = tmp_path / "docs" / "shipped.md"
    shipped_doc.write_text("provider-neutral documentation")

    _assert_fmp_absent(tmp_path)

    shipped_doc.write_text("Financial Modeling Prep")
    with pytest.raises(AssertionError, match="shipped.md"):
        _assert_fmp_absent(tmp_path)


def test_fmp_is_absent_from_runtime_configuration_and_documentation():
    assert not (ROOT / "backend" / "data_sources" / "fmp.py").exists()
    _assert_fmp_absent(ROOT)


def test_release_smoke_checks_aapl_amzn_and_reports_market_status_once():
    workflow = (
        ROOT / ".github" / "workflows" / "smoke-tests.yml"
    ).read_text()

    assert workflow.count("/api/fundamentals-health") == 1
    assert workflow.count("/api/market-data-health") == 1
    assert workflow.count("/api/analyze") == 2
    assert '"source":"sec"' in workflow
    assert '"fundamentals":{"available":true,"source":"sec"' in workflow
    assert '"ticker":"AAPL"' in workflow
    assert '"ticker":"AMZN"' in workflow
    assert "revenue_minus_cost_of_revenue_divided_by_revenue" in workflow
    assert "not evidence['dividend_yield']['available']" in workflow

    assert "evidence['earnings_turnaround']['value'] == 1.0" in workflow
    assert "evidence['earnings_growth']['value'] is None" in workflow
    assert '"agents":["buffett","lynch","graham","munger","dalio"]' in workflow
    assert "assert body['completed_analyses'] == 5" in workflow
    assert "assert body['requested_analyses'] == 5" in workflow
    assert "assert body['agreement_score'] is not None" in workflow
    assert "assert all(item['status'] == 'complete'" in workflow
    assert 'RELEASE=$(python3 -c' in workflow
    assert '[[ "$RELEASE" == "$EXPECTED_SHA" ]]' in workflow
    assert '[[ "$BODY" == *' not in workflow


def test_repository_requires_explicit_deployment_and_canary_gate():
    render = (ROOT / "render.yaml").read_text()
    workflow = (ROOT / ".github" / "workflows" / "smoke-tests.yml").read_text()

    assert "autoDeploy: false" in render
    assert "push:" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "EXPECTED_SHA: ${{ inputs.expected_sha }}" in workflow
    assert "EXPECTED_SHA: ${{ github.sha }}" not in workflow
    assert "if: ${{ github.event_name == 'workflow_dispatch' }}" in workflow


def test_frontend_plots_explicit_adjusted_close_history():
    frontend = (ROOT / "frontend" / "index.html").read_text()

    assert "value: p.adjusted_close" in frontend
    assert "value: p.close" not in frontend


def test_unreleased_adjusted_close_change_stays_in_0_2_0_migration():
    note = ROOT / "docs" / "migrations" / "0.2.0-ratio-contract-removal.md"
    superseding_note = ROOT / "docs" / "migrations" / "0.2.1-adjusted-price-history.md"

    content = note.read_text()
    assert "adjusted_close" in content
    assert "close" in content
    assert "additive" in content.lower()
    assert not superseding_note.exists()
    with TestClient(main.app) as client:
        schema = client.get("/openapi.json").json()
    assert schema["info"]["version"] == "0.2.0"


def test_frontend_defines_agreement_as_directional_support():
    frontend = (ROOT / "frontend" / "index.html").read_text()

    assert "Directional Support" in frontend
    assert "GOAT Agreement" not in frontend


def test_frontend_labels_sec_period_end_and_filing_date_separately():
    frontend = (ROOT / "frontend" / "index.html").read_text()

    assert "as of ${evidence.as_of}" in frontend
    assert "filed ${evidence.filed_at}" in frontend


def test_yahoo_unofficial_best_effort_risk_is_disclosed_in_docs_and_ui():
    docs = (ROOT / "docs" / "agent-data-contracts.md").read_text().lower()
    frontend = (ROOT / "frontend" / "index.html").read_text().lower()

    for phrase in ("unofficial", "best-effort", "no sla", "public-use risk"):
        assert phrase in docs
    assert "unofficial, best-effort source with no sla" in frontend


def test_no_unproven_synthetic_demo_claims_remain():
    backend = (ROOT / "backend" / "main.py").read_text()
    frontend = (ROOT / "frontend" / "index.html").read_text()
    workflow = (ROOT / ".github" / "workflows" / "smoke-tests.yml").read_text()
    assert "/api/demo" not in backend
    assert "/api/demo" not in frontend
    assert "Demo API returns valid data" not in workflow


def _assert_removed_ratio_surface_absent(root):
    forbidden = (
        "yfinance",
        "get_info",
        "yahooratiosclient",
        "provider_metric",
        "normalized_ratios",
        "quote-summary",
    )
    permitted = root / "docs" / "migrations" / "0.2.0-ratio-contract-removal.md"
    for path in _provider_isolation_paths(root):
        relative_parts = path.relative_to(root).parts
        if path == permitted or relative_parts[:2] == ("backend", "tests"):
            continue
        content = path.read_text(errors="replace").lower()
        for token in forbidden:
            assert token.lower() not in content, f"{token}: {path}"


def test_shipped_architecture_and_frontend_describe_the_narrow_yahoo_boundary():
    architecture = (ROOT / "ARCHITECTURE.md").read_text()
    frontend = (ROOT / "frontend" / "index.html").read_text()
    prd = (ROOT / "PRD.md").read_text()

    assert "backend/data_sources/yahoo_chart.py" in architecture
    assert "backend/data_sources/yahoo.py" not in architecture
    assert "market/earnings/analyst enrichment" not in architecture
    assert "earnings, and analyst context" not in architecture
    assert "earnings, and analyst context" not in prd
    assert "volume, earnings, and analyst estimates" not in prd
    assert "current price and adjusted price history" in prd
    assert "owner earnings" not in prd.lower()
    assert "free cash flow" in prd.lower()
    assert "EPS — always present (Yahoo)" not in frontend
    assert "EPS — render only when reported" in frontend


def test_runtime_uses_only_the_narrow_yahoo_chart_boundary():
    assert not (ROOT / "backend" / "data_sources" / "yahoo.py").exists()
    assert not (ROOT / "backend" / "data_sources" / "yahoo_ratios.py").exists()
    _assert_removed_ratio_surface_absent(ROOT)
    with TestClient(main.app) as client:
        openapi = client.get("/openapi.json").text.lower()
    assert "normalized_ratios" not in openapi
    assert "provider_metric" not in openapi
    for shipped_doc in (ROOT / "README.md", ROOT / "docs" / "agent-data-contracts.md"):
        content = shipped_doc.read_text().lower()
        assert "yfinance" not in content
        assert "provider-normalized" not in content


def test_versioned_breaking_change_ships_a_migration_note():
    note = ROOT / "docs" / "migrations" / "0.2.0-ratio-contract-removal.md"
    with TestClient(main.app) as client:
        schema = client.get("/openapi.json").json()

    assert schema["info"]["version"] == "0.2.0"
    content = note.read_text()
    assert "data_status.normalized_ratios" in content
    assert "provider_metric" in content
    assert "removed" in content.lower()


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


def test_every_consumed_strategy_metric_has_one_central_contract():
    expected = {
        "buffett": {
            "agent": "Warren Buffett",
            "mandatory": ("roe", "profit_margin", "debt_to_equity", "free_cash_flow"),
            "optional": (),
        },
        "lynch": {
            "agent": "Peter Lynch",
            "mandatory": ("pe_ratio", "earnings_growth", "debt_to_equity"),
            "optional": ("institutional_ownership", "cash_and_equivalents"),
        },
        "graham": {
            "agent": "Benjamin Graham",
            "mandatory": (
                "pe_ratio", "pb_ratio", "current_ratio", "current_price",
                "current_assets", "total_liabilities", "shares_outstanding", "eps",
                "book_value_per_share",
            ),
            "optional": ("dividend_yield", "consecutive_dividend_years"),
        },
        "munger": {
            "agent": "Charlie Munger",
            "mandatory": (
                "gross_margin", "operating_margin", "revenue_growth_5y",
                "capex_to_revenue", "debt_to_equity", "margin_trend",
            ),
            "optional": ("roic", "insider_ownership"),
        },
        "dalio": {
            "agent": "Ray Dalio",
            "mandatory": (
                "beta", "volatility_252d", "debt_to_equity",
                "correlation_sp500", "realized_total_return_252d",
            ),
            "optional": ("interest_coverage", "debt_growth_3y"),
        },
    }

    assert main._AGENT_CONSUMPTION_CONTRACTS == expected
    assert main._AGENT_REQUIRED_METRICS == {
        agent: contract["mandatory"] for agent, contract in expected.items()
    }
    evidence_backed = {
        metric
        for contract in expected.values()
        for metric in contract["mandatory"] + contract["optional"]
        if metric not in {"institutional_ownership", "insider_ownership"}
    }
    non_agent_public_metrics = {"earnings_turnaround", "market_cap", "ps_ratio"}
    assert set(main._METRIC_EVIDENCE_CONTRACTS) == (
        evidence_backed | non_agent_public_metrics
    )


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
