"""Repository-level architecture constraints for external data providers."""

from pathlib import Path


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
    assert '"completed_analyses":1' in workflow
    assert '"requested_analyses":1' in workflow
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
