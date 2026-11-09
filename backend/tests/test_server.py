import json

import pytest
from fastapi.testclient import TestClient

from conftest import company
from lens.build import analyze


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "snapshots").mkdir()
    snapshot = analyze(company(ticker="BRK-B", company_name="Berkshire Hathaway"))
    (tmp_path / "snapshots" / "BRK-B.json").write_text(json.dumps(snapshot))
    (tmp_path / "index.json").write_text(json.dumps([{"ticker": "BRK-B", "name": "Berkshire", "complete": 5}]))
    (tmp_path / "coverage.json").write_text(json.dumps({"covered": 1, "all_five_rate": 1.0}))
    import main
    monkeypatch.setattr(main, "SNAPSHOT_DIR", str(tmp_path))
    main._cache.clear()
    return TestClient(main.app)


def test_home_page_has_matching_script_hash(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "'sha256-" in response.headers["content-security-policy"]


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"
    assert client.head("/ping").status_code == 200


def test_analyze_accepts_dot_tickers_and_filters_agents(client):
    body = client.post("/api/analyze", json={"ticker": "brk.b", "agents": ["buffett", "graham"]}).json()
    assert body["ticker"] == "BRK-B"
    assert [r["id"] for r in body["agent_results"]] == ["buffett", "graham"]
    assert body["requested_analyses"] == 2
    assert "prices" not in body


def test_uncovered_ticker_gets_a_helpful_404(client):
    response = client.post("/api/analyze", json={"ticker": "ZZZZ"})
    assert response.status_code == 404
    assert "isn't covered yet" in response.json()["detail"]


def test_bad_ticker_rejected(client):
    assert client.post("/api/analyze", json={"ticker": "not a ticker!"}).status_code == 422


def test_price_history_and_earnings(client):
    prices = client.get("/api/price-history/BRK-B?period=1y").json()
    assert prices["prices"][-1]["close"] == 101
    assert prices["earnings_markers"][0]["beat_miss"] == "beat"
    earnings = client.get("/api/earnings/BRK-B").json()
    assert earnings["earnings_streak"]["beats"] == 1


def test_agents_and_coverage(client):
    assert len(client.get("/api/agents").json()["agents"]) == 5
    assert client.get("/api/coverage").json()["covered"] == 1
    assert client.get("/api/tickers").json()[0]["ticker"] == "BRK-B"


def test_funds_get_a_clear_message(client):
    response = client.post("/api/analyze", json={"ticker": "SPY"})
    assert response.status_code == 404 and "individual companies" in response.json()["detail"]


def test_company_name_suggests_ticker(client):
    detail = client.post("/api/analyze", json={"ticker": "BERKSHIRE"}).json()["detail"]
    assert "BRK-B" in detail


def test_head_root_and_bad_period(client):
    assert client.head("/").status_code == 200
    assert client.get("/api/price-history/BRK-B?period=zzz").status_code == 422
