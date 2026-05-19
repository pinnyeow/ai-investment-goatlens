"""Standalone tests for recent_news feature. Run with: python3 tests/test_recent_news.py"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # backend/

from unittest.mock import AsyncMock
from data_sources.yahoo import YahooFinanceClient
from agents.buffett import BuffettAgent

MOCK_FINANCIALS = {
    "roe": 0.28,
    "profit_margin": 0.25,
    "debt_to_equity": 0.45,
    "free_cash_flow": 100_000_000_000,
    "insider_ownership": 0.02,
}

VALID_VERDICTS = {"strong_buy", "buy", "hold", "sell", "strong_sell"}


async def test_news_injected_into_llm_prompt():
    captured = []
    mock_llm = AsyncMock()
    mock_llm.analyze = AsyncMock(
        side_effect=lambda prompt, **kw: captured.append(prompt) or "test insight"
    )

    mock_news = [
        {
            "headline": "Apple Q4 beats estimates",
            "source": "Reuters",
            "published": "2026-05-18",
            "summary": "Strong iPhone demand.",
        }
    ]
    agent = BuffettAgent(llm_client=mock_llm)
    await agent.analyze("AAPL", MOCK_FINANCIALS, recent_news=mock_news)

    assert captured, "LLM was not called"
    assert "Apple Q4 beats estimates" in captured[0], "Headline missing from LLM prompt"
    assert "Reuters" in captured[0], "Source missing from LLM prompt"
    print("✓ News headline and source appear in LLM prompt")


async def test_no_news_graceful():
    agent = BuffettAgent(llm_client=None)
    for news_arg in (None, []):
        result = await agent.analyze("AAPL", MOCK_FINANCIALS, recent_news=news_arg)
        assert result["verdict"] in VALID_VERDICTS, (
            f"Unexpected verdict '{result['verdict']}' for recent_news={news_arg!r}"
        )
    print("✓ None and [] recent_news both degrade gracefully")


async def test_get_recent_news():
    async with YahooFinanceClient() as client:
        news = await client.get_recent_news("AAPL")

    assert isinstance(news, list), f"Expected list, got {type(news)}"
    assert len(news) <= 10, f"Expected at most 10 items, got {len(news)}"

    if news:
        item = news[0]
        for field in ("headline", "source", "published", "summary"):
            assert field in item, f"Missing field '{field}' in news item"
        assert item["published"] != "", "published date should not be empty"

    print(f"✓ get_recent_news: {len(news)} items returned")
    if news:
        print(f"  First: {news[0]['headline'][:70]}...")


async def main():
    print("Testing recent_news feature...\n")
    await test_news_injected_into_llm_prompt()
    await test_no_news_graceful()
    await test_get_recent_news()  # live network call last
    print("\n✓ All tests passed")


if __name__ == "__main__":
    asyncio.run(main())
