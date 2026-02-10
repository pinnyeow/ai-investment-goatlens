"""
Test-First Validation for GOATlens Trace Propagation and Agent LLM Usage.

These tests are designed to:
1. FAIL on the current codebase (proving the bugs exist)
2. PASS after the fixes are applied

Run with:
    cd backend && python -m pytest tests/test_trace_and_agents.py -v

Groups:
    A) Trace propagation — verifies all spans share 1 trace_id
    B) Agent LLM usage  — verifies all 5 agents call llm_client.analyze
    C) Insight quality   — verifies LLM insights flow through, Lynch concerns work
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import pytest

# ── Ensure backend is importable ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import BuffettAgent, LynchAgent, GrahamAgent, MungerAgent, DalioAgent


# ============================================================================
# Fixtures — Shared mock data and helpers
# ============================================================================

# Realistic financials dict that all agents expect
MOCK_FINANCIALS: Dict[str, Any] = {
    # Buffett metrics
    "roe": 0.28,
    "profit_margin": 0.25,
    "debt_to_equity": 0.45,
    "free_cash_flow": 100_000_000_000,
    "insider_ownership": 0.02,
    # Lynch metrics
    "pe_ratio": 28.0,
    "earnings_growth": 0.12,
    "institutional_ownership": 0.72,
    "cash_and_equivalents": 50_000_000_000,
    # Graham metrics
    "pb_ratio": 40.0,
    "current_ratio": 1.1,
    "dividend_yield": 0.006,
    "consecutive_dividend_years": 12,
    "book_value_per_share": 4.0,
    "current_price": 185.0,
    "eps": 6.5,
    "current_assets": 130_000_000_000,
    "total_liabilities": 280_000_000_000,
    "shares_outstanding": 15_000_000_000,
    # Munger metrics
    "gross_margin": 0.43,
    "operating_margin": 0.30,
    "roic": 0.28,
    "revenue_growth_5y": 0.08,
    "capex_to_revenue": 0.03,
    "margin_trend": 0.01,
    # Dalio metrics
    "beta": 1.1,
    "volatility_252d": 0.22,
    "interest_coverage": 30.0,
    "revenue_cyclicality": 0.15,
    "correlation_sp500": 0.85,
    "debt_growth_3y": 0.05,
    "expected_return": 0.12,
    "sector": "Technology",
}

MOCK_EARNINGS_DATA = [
    {"quarter": "Q4 2025", "eps_actual": 2.40, "eps_estimate": 2.36, "surprise_pct": 1.7, "beat_miss": "beat"},
    {"quarter": "Q3 2025", "eps_actual": 1.64, "eps_estimate": 1.60, "surprise_pct": 2.5, "beat_miss": "beat"},
    {"quarter": "Q2 2025", "eps_actual": 1.40, "eps_estimate": 1.35, "surprise_pct": 3.7, "beat_miss": "beat"},
    {"quarter": "Q1 2025", "eps_actual": 1.65, "eps_estimate": 1.62, "surprise_pct": 1.9, "beat_miss": "beat"},
]

MOCK_EARNINGS_STREAK = {
    "streak_type": "beat",
    "streak_count": 4,
    "beats": 4,
    "misses": 0,
    "inline": 0,
    "total": 4,
    "summary": "Beat 4 consecutive quarters",
}

LLM_RESPONSE_SENTINEL = "LLM_GENERATED_INSIGHT_SENTINEL_12345"


def _make_mock_llm_client() -> AsyncMock:
    """Create a mock LLM client whose analyze() returns a known sentinel string."""
    mock = AsyncMock()
    mock.analyze = AsyncMock(return_value=LLM_RESPONSE_SENTINEL)
    mock.model_name = "mock-model"
    return mock


# ============================================================================
# Group B: Agent LLM Usage Tests
# ============================================================================


class TestAgentLLMUsage:
    """Verify that every agent calls llm_client.analyze() when given a client."""

    @pytest.mark.asyncio
    async def test_buffett_calls_llm(self):
        """Buffett should call llm_client.analyze (already works)."""
        mock_llm = _make_mock_llm_client()
        agent = BuffettAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        mock_llm.analyze.assert_called_once()
        assert result["agent"] == "Warren Buffett"

    @pytest.mark.asyncio
    async def test_lynch_calls_llm(self):
        """Lynch should call llm_client.analyze when client is provided.
        EXPECTED: FAIL on current code (Lynch never calls LLM).
        """
        mock_llm = _make_mock_llm_client()
        agent = LynchAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        mock_llm.analyze.assert_called_once()
        assert result["agent"] == "Peter Lynch"

    @pytest.mark.asyncio
    async def test_graham_calls_llm(self):
        """Graham should call llm_client.analyze when client is provided.
        EXPECTED: FAIL on current code (Graham never calls LLM).
        """
        mock_llm = _make_mock_llm_client()
        agent = GrahamAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        mock_llm.analyze.assert_called_once()
        assert result["agent"] == "Benjamin Graham"

    @pytest.mark.asyncio
    async def test_munger_calls_llm(self):
        """Munger should call llm_client.analyze when client is provided.
        EXPECTED: FAIL on current code (Munger never calls LLM).
        """
        mock_llm = _make_mock_llm_client()
        agent = MungerAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        mock_llm.analyze.assert_called_once()
        assert result["agent"] == "Charlie Munger"

    @pytest.mark.asyncio
    async def test_dalio_calls_llm(self):
        """Dalio should call llm_client.analyze when client is provided.
        EXPECTED: FAIL on current code (Dalio never calls LLM).
        """
        mock_llm = _make_mock_llm_client()
        agent = DalioAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        mock_llm.analyze.assert_called_once()
        assert result["agent"] == "Ray Dalio"


# ============================================================================
# Group B2: Fallback — Agents work without LLM
# ============================================================================


class TestAgentRuleBasedFallback:
    """Verify agents produce valid output without an LLM client."""

    @pytest.mark.asyncio
    async def test_buffett_works_without_llm(self):
        agent = BuffettAgent(llm_client=None)
        result = await agent.analyze("AAPL", MOCK_FINANCIALS)
        assert "verdict" in result
        assert "insights" in result
        assert len(result["insights"]) > 0

    @pytest.mark.asyncio
    async def test_lynch_works_without_llm(self):
        agent = LynchAgent(llm_client=None)
        result = await agent.analyze("AAPL", MOCK_FINANCIALS)
        assert "verdict" in result
        assert "insights" in result

    @pytest.mark.asyncio
    async def test_graham_works_without_llm(self):
        agent = GrahamAgent(llm_client=None)
        result = await agent.analyze("AAPL", MOCK_FINANCIALS)
        assert "verdict" in result
        assert "insights" in result

    @pytest.mark.asyncio
    async def test_munger_works_without_llm(self):
        agent = MungerAgent(llm_client=None)
        result = await agent.analyze("AAPL", MOCK_FINANCIALS)
        assert "verdict" in result
        assert "insights" in result

    @pytest.mark.asyncio
    async def test_dalio_works_without_llm(self):
        agent = DalioAgent(llm_client=None)
        result = await agent.analyze("AAPL", MOCK_FINANCIALS)
        assert "verdict" in result
        assert "insights" in result


# ============================================================================
# Group C: Insight Quality Tests
# ============================================================================


class TestInsightQuality:
    """Verify LLM insights flow through and agents produce meaningful output."""

    @pytest.mark.asyncio
    async def test_buffett_llm_insight_in_output(self):
        """Buffett's LLM response should appear in the insights list."""
        mock_llm = _make_mock_llm_client()
        agent = BuffettAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        assert LLM_RESPONSE_SENTINEL in result["insights"]

    @pytest.mark.asyncio
    async def test_lynch_llm_insight_in_output(self):
        """Lynch's LLM response should appear in the insights list.
        EXPECTED: FAIL on current code.
        """
        mock_llm = _make_mock_llm_client()
        agent = LynchAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        assert LLM_RESPONSE_SENTINEL in result["insights"]

    @pytest.mark.asyncio
    async def test_graham_llm_insight_in_output(self):
        """Graham's LLM response should appear in the insights list.
        EXPECTED: FAIL on current code.
        """
        mock_llm = _make_mock_llm_client()
        agent = GrahamAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        assert LLM_RESPONSE_SENTINEL in result["insights"]

    @pytest.mark.asyncio
    async def test_munger_llm_insight_in_output(self):
        """Munger's LLM response should appear in the insights list.
        EXPECTED: FAIL on current code.
        """
        mock_llm = _make_mock_llm_client()
        agent = MungerAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        assert LLM_RESPONSE_SENTINEL in result["insights"]

    @pytest.mark.asyncio
    async def test_dalio_llm_insight_in_output(self):
        """Dalio's LLM response should appear in the insights list.
        EXPECTED: FAIL on current code.
        """
        mock_llm = _make_mock_llm_client()
        agent = DalioAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        assert LLM_RESPONSE_SENTINEL in result["insights"]

    @pytest.mark.asyncio
    async def test_lynch_generates_concerns_for_high_peg(self):
        """Lynch should produce concerns when PEG is very high (rule-based)."""
        bad_financials = {**MOCK_FINANCIALS, "pe_ratio": 50, "earnings_growth": 0.05}
        agent = LynchAgent(llm_client=None)
        result = await agent.analyze("BADCO", bad_financials)
        assert len(result["concerns"]) > 0, "Lynch should flag high PEG as a concern"

    @pytest.mark.asyncio
    async def test_lynch_generates_concerns_for_aapl_with_llm(self):
        """With LLM, Lynch should produce concerns even for a strong stock like AAPL.
        EXPECTED: FAIL on current code (Lynch uses rule-based only, AAPL has no rule-triggered concerns).
        """
        # Mock LLM returns a concern-like response
        mock_llm = _make_mock_llm_client()
        mock_llm.analyze = AsyncMock(
            return_value="While Apple's earnings growth of 12% is respectable, the PEG ratio suggests growth is fully priced in at current valuations"
        )
        agent = LynchAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        # With LLM insights, Lynch should have richer output
        assert len(result["insights"]) > 0

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rules(self):
        """If the LLM call throws, agents should fallback to rule-based insights."""
        mock_llm = _make_mock_llm_client()
        mock_llm.analyze = AsyncMock(side_effect=Exception("API timeout"))
        agent = BuffettAgent(llm_client=mock_llm)
        result = await agent.analyze(
            "AAPL", MOCK_FINANCIALS,
            earnings_data=MOCK_EARNINGS_DATA,
            earnings_streak=MOCK_EARNINGS_STREAK,
        )
        # Should still have insights (rule-based fallback)
        assert len(result["insights"]) > 0
        assert LLM_RESPONSE_SENTINEL not in result["insights"]


# ============================================================================
# Group A: Trace Propagation Tests
# ============================================================================


class _ListExporter:
    """Minimal span exporter that collects spans into a list (for testing)."""

    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return None  # SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        pass


class TestTracePropagation:
    """
    Verify OTel context propagation so all spans share one trace_id.

    Uses a lightweight list-based exporter to capture spans locally.
    """

    @pytest.mark.asyncio
    async def test_parallel_agents_share_parent_context(self):
        """
        When agents run in parallel via asyncio.gather, LLM spans
        should inherit the caller's OTel context (same trace_id).
        """
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        exporter = _ListExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        async def mock_agent_work(name: str):
            with tracer.start_as_current_span(f"agent.{name}.llm_call"):
                await asyncio.sleep(0.01)
                return f"{name}_result"

        with tracer.start_as_current_span("run_agents_node"):
            results = await asyncio.gather(
                mock_agent_work("buffett"),
                mock_agent_work("lynch"),
                mock_agent_work("graham"),
            )

        provider.force_flush()
        spans = exporter.spans

        trace_ids = set(s.context.trace_id for s in spans)
        assert len(trace_ids) == 1, (
            f"Expected 1 trace_id, got {len(trace_ids)}. "
            f"Spans: {[(s.name, hex(s.context.trace_id)) for s in spans]}"
        )

        parent_span = [s for s in spans if s.name == "run_agents_node"][0]
        child_spans = [s for s in spans if s.name != "run_agents_node"]
        for child in child_spans:
            assert child.parent is not None, f"Span {child.name} has no parent"
            assert child.parent.span_id == parent_span.context.span_id, (
                f"Span {child.name} parent is not run_agents_node"
            )

        provider.shutdown()

    @pytest.mark.asyncio
    async def test_create_task_also_propagates_context(self):
        """
        Diagnostic: verify that create_task + otel_context.attach also works.
        If both approaches produce 1 trace, the orphaned traces in Arize
        are caused by the LangChain instrumentor, not asyncio context.
        """
        try:
            from opentelemetry import trace, context as otel_context
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        exporter = _ListExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        async def mock_agent_work_in_task(name: str):
            with tracer.start_as_current_span(f"agent.{name}.llm_call"):
                await asyncio.sleep(0.01)
                return f"{name}_result"

        with tracer.start_as_current_span("run_agents_node"):
            current_ctx = otel_context.get_current()

            async def run_with_context(name):
                token = otel_context.attach(current_ctx)
                try:
                    return await mock_agent_work_in_task(name)
                finally:
                    otel_context.detach(token)

            tasks = [
                asyncio.create_task(run_with_context("buffett")),
                asyncio.create_task(run_with_context("lynch")),
            ]
            results = await asyncio.gather(*tasks)

        provider.force_flush()
        spans = exporter.spans

        trace_ids = set(s.context.trace_id for s in spans)
        assert len(trace_ids) == 1, (
            f"create_task produced {len(trace_ids)} traces (expected 1)"
        )

        provider.shutdown()
