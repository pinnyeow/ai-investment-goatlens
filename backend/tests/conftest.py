"""Shared isolation for optional external providers."""

import pytest


@pytest.fixture(autouse=True)
def isolate_yahoo_ratio_network(monkeypatch):
    """Tests opt into ratio fixtures instead of reaching Yahoo over the network."""
    import main
    from data_sources.yahoo_ratios import YahooRatioTemporarilyUnavailable

    class UnavailableRatios:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def get_snapshot(self, ticker):
            raise YahooRatioTemporarilyUnavailable(
                "Normalized ratios are unavailable in tests."
            )

    monkeypatch.setattr(main, "YahooRatiosClient", UnavailableRatios, raising=False)
