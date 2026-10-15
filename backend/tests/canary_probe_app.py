"""Real-server fixture for canary exception-log sanitization."""

import main

SENTINEL = "credential=PRIVATE-CANARY-SECRET internal-host.example"


class ExplodingSEC:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def get_fundamentals(self, ticker, as_of=None):
        raise RuntimeError(SENTINEL)


main.SECFundamentalsClient = ExplodingSEC
main._fundamentals_health_cache.clear()
main._fundamentals_health_inflight.clear()
app = main.app
