.PHONY: lint test check dev

lint:
	cd backend && ruff check .

test:
	cd backend && python -m pytest -q tests

check: lint test

# Build snapshots for a few tickers and serve them locally at http://localhost:8000
dev:
	cd backend && python -m lens.build --out ../.snapshots --tickers AAPL,KO,JPM,WMT,BRK-B && \
	SNAPSHOT_DIR=../.snapshots uvicorn main:app --reload
