.PHONY: lint test check

lint:
	cd backend && source venv/bin/activate && ruff check .

test:
	cd backend && source venv/bin/activate && python -m pytest tests/ -v

check: lint test
