"""Allow running evals as: python -m backend.evals.run_evals"""
from backend.evals.run_evals import main
import asyncio

asyncio.run(main())
