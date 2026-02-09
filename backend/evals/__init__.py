"""
GOATlens Evaluation Framework.

Three layers of evaluation to catch problems at every level:
  Layer 1: Data Pipeline — is the data clean?
  Layer 2: Agent Quality — are insights grounded, on-brand, and useful?
  Layer 3: Consensus     — is the synthesis coherent?

Usage:
    python -m backend.evals.run_evals           # all layers
    python -m backend.evals.run_evals --layer 1  # data only
    python -m backend.evals.run_evals --layer 2  # agents only
"""

from .data_pipeline_evals import (
    run_data_pipeline_evals,
    EvalResult,
    LayerResults,
)
from .agent_evals import run_agent_evals
from .consensus_evals import run_consensus_evals
from .golden_dataset import GOLDEN_TICKERS, PHILOSOPHY_KEYWORDS

__all__ = [
    "run_data_pipeline_evals",
    "run_agent_evals",
    "run_consensus_evals",
    "EvalResult",
    "LayerResults",
    "GOLDEN_TICKERS",
    "PHILOSOPHY_KEYWORDS",
]
