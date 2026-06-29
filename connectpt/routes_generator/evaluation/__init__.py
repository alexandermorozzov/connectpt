"""Evaluation application: compute structured metrics (no plotting).

Evaluation *computes* numbers and saves structured artifacts; turning those into
tables/figures is the reports layer's job. Depends on core/ models/ objectives/;
must not import training/ or search/ internals (it consumes checkpoints).
"""

from .result_schema import EvaluationResult
from .metrics import MetricComputer
from .evaluators import EditModelEvaluator
from .runs import ModelEvaluationRun


__all__ = [
    "EvaluationResult",
    "MetricComputer",
    "EditModelEvaluator",
    "ModelEvaluationRun",
]
