"""Evaluators -- run a model over a dataset and produce an EvaluationResult.

The edit evaluator delegates the actual rollout to the existing
``evaluate_lc_improvement`` (so behaviour matches the trainer's validation), then
shapes the per-instance + summary tables. No plotting here.
"""
from __future__ import annotations

import pandas as pd

from .result_schema import EvaluationResult


class EditModelEvaluator:
    def evaluate(self, model, cost_obj, data_module, indices, *,
                 min_route_len, max_route_len, batch_size=8,
                 target_n_routes=None, metadata=None) -> EvaluationResult:
        from ..improvement_learning import evaluate_lc_improvement

        result = evaluate_lc_improvement(
            model, cost_obj, data_module.graphs, data_module.seed_routes, indices,
            next(model.parameters()).device, min_route_len, max_route_len,
            batch_size=batch_size, target_n_routes=target_n_routes,
        )
        # summary: the scalar metrics evaluate_lc_improvement reports
        scalar = {k: v for k, v in result.items()
                  if isinstance(v, (int, float))}
        summary = pd.DataFrame([scalar])
        per_instance = pd.DataFrame({"instance": list(range(len(indices)))})
        return EvaluationResult(per_instance=per_instance, summary=summary,
                                metadata=metadata or {})
