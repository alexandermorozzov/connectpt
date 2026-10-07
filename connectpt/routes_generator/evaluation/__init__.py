"""Evaluation application: compute structured metrics (no plotting).

Evaluation *computes* numbers and saves structured artifacts; turning those into
tables/figures is the reports layer's job. Depends on core/ models/ objectives/;
must not import training/ or search/ internals (it consumes checkpoints).
"""

from .route_scoring import (adj_vs_init, conn_metric, full_metric_row,
                           initial_row, metric_value, redundancy_pct,
                           score_fixed_routes, select_metrics)


__all__ = [
    "adj_vs_init",
    "conn_metric",
    "full_metric_row",
    "initial_row",
    "metric_value",
    "redundancy_pct",
    "score_fixed_routes",
    "select_metrics",
]
