"""paper_combined.ipynb support: unified-objective constants (thin).

The metric scoring and the ``paper_results`` output sinks now live in the
library (``reports.paper_io`` + ``evaluation``); this module re-exports those
names and keeps only the unified-objective constants the notebook reads.
"""
from connectpt.routes_generator.objectives import load_unified_objective
# Metric scoring -- library (single implementation).
from connectpt.routes_generator.evaluation import (  # noqa: F401
    adj_vs_init, conn_metric, full_metric_row as full_metrics, metric_value,
    redundancy_pct)
# paper_results sinks + paper_row + ravel_hist -- library reports layer.
from connectpt.routes_generator.reports.paper_io import (  # noqa: F401
    PAPER_DIR, append_paper_row, paper_path, paper_row, ravel_hist,
    reset_paper_table, save_paper_fig, save_paper_routes, save_paper_table)
# MACSA eval-bounds -- library (single implementation).
from connectpt.routes_generator.data.loaders import macsa_eval_bounds  # noqa: F401

# --- unified objective constants (read from cfg/objective/*.yaml) -----------
_OBJ = load_unified_objective()
CONNECTIVITY_MODE = _OBJ.connectivity_mode
UNIFIED_COST_WEIGHTS = _OBJ.weights
ADJ_GAP = _OBJ.adj_gap
ADJ_MODE = _OBJ.adj_mode
# Non-objective route-length defaults (were eval_lib.params knobs).
MIN_ROUTE_LEN = 2
MAX_ROUTE_LEN = 12
# Two-sided |adj - target| penalty kwargs shared by every unified-objective run.
UNIFIED_ADJ = dict(_OBJ.adj_kwargs)
