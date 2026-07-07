"""paper_combined.ipynb support: unified-objective metrics + paper_results IO.

The unified-objective adjustment kwargs, the full metric row computed for
every results table, and the ``paper_results`` output sink. Config
composition (``set_cfg_value`` / ``compose_experiment_cfg`` / scoring cfgs)
lives in :mod:`eval_lib.experiments`; experiment design stays in YAML.
"""
import numpy as np

from connectpt.routes_generator.objectives import load_unified_objective
# Metric scoring lives in the library (single implementation); re-exported here
# so the notebook / frozen callers keep the same names.
from connectpt.routes_generator.evaluation import (  # noqa: F401
    adj_vs_init, conn_metric, full_metric_row as full_metrics, metric_value,
    redundancy_pct)

from .context import ARTIFACTS_DIR
from .helpers import as_route_tensor

# --- unified objective ------------------------------------------------------
# Read from the single source (cfg/objective YAML) via the library factory.
_OBJ = load_unified_objective()
CONNECTIVITY_MODE = _OBJ.connectivity_mode
UNIFIED_COST_WEIGHTS = _OBJ.weights
ADJ_GAP = _OBJ.adj_gap
ADJ_MODE = _OBJ.adj_mode
# Non-objective route-length defaults (were eval_lib.params knobs).
MIN_ROUTE_LEN = 2
MAX_ROUTE_LEN = 12

# Adjustment kwargs shared by every unified-objective run (E1u baselines,
# NSGA-II, MACSA): two-sided |adj - target| penalty. Sourced from the objective.
UNIFIED_ADJ = dict(_OBJ.adj_kwargs)


def ravel_hist(h):
    """Flatten a per-sample cost history to a plain list of floats (or [])."""
    if h is None:
        return []
    arr = np.asarray(h.detach().cpu() if hasattr(h, "detach") else h).ravel()
    return [float(v) for v in arr]


# --- metrics ----------------------------------------------------------------
# adj_vs_init / redundancy_pct / conn_metric / full_metrics (== full_metric_row)
# / metric_value are imported from the library above -- one implementation.


def paper_row(city, method, source, m, rt, seed, duration_s=None):
    """One results-table row: identity columns + the full metric set."""
    return {"city": city, "method": method, "source": source,
            "duration_s": (round(float(duration_s), 1)
                           if duration_s is not None else None),
            **full_metrics(m, rt, seed)}


# MACSA eval-bounds moved to the library -- single implementation.
from connectpt.routes_generator.data.loaders import macsa_eval_bounds  # noqa: F401


# --- paper_results output sink ----------------------------------------------
#
# Every sink takes the output-filename prefix EXPLICITLY (keyword-only,
# required). The caller threads it from ``RunContext.output_prefix`` ("TEMP_"
# on smoke runs, "" on full runs) -- there is no module-global prefix anymore.

# The tabular / route-dump IO is the library ArtifactStore (single sink); these
# stay as thin prefix-aware wrappers over it (the notebook keeps the names).
from connectpt.routes_generator.core import ArtifactStore

PAPER_DIR = ARTIFACTS_DIR / "paper_results"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
_STORE = ArtifactStore(PAPER_DIR)


def paper_path(name, *, prefix):
    """Prefix-aware path under paper_results (e.g. to read back a saved dump).
    Mirrors what save_paper_* write, so reads find TEMP_ files on smoke runs."""
    return PAPER_DIR / f"{prefix}{name}"


def save_paper_table(df, name, *, prefix):
    path = _STORE.save_table(df, f"{prefix}{name}")
    print(f"[paper] table ({len(df)} rows) -> {path}")
    return path


def reset_paper_table(name, *, prefix):
    path = PAPER_DIR / f"{prefix}{name}.csv"
    if path.exists():
        path.unlink()
    return path


def append_paper_row(row, name, ndigits=3, *, prefix):
    path = _STORE.append_row(row, f"{prefix}{name}", ndigits=ndigits)
    print(f"[paper] row -> {path}", flush=True)
    return path


def save_paper_fig(fig, name):
    # Figures are intentionally NOT persisted as images. The underlying data is
    # saved instead (CSV tables + route .pt dumps) so any figure can be rebuilt.
    print(f"[paper] figure '{name}' shown inline (rebuild from CSV / route dump)")
    return None


def save_paper_routes(name, routes, coords=None, street_adj=None, meta=None, *,
                      prefix):
    """Dump a {label: route_tensor} mapping (+ coords/street_adj) to paper_results
    so the route figures can be reconstructed later."""
    payload = {
        "routes": {k: as_route_tensor(v).cpu() for k, v in routes.items()},
        "coords": (coords.cpu() if hasattr(coords, "cpu") else coords),
        "street_adj": (street_adj.cpu() if hasattr(street_adj, "cpu") else street_adj),
        "meta": meta or {},
    }
    path = _STORE.save_routes(payload, f"{prefix}{name}_routes")
    print(f"[paper] route dump ({len(payload['routes'])} sets) -> {path}")
    return path
