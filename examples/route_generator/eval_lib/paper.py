"""paper_combined.ipynb support: unified-objective helpers + paper_results IO.

Everything here is generic plumbing extracted from the notebook's helper cell:
config mutation utilities, the unified-objective adjustment kwargs, the full
metric row computed for every results table, and the ``paper_results`` output
sink. Experiment design (ablation variants, per-city budgets, init pipeline)
stays in the notebook.
"""
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from connectpt.routes_generator.bee_colony import get_adjustment_degrees

from connectpt.routes_generator.objectives import load_unified_objective

from .context import ARTIFACTS_DIR
from .helpers import as_route_tensor, metric_value
from .baselines import build_sa_cfg
from .route_copies import redundancy_fraction

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


def set_cfg_value(cfg, dotted_key, value):
    """Set a nested OmegaConf value even when the composed cfg is structured."""
    target = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        OmegaConf.set_struct(target, False)
        target = target[part]
    OmegaConf.set_struct(target, False)
    target[parts[-1]] = value
    return cfg


def bco_cfg_set(cfg, **kv):
    """Set fields BCO reads from cfg (n_iterations, adjustment_degree_*)."""
    OmegaConf.set_struct(cfg, False)
    for k, v in kv.items():
        cfg[k] = v
    return cfg


def unify_weights(cfg):
    """Set a baseline cfg's cost weights to the unified objective (RTT+WMC, demand off)."""
    for key, value in UNIFIED_COST_WEIGHTS.items():
        set_cfg_value(cfg, f"experiment.cost_function.kwargs.{key}", value)
    set_cfg_value(cfg, "experiment.cost_function.kwargs.connectivity_mode",
                  CONNECTIVITY_MODE)
    return cfg


def eval_routes_cfg(city, spec):
    """1-iteration SA cfg used only to evaluate a fixed route set's metrics."""
    return build_sa_cfg(f"{city}_eval", spec["n_routes"],
                        spec["min_route_len"], spec["max_route_len"],
                        n_iterations=1, connectivity_mode=CONNECTIVITY_MODE)


def ravel_hist(h):
    """Flatten a per-sample cost history to a plain list of floats (or [])."""
    if h is None:
        return []
    arr = np.asarray(h.detach().cpu() if hasattr(h, "detach") else h).ravel()
    return [float(v) for v in arr]


# --- metrics ----------------------------------------------------------------

def adj_vs_init(routes, init_routes):
    """Mean adjustment degree of a route set vs the init network."""
    r = as_route_tensor(routes)
    s = as_route_tensor(init_routes)
    if r.ndim == 2:
        r = r[None]
    if s.ndim == 2:
        s = s[None]
    nr = min(r.shape[1], s.shape[1])
    w = min(r.shape[-1], s.shape[-1])
    return float(get_adjustment_degrees(
        r[:, :nr, :w], s[:, :nr, :w], True, gap=ADJ_GAP, mode=ADJ_MODE).mean().item())


def redundancy_pct(routes):
    """% of edge traversals that re-cover an already-covered edge."""
    R = as_route_tensor(routes)
    if R.ndim == 3:
        R = R[0]
    return 100.0 * redundancy_fraction(R)


def conn_metric(m):
    """The optimized weighted connectivity (falls back to the plain median)."""
    v = metric_value(m, "median_connectivity_weighted")
    return v if np.isfinite(v) else metric_value(m, "median_connectivity")


def full_metrics(m, rt, seed):
    """Full metric set for every results row: we optimize a subset (RTT+conn+adj)
    but always compute/report/save ALL of them."""
    rt = as_route_tensor(rt)
    return {
        "ATT": metric_value(m, "ATT"), "RTT": metric_value(m, "RTT"),
        # WMC = the optimized weighted connectivity (median by default). Both
        # demand-weighted variants are reported: WMC_mean and WMC_median.
        "WMC": conn_metric(m),
        "WMC_mean": metric_value(m, "WMC_mean"),
        "WMC_median": metric_value(m, "WMC_median"),
        "adj_vs_seed": adj_vs_init(rt, seed),
        # normalized cost components -- the RTT / WMC the algorithm actually
        # optimizes (route_cost and weighted_median_connectivity, /time_normalizer).
        "rtt_cost": metric_value(m, "cost_route_component"),
        "wmc_cost": metric_value(m, "cost_connectivity_component"),
        "cost": metric_value(m, "cost"),
        "d0": metric_value(m, "$d_0$"), "d1": metric_value(m, "$d_1$"),
        "d2": metric_value(m, "$d_2$"), "d_un": metric_value(m, "$d_{un}$"),
        "redun%": redundancy_pct(rt),
    }


def paper_row(city, method, source, m, rt, seed, duration_s=None):
    """One results-table row: identity columns + the full metric set."""
    return {"city": city, "method": method, "source": source,
            "duration_s": (round(float(duration_s), 1)
                           if duration_s is not None else None),
            **full_metrics(m, rt, seed)}


def macsa_eval_bounds(scenario):
    """Route-count / length bounds for evaluating a MACSA scenario."""
    n_nodes = int(scenario["tensors"]["node_locs"].shape[0])
    n_routes = int(scenario["routes"].shape[1])
    seed_route_lens = (scenario["routes"] > -1).sum(dim=-1)
    longest_seed_route = (int(seed_route_lens.max().item())
                          if seed_route_lens.numel() else MIN_ROUTE_LEN)
    max_route_len = min(n_nodes, max(MAX_ROUTE_LEN, longest_seed_route))
    return n_routes, MIN_ROUTE_LEN, max_route_len


# --- paper_results output sink ----------------------------------------------
#
# Every sink takes the output-filename prefix EXPLICITLY (keyword-only,
# required). The caller threads it from ``RunContext.output_prefix`` ("TEMP_"
# on smoke runs, "" on full runs) -- there is no module-global prefix anymore.

PAPER_DIR = ARTIFACTS_DIR / "paper_results"
PAPER_DIR.mkdir(parents=True, exist_ok=True)


def paper_path(name, *, prefix):
    """Prefix-aware path under paper_results (e.g. to read back a saved dump).
    Mirrors what save_paper_* write, so reads find TEMP_ files on smoke runs."""
    return PAPER_DIR / f"{prefix}{name}"


def save_paper_table(df, name, *, prefix):
    path = PAPER_DIR / f"{prefix}{name}.csv"
    df.to_csv(path, index=False)
    print(f"[paper] table ({len(df)} rows) -> {path}")
    return path


def reset_paper_table(name, *, prefix):
    path = PAPER_DIR / f"{prefix}{name}.csv"
    if path.exists():
        path.unlink()
    return path


def append_paper_row(row, name, ndigits=3, *, prefix):
    path = PAPER_DIR / f"{prefix}{name}.csv"
    header = not path.exists()
    pd.DataFrame([row]).round(ndigits).to_csv(path, mode="a", header=header,
                                              index=False)
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
    path = PAPER_DIR / f"{prefix}{name}_routes.pt"
    torch.save(payload, path)
    print(f"[paper] route dump ({len(payload['routes'])} sets) -> {path}")
    return path
