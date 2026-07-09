"""Score a route set into the full paper metric row.

The unified objective computes normalized RTT / WMC / cost components during the
search; this module turns the engine's raw metrics dict (plus the route tensors)
into the full metric row every results table reports -- including the derived
metrics the engine does not emit (adjustment degree vs the seed network, edge
redundancy). One implementation: ``eval_lib`` re-exports these names so the
notebook / frozen path use the same code.

Adjustment-degree gap/mode are read from the single objective source
(``cfg/objective/*.yaml``) via :func:`load_unified_objective`, matching what the
search acceptance and PPO reward use.
"""
from __future__ import annotations

import numpy as np
import torch

from ..bee_colony import get_adjustment_degrees
from ..data.route_copies import redundancy_fraction
from ..data.routes import as_route_tensor
from ..objectives import load_unified_objective

_OBJ = load_unified_objective()
_ADJ_GAP = _OBJ.adj_gap
_ADJ_MODE = _OBJ.adj_mode


def metric_value(metrics, key, default=np.nan):
    """Read one metric as a plain float (mean over the batch), or ``default``."""
    if key not in metrics or metrics[key] is None:
        return default
    value = metrics[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 0:
            return default
        return float(value.float().mean().item())
    return float(value)


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
        r[:, :nr, :w], s[:, :nr, :w], True, gap=_ADJ_GAP, mode=_ADJ_MODE).mean().item())


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


def full_metric_row(m, rt, seed):
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


def select_metrics(row: dict, keep) -> dict:
    """Filter a metric row to the ``keep`` keys (preserving present ones)."""
    keep = list(keep)
    return {k: row[k] for k in keep if k in row}


def score_fixed_routes(routes, tensors, spec, *, alpha=None, adj_target=None,
                       adj_objective=None, seed_routes=None, device=None):
    """Score a FIXED route set under the unified objective -- no search at all.

    The atomic evaluation every "compare against an external network" experiment
    uses (MACSA Table-B, the pinned-Initial parity check): build the unified cost
    (``CostFactory.build_unified``), evaluate ``routes`` on the instance
    ``tensors`` with the ``spec`` route bounds, and return the engine metrics
    (with the cost breakdown attached) plus the scored route tensor.

    ``alpha`` sets the RTT/WMC trade-off (``route_time_weight=alpha``,
    ``median_connectivity_weight=1-alpha``). The adjustment penalty is applied
    only when both ``adj_target`` and ``seed_routes`` are given (weight/gap/mode
    come from the objective YAML; ``adj_objective`` optionally overrides the
    two-sided default) -- otherwise it is off, matching how the paper scores
    Initial rows. Runs on CPU unless ``device`` says otherwise.
    """
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader

    from ..citygraph_dataset import get_dataset_from_config
    from ..objectives import CostFactory
    from ..utils import test_method
    from .cost_breakdown import add_cost_breakdown_to_metrics

    device = torch.device("cpu") if device is None else device
    cost_obj = CostFactory.build_unified("rtt_wmc_no_demand", for_training=False)
    if alpha is not None:
        cost_obj.route_time_weight = float(alpha)
        cost_obj.median_connectivity_weight = float(1.0 - float(alpha))
    if adj_target is not None and seed_routes is not None:
        cost_obj.adjustment_degree_target = float(adj_target)
        if adj_objective is not None:
            cost_obj.adjustment_degree_objective = str(adj_objective)
        seed = as_route_tensor(seed_routes)
        cost_obj.adjustment_seed = (seed[None] if seed.dim() == 2 else seed).to(device)
    else:
        cost_obj.adjustment_degree_weight = 0.0
    cost_obj.to(device)

    dataloader = DataLoader(
        get_dataset_from_config(OmegaConf.create({"type": "tensor"}), tensors=tensors),
        batch_size=1)
    eval_cfg = OmegaConf.create({"csv": False, **dict(spec)})
    out = test_method(
        None, dataloader, eval_cfg, OmegaConf.create({"method": "tensor"}),
        cost_obj, silent=True, device=device, return_routes=True,
        routes_tensor=as_route_tensor(routes))
    _mean, _std, _unserved, metrics, scored = out
    scored = as_route_tensor(scored)
    metrics = add_cost_breakdown_to_metrics(
        metrics, dataloader, eval_cfg, cost_obj, scored, device)
    return metrics, scored
