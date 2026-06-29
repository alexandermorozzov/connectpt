"""Unified experiment runner: one code path for every experiment.

An experiment is fully described by a YAML spec -- data source, method (a
captured bee-mix config), the sweep (alpha grid + adjustment target +
iterations) and which metrics to report. ``run_experiment`` loads the data,
sweeps the grid (NO Python alpha loop in the notebook -- the grid is YAML), runs
the method per point, scores the selected metrics and returns a structured
result the report layer renders.

This replaces the per-experiment cells that each rebuilt configs in Python and
looped over alpha by hand (EKB / E1 / E2 / M0 / MACSA).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from omegaconf import OmegaConf

from .context import CFG_DIR
from .data_sources import create_data_source
from .experiments import load_experiment_cfg
from .paper import UNIFIED_ADJ, bco_cfg_set, full_metrics, set_cfg_value


@dataclass
class ExperimentResult:
    name: str
    table: pd.DataFrame
    routes: dict
    instance: Any
    spec: Any = None
    meta: dict = field(default_factory=dict)


def load_experiment_spec(name):
    """Load an experiment spec YAML from cfg/experiments/<name>.yaml."""
    return OmegaConf.load(CFG_DIR / "experiments" / f"{name}.yaml")


def _alpha_weights(alpha):
    return {"route_time_weight": float(alpha),
            "median_connectivity_weight": float(1.0 - alpha)}


def _default_metrics(metrics_obj, routes, init_routes, *, keep):
    row = full_metrics(metrics_obj, routes, init_routes)
    return {k: row[k] for k in keep if k in row}


def run_experiment(spec, *, method_fn: Callable = None,
                   metrics_fn: Callable = None) -> ExperimentResult:
    """Run a full experiment from its spec.

    ``method_fn`` defaults to eval_lib.run_bco; ``metrics_fn`` defaults to the
    paper full-metric set filtered to ``spec.metrics``. Both are injectable so
    the orchestration (sweep + overrides) can be unit-tested without running BCO.
    """
    if method_fn is None:
        from .helpers import run_bco as method_fn  # noqa: PLW0127
    if metrics_fn is None:
        keep = list(spec.get("metrics", []))
        def metrics_fn(m, routes, init):  # noqa: E306
            return _default_metrics(m, routes, init, keep=keep)

    inst = create_data_source(spec.data).load()

    # one or many methods to compare (each a captured config + label). A single
    # ``method`` block is treated as a one-element list.
    methods = spec.get("methods")
    if methods is None:
        methods = [OmegaConf.create({"label": spec.get("method_label", "method"),
                                     **dict(spec.method)})]

    sweep = spec.sweep
    alphas = list(sweep.get("alpha", [None]))
    adj_target = float(sweep.adj_target)
    n_iterations = int(sweep.n_iterations)
    # adjustment kwargs default to the unified penalty; sweep.adj_weight can
    # override the weight (e.g. 0 for the adj-off Pareto-front experiment).
    adj_kwargs = dict(UNIFIED_ADJ, adjustment_degree_target=adj_target)
    if sweep.get("adj_weight") is not None:
        adj_kwargs["adjustment_degree_weight"] = float(sweep.adj_weight)

    rows, routes = [], {"Initial": inst.init_routes}
    for method in methods:
        base_cfg = load_experiment_cfg(method.config)
        if method.get("force_cpu") is not None:
            set_cfg_value(base_cfg, "experiment.cpu", bool(method.force_cpu))
        if method.get("process_neural_bees_sequentially") is not None:
            set_cfg_value(base_cfg, "process_neural_bees_sequentially",
                          bool(method.process_neural_bees_sequentially))
        label = method.get("label", "method")

        for alpha in alphas:
            cfg = copy.deepcopy(base_cfg)
            if alpha is not None:
                for key, value in _alpha_weights(alpha).items():
                    set_cfg_value(cfg, f"experiment.cost_function.kwargs.{key}", value)
            bco_cfg_set(cfg, n_iterations=n_iterations, **adj_kwargs)

            out = method_fn(cfg, inst.init_routes, tensors=inst.tensors,
                            run_name_scope=f"{inst.label}_{label}_alpha{alpha}_")
            _run_name, m, _unserved, out_routes, *_ = out

            row = dict(metrics_fn(m, out_routes, inst.init_routes))
            row.update(method=label, alpha=alpha, adj_target=adj_target,
                       n_iterations=n_iterations)
            rows.append(row)
            routes[f"{label} alpha={alpha}"] = out_routes

    return ExperimentResult(
        name=spec.name, table=pd.DataFrame(rows), routes=routes, instance=inst,
        spec=spec, meta={"label": inst.label},
    )
