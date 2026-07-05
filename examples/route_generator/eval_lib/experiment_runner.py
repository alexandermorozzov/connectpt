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
from omegaconf import OmegaConf, ListConfig

from .context import CFG_DIR
from .data_sources import create_data_source
from .experiments import compose_experiment_cfg
from .paper import UNIFIED_ADJ, full_metrics


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


def _default_metrics(metrics_obj, routes, init_routes, *, keep):
    row = full_metrics(metrics_obj, routes, init_routes)
    return {k: row[k] for k in keep if k in row}


def run_experiment(spec, *, ctx=None, method_fn: Callable = None,
                   metrics_fn: Callable = None) -> ExperimentResult:
    """Run a full experiment from its spec.

    ``ctx`` (a :class:`eval_lib.run_context.RunContext`) supplies the edit-model
    checkpoint the default method uses; it is required unless an explicit
    ``method_fn`` is injected. ``method_fn`` defaults to the library-owned
    ``run_bco_from_cfg``; ``metrics_fn`` defaults to the paper full-metric set
    filtered to ``spec.metrics``. Both are injectable so the orchestration
    (sweep + overrides) can be unit-tested without running BCO.
    """
    if method_fn is None:
        if ctx is None:
            raise TypeError(
                "run_experiment: pass ctx=RunContext(...) so the default "
                "method knows which edit checkpoint to load (or inject an "
                "explicit method_fn)")
        from connectpt.routes_generator.search.cfg_run import run_bco_from_cfg
        _edit_path = ctx.edit_weights_path
        _edit_feats = int(ctx.edit_adj_cond_feats)

        def method_fn(cfg, init_routes, *, tensors, run_name_scope=""):
            return run_bco_from_cfg(
                cfg, init_routes, tensors, run_name_scope=run_name_scope,
                edit_weights_path=_edit_path,
                edit_n_adjustment_cond_feats=_edit_feats)
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
    # adj_target may be a scalar or a list (2D alpha x target Pareto sweep).
    _at = sweep.adj_target
    adj_targets = [float(t) for t in _at] if isinstance(_at, (list, ListConfig)) \
        else [float(_at)]
    n_iterations = int(sweep.n_iterations)
    adj_weight_override = (float(sweep.adj_weight)
                           if sweep.get("adj_weight") is not None else None)

    rows, routes = [], {"Initial": inst.init_routes}
    for method in methods:
        # route bounds come from the loaded instance (the data source knows the
        # right n_routes / lengths -- e.g. a MACSA scenario differs from any
        # benchmark city), so one captured bee-mix config works across sources.
        base_cfg = compose_experiment_cfg(
            method.config, bounds=dict(inst.spec),
            cpu=method.get("force_cpu"),
            seq_bees=method.get("process_neural_bees_sequentially"))
        label = method.get("label", "method")

        for alpha in alphas:
            for adj_target in adj_targets:
                adj_kwargs = dict(UNIFIED_ADJ, adjustment_degree_target=adj_target)
                if adj_weight_override is not None:
                    adj_kwargs["adjustment_degree_weight"] = adj_weight_override
                cfg = compose_experiment_cfg(
                    copy.deepcopy(base_cfg), alpha=alpha,
                    n_iterations=n_iterations, adj=adj_kwargs)

                out = method_fn(cfg, inst.init_routes, tensors=inst.tensors,
                                run_name_scope=f"{inst.label}_{label}_a{alpha}_t{adj_target}_")
                _run_name, m, _unserved, out_routes, *_ = out

                row = dict(metrics_fn(m, out_routes, inst.init_routes))
                row.update(method=label, alpha=alpha, adj_target=adj_target,
                           n_iterations=n_iterations)
                rows.append(row)
                routes[f"{label} a={alpha} t={adj_target}"] = out_routes

    return ExperimentResult(
        name=spec.name, table=pd.DataFrame(rows), routes=routes, instance=inst,
        spec=spec, meta={"label": inst.label},
    )
