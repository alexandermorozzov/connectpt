"""Unified seed-sweep wrapper over :func:`run_method`.

`run_seed_sweep` is the single seed-sweep entry point shared by every
experiment section (worse-accept LC sweep, NX-heuristic sweep, MACSA seeded
comparison, ...). It is a thin loop over graph x method x accept-mode x seed
that calls :func:`run_method` and accumulates the uniform :class:`RunResult`
objects, then derives:

* ``rows_df``  -- one row per individual run (via ``run_result_row``);
* ``summary_df`` -- seed-averaged mean/std grouped by dataset/method/accept;
* ``results``  -- the raw ``RunResult`` list (carries ``mutation_stats`` /
  ``action_stats`` for the mutation histograms and route figures).

It replaces the three former bespoke sweeps (``run_worse_accept_seed_sweep``,
``run_nx_dataset_bco_seed_sweep``, ``run_macsa_seeded_comparison``).
"""
import pandas as pd

from .params import *  # noqa: F401,F403  (N_ROUTES, MIN/MAX_ROUTE_LEN ...)
from .run import run_method
from .tables import run_result_row, _METRIC_COLUMNS
from . import plots as _plots


def _summarize_seed_sweep(rows_df: pd.DataFrame) -> pd.DataFrame:
    """Seed-average ``rows_df`` -> mean/std per dataset/method/accept_mode."""
    if rows_df.empty:
        return rows_df
    group_cols = [c for c in ("dataset", "method", "kind", "accept_mode")
                  if c in rows_df.columns]
    metric_cols = _plots.filter_component_columns(
        [c for c in _METRIC_COLUMNS if c in rows_df.columns],
        ENABLED_COST_COMPONENTS)
    grouped = rows_df.groupby(group_cols, sort=False)
    stats = grouped[metric_cols].agg(["mean", "std"])
    stats.columns = [f"{stat}_{col}" for col, stat in stats.columns]
    summary = stats
    if "seed" in rows_df.columns:
        summary = summary.join(grouped["seed"].nunique().rename("n_seeds"))
    return summary.reset_index()


def run_seed_sweep(graph_specs, method_specs, seeds,
                   accept_modes=("worse", "without_worse"),
                   run_name_scope="", progress=True) -> dict:
    """Run ``run_method`` across graphs x methods x accept-modes x seeds.

    ``graph_specs`` is a list of dicts describing each initial-data set::

        {"dataset": "LC",            # label carried into every row
         "init_routes": <tensor>,    # required -- the seed route set
         "tensors": <dict | None>,   # None -> Mumford0 dataloader
         "n_routes": N_ROUTES,       # optional eval bounds overrides
         "min_route_len": ..., "max_route_len": ...,
         "weights": {...}}           # optional cost-weight override

    ``method_specs`` is a list of method dicts for :func:`run_method` (build
    them with :func:`bco_method`, :data:`RL_ONLY_METHOD`,
    :data:`INITIAL_METHOD`). ``accept_modes`` only fans out the BCO methods;
    ``initial`` / ``rl_only`` runs are deterministic so they run once per graph
    with ``accept_mode='n/a'`` and the first seed.

    Returns ``{"rows_df", "summary_df", "results"}``.
    """
    seeds = list(seeds) or [None]
    rows, results = [], []

    plan = []
    for graph_index, gspec in enumerate(graph_specs):
        for method in method_specs:
            if method["kind"] == "bco":
                method_accepts, method_seeds = list(accept_modes), seeds
            else:
                method_accepts, method_seeds = ["n/a"], [seeds[0]]
            for accept_mode in method_accepts:
                for seed in method_seeds:
                    plan.append((graph_index, gspec, method, accept_mode, seed))

    total = len(plan)
    for run_idx, (graph_index, gspec, method, accept_mode, seed) in \
            enumerate(plan, 1):
        dataset = gspec.get("dataset", "")
        if progress:
            print(f"[seed-sweep {run_idx}/{total}] dataset={dataset} "
                  f"method={method.get('label', method['kind'])} "
                  f"accept={accept_mode} seed={seed}")
        result = run_method(
            method,
            init_routes=gspec["init_routes"],
            n_routes=gspec.get("n_routes", N_ROUTES),
            min_route_len=gspec.get("min_route_len", MIN_ROUTE_LEN),
            max_route_len=gspec.get("max_route_len", MAX_ROUTE_LEN),
            tensors=gspec.get("tensors"),
            weights=gspec.get("weights"),
            accept_mode=(accept_mode if accept_mode != "n/a"
                         else "without_worse"),
            seed=seed,
            dataset=dataset,
            run_name_scope=run_name_scope,
        )
        results.append(result)
        row = run_result_row(result)
        row["graph_index"] = graph_index
        row["seed"] = seed
        rows.append(row)

    rows_df = pd.DataFrame(rows)
    return {
        "rows_df": rows_df,
        "summary_df": _summarize_seed_sweep(rows_df),
        "results": results,
    }
