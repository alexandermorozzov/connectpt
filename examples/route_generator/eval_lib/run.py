"""Unified validation entry point: ``run_method`` + ``RunResult``.

One function -- :func:`run_method` -- produces a uniform :class:`RunResult` for
any method (the initial route set / RL-improvement-only / a BCO variant) on any
data set (LC, NX-heuristic, MACSA, Mumford/Mandl), with or without worse-accept.
:func:`run_seed_sweep` (eval_lib.sweep), :func:`build_comparison_table`
(eval_lib.tables) and the route figures are all built on top of it.

``run_method`` is a thin dispatcher -- it wraps the existing runners
``run_bco`` / ``run_rl_improvement`` / ``_run_baseline`` and packages their
output into the uniform ``RunResult``; the runners themselves are unchanged.
"""
from dataclasses import dataclass, field

from .params import *  # noqa: F401,F403  (N_ROUTES, BCO_WORSE_* ...)
from .helpers import (build_bco_cfg, run_bco, run_rl_improvement,
                      as_route_tensor, mutation_stat_bucket,
                      selection_stat_bucket)
from .baselines import build_sa_cfg, _run_baseline


@dataclass
class RunResult:
    """Uniform result of one method run, consumed by every table / figure."""
    label: str
    kind: str                       # "initial" | "rl_only" | "bco"
    accept_mode: str = "n/a"        # "worse" | "without_worse" | "n/a"
    dataset: str = ""               # "LC" | "NX" | "MACSA" | benchmark city
    metrics: dict = field(default_factory=dict)
    routes: object = None           # final route tensor
    seed_routes: object = None      # initial route tensor
    unserved_demand: object = None
    mutation_stats: dict = field(default_factory=dict)  # bco only
    action_stats: dict = field(default_factory=dict)    # rl only
    weights: dict = field(default_factory=dict)
    run_name: str = ""


# "ON" annealing temperatures for the worse-accept mode (mirror of
# eval_lib.sweeps.WORSE_ACCEPT_EXPERIMENTS; kept here to avoid an import cycle
# since eval_lib.sweep imports this module).
WORSE_ACCEPT_ON_TEMPERATURE = 1
WORSE_SELECTION_ON_TEMPERATURE = 1

# accept-mode -> worse-accept knobs passed straight to build_bco_cfg.
ACCEPT_MODES = {
    "without_worse": dict(
        worse_accept_temperature=0.0,
        worse_accept_decay=BCO_WORSE_ACCEPT_DECAY,
        worse_accept_min_temperature=BCO_WORSE_ACCEPT_MIN_TEMPERATURE,
        worse_selection_temperature=0.0,
        worse_selection_decay=BCO_WORSE_SELECTION_DECAY,
        worse_selection_min_temperature=BCO_WORSE_SELECTION_MIN_TEMPERATURE,
        worse_selection_uniform_mix=BCO_WORSE_SELECTION_UNIFORM_MIX,
        worse_selection_elite_count=BCO_WORSE_SELECTION_ELITE_COUNT,
        trim_grace_period=0,
    ),
    "worse": dict(
        worse_accept_temperature=WORSE_ACCEPT_ON_TEMPERATURE,
        worse_accept_decay=BCO_WORSE_ACCEPT_DECAY,
        worse_accept_min_temperature=BCO_WORSE_ACCEPT_MIN_TEMPERATURE,
        worse_selection_temperature=WORSE_SELECTION_ON_TEMPERATURE,
        worse_selection_decay=BCO_WORSE_SELECTION_DECAY,
        worse_selection_min_temperature=BCO_WORSE_SELECTION_MIN_TEMPERATURE,
        worse_selection_uniform_mix=BCO_WORSE_SELECTION_UNIFORM_MIX,
        worse_selection_elite_count=BCO_WORSE_SELECTION_ELITE_COUNT,
        trim_grace_period=BCO_TRIM_GRACE_PERIOD,
    ),
}


def bco_method(variant, label=None):
    """Build a BCO method spec from a BCO_VARIANTS entry."""
    return {"kind": "bco", "variant": variant,
            "label": label or variant["summary_label"]}


RL_ONLY_METHOD = {"kind": "rl_only", "label": "RL improvement only"}
INITIAL_METHOD = {"kind": "initial", "label": "Initial network"}


def run_method(method, *, init_routes, n_routes=N_ROUTES,
               min_route_len=MIN_ROUTE_LEN, max_route_len=MAX_ROUTE_LEN,
               tensors=None, weights=None, accept_mode="without_worse",
               seed=None, dataset="", run_name_scope="") -> RunResult:
    """Run one method and return a uniform RunResult.

    ``method`` is a spec dict with a ``kind`` of ``initial`` / ``rl_only`` /
    ``bco`` (use :func:`bco_method`, :data:`RL_ONLY_METHOD`,
    :data:`INITIAL_METHOD`). ``accept_mode`` selects worse / without-worse for
    BCO; it is ignored for the other kinds.
    """
    kind = method["kind"]
    label = method.get("label", kind)

    if kind == "initial":
        cfg = build_sa_cfg(f"{run_name_scope}initial", n_routes,
                           min_route_len, max_route_len)
        run_name, metrics, unserved, routes = _run_baseline(
            None, cfg, init_routes, f"{run_name_scope}initial_", {},
            tensors=tensors)
        return RunResult(
            label=label, kind="initial", dataset=dataset, metrics=metrics,
            routes=routes, seed_routes=as_route_tensor(init_routes),
            unserved_demand=unserved, weights=weights or {}, run_name=run_name)

    if kind == "rl_only":
        run_name, metrics, unserved, routes, _, action_stats = (
            run_rl_improvement(
                init_routes, run_name=f"{run_name_scope}rl_only",
                n_routes=n_routes, min_route_len=min_route_len,
                max_route_len=max_route_len, tensors=tensors,
                weights=weights))
        return RunResult(
            label=label, kind="rl_only", dataset=dataset, metrics=metrics,
            routes=routes, seed_routes=as_route_tensor(init_routes),
            unserved_demand=unserved, action_stats=action_stats,
            weights=weights or {}, run_name=run_name)

    if kind == "bco":
        variant = method["variant"]
        acc = ACCEPT_MODES[accept_mode]
        cfg = build_bco_cfg(
            run_name=f"{run_name_scope}{variant['run_name']}_{accept_mode}",
            n_routes=n_routes, min_route_len=min_route_len,
            max_route_len=max_route_len,
            use_neural_bees=variant["use_neural_bees"],
            n_type1_bees=variant["n_type1_bees"],
            n_type2_bees=variant["n_type2_bees"],
            n_type4_bees=variant["n_type4_bees"],
            n_type5_bees=variant.get("n_type5_bees", 0),
            n_type6_bees=variant.get("n_type6_bees", 0),
            n_type7_bees=variant.get("n_type7_bees", 0),
            **acc,
        )
        if seed is not None:
            cfg.experiment.seed = int(seed)
        mutation_counts = {}
        run_name, metrics, unserved, routes, mutation_counts = run_bco(
            cfg, init_routes, mutation_counts_out=mutation_counts,
            tensors=tensors, run_name_scope=run_name_scope)
        mutation_stats = {
            "attempted": mutation_stat_bucket(mutation_counts, "attempted"),
            "accepted": mutation_stat_bucket(mutation_counts, "accepted"),
            "worse_accepted": mutation_stat_bucket(mutation_counts,
                                                   "worse_accepted"),
            "selection": selection_stat_bucket(mutation_counts),
        }
        return RunResult(
            label=label, kind="bco", accept_mode=accept_mode, dataset=dataset,
            metrics=metrics, routes=routes,
            seed_routes=as_route_tensor(init_routes), unserved_demand=unserved,
            mutation_stats=mutation_stats, weights=weights or {},
            run_name=run_name)

    raise ValueError(f"unknown method kind: {kind!r}")
