"""Bridge a declarative BeeColonyPlan to the flat bee_colony search cfg.

The seeded executor (``run_seeded_bee_colony`` -> ``build_bee_colony_kwargs``)
consumes a flat cfg (``n_type*_bees`` + halt flags + worse-accept schedule +
adjustment block). ``plan_to_search_cfg`` produces exactly that node from a
``BeeColonyPlan`` (bee counts + per-type halt derived from the declarative
``allowed_actions``): the objective drives the adjustment block, and the BCO
algorithm config drives the worse-accept schedule / ignore-max-len knobs -- the
same single sources ``compose_bco_cfg`` reads. This lets the declarative
bee_set path reuse the identical executor + kwargs contract as the flat path.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from ..objectives import load_bco_algo_config, load_unified_objective


def plan_to_search_cfg(plan, *, n_bees: int, n_iterations: int, acceptance=None):
    """Flat search cfg equivalent to a BeeColonyPlan, for run_seeded_bee_colony.

    ``acceptance`` is an optional worse-accept schedule (the ``search.acceptance``
    config group: worse_accept_* / worse_selection_*). When omitted, the BCO
    algorithm-config defaults (``cfg/bco_mumford.yaml``) are used -- so callers
    that don't compose an acceptance group are unaffected.
    """
    obj = load_unified_objective()
    bco = load_bco_algo_config()
    acc = dict(acceptance) if acceptance is not None else {}

    def _acc(key):
        return acc[key] if key in acc else getattr(bco, key)

    counts = plan.counts
    node = {
        "n_bees": int(n_bees),
        "n_iterations": int(n_iterations),
        "n_type1_bees": int(counts["n_type1"]),
        "n_type2_bees": int(counts["n_type2"]),
        "n_type4_bees": int(counts["n_type4"]),
        "n_type5_bees": int(counts["n_type5"]),
        "n_type6_bees": int(counts["n_type6"]),
        "n_type7_bees": int(counts["n_type7"]),
        # per-type halt permission derived from the declarative allowed_actions
        **{k: bool(v) for k, v in plan.allow_halt.items()},
        # adjustment penalty -- unified objective, search (two-sided) form
        "adjustment_degree_weight": obj.adj_weight,
        "adjustment_degree_target": obj.adj_target,
        "adjustment_degree_objective": obj.adj_objective,
        "adjustment_degree_gap": obj.adj_gap,
        "adjustment_degree_mode": obj.adj_mode,
        # worse-accept / selection schedule + ignore-max-len -- BCO algo config
        "worse_accept_temperature": float(_acc("worse_accept_temperature")),
        "worse_accept_decay": float(_acc("worse_accept_decay")),
        "worse_accept_min_temperature": float(_acc("worse_accept_min_temperature")),
        "worse_selection_temperature": float(_acc("worse_selection_temperature")),
        "worse_selection_decay": float(_acc("worse_selection_decay")),
        "worse_selection_min_temperature": float(_acc("worse_selection_min_temperature")),
        "worse_selection_uniform_mix": float(_acc("worse_selection_uniform_mix")),
        "worse_selection_elite_count": int(_acc("worse_selection_elite_count")),
        "ignore_type4_max_route_len": bool(bco.ignore_type4_max_route_len),
        "ignore_type5_max_route_len": bool(bco.ignore_type5_max_route_len),
        "ignore_type6_max_route_len": bool(bco.ignore_type6_max_route_len),
        "ignore_type7_max_route_len": bool(bco.ignore_type7_max_route_len),
        "trim_grace_period": 0,
        "process_neural_bees_sequentially": False,
        "use_demand_weighted_route_selection": False,
        "force_linking_unlinked": False,
        "early_stop_min_delta": 0.0,
    }
    return OmegaConf.create(node)
