"""Build the run-level BCO schedule cfg (worse-accept / adjustment block).

The bee taxonomy (per-type counts, model roles, halt / max-len flags) lives in
the ``ExecutablePlan``; the seeded executor still needs a small cfg node with the
*run schedule* -- ``n_bees`` / ``n_iterations``, the worse-accept / worse-
selection annealing schedule and the adjustment block. ``build_bco_schedule_cfg``
produces exactly that from the unified objective (adjustment) + the BCO algorithm
config (schedule), the same single sources ``compose_bco_cfg`` reads. It carries
no ``n_type*`` -- the plan owns the taxonomy.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from ..objectives import load_bco_algo_config, load_unified_objective


def build_bco_schedule_cfg(*, n_bees: int, n_iterations: int, acceptance=None):
    """Run-schedule cfg for ``run_seeded_bee_colony`` (no bee taxonomy).

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

    node = {
        "n_bees": int(n_bees),
        "n_iterations": int(n_iterations),
        # adjustment penalty -- unified objective, search (two-sided) form
        "adjustment_degree_weight": obj.adj_weight,
        "adjustment_degree_target": obj.adj_target,
        "adjustment_degree_objective": obj.adj_objective,
        "adjustment_degree_gap": obj.adj_gap,
        "adjustment_degree_mode": obj.adj_mode,
        # worse-accept / selection schedule -- BCO algo config (+ acceptance)
        "worse_accept_temperature": float(_acc("worse_accept_temperature")),
        "worse_accept_decay": float(_acc("worse_accept_decay")),
        "worse_accept_min_temperature": float(_acc("worse_accept_min_temperature")),
        "worse_selection_temperature": float(_acc("worse_selection_temperature")),
        "worse_selection_decay": float(_acc("worse_selection_decay")),
        "worse_selection_min_temperature": float(_acc("worse_selection_min_temperature")),
        "worse_selection_uniform_mix": float(_acc("worse_selection_uniform_mix")),
        "worse_selection_elite_count": int(_acc("worse_selection_elite_count")),
        "trim_grace_period": 0,
        "process_neural_bees_sequentially": False,
        "use_demand_weighted_route_selection": False,
        "force_linking_unlinked": False,
        "early_stop_min_delta": 0.0,
    }
    return OmegaConf.create(node)
