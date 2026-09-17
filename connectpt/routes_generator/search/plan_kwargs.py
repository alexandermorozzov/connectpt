"""Build the run-level BCO schedule cfg.

The bee taxonomy (per-type counts, model roles, halt / max-len flags) lives in
the ``ExecutablePlan``; the seeded executor needs only ``n_bees`` /
``n_iterations`` plus the adjustment block, which comes from the cost config.
Every other knob of ``run_bee_colony_plan`` keeps its own default.
"""
from __future__ import annotations

from omegaconf import OmegaConf

def build_bco_schedule_cfg(*, n_bees: int, n_iterations: int,
                           cost_cfg,
                           process_neural_bees_sequentially: bool = False,
                           adj_weight=None):
    """Run-schedule cfg for ``run_seeded_bee_colony`` (no bee taxonomy).

    ``process_neural_bees_sequentially`` runs neural/edit bees one at a time
    (needed for large graphs like EKB to avoid OOM). ``adj_weight`` overrides the
    cost config's adjustment-degree weight for this run (``0`` turns the
    adjustment penalty off).
    """
    kw = cost_cfg.kwargs

    node = {
        "n_bees": int(n_bees),
        "n_iterations": int(n_iterations),
        # adjustment penalty from the cost config (adj_weight turns it off).
        "adjustment_degree_weight": (float(adj_weight) if adj_weight is not None
                                     else kw.adjustment_degree_weight),
        "adjustment_degree_target": kw.adjustment_degree_target,
        "adjustment_degree_objective": kw.adjustment_degree_objective,
        "adjustment_degree_gap": kw.adjustment_degree_gap,
        "adjustment_degree_mode": kw.adjustment_degree_mode,
        "process_neural_bees_sequentially": bool(process_neural_bees_sequentially),
    }
    return OmegaConf.create(node)
