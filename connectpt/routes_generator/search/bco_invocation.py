"""The bee-colony invocation contract, owned by the library.

``build_bee_colony_kwargs`` translates a composed BCO config + a prebuilt
``ExecutablePlan`` into the exact keyword arguments the ``run_bee_colony_plan``
executor expects. The plan owns the bee taxonomy (per-type counts, model roles,
halt / max-len flags); this function only threads the run-level schedule
(worse-accept / worse-selection, trim-grace, early-stop) and the adjustment
block. Adjustment defaults come from the unified objective (single source), not
from scattered literals.

The plan is the single source for *what the bees are*; the caller builds it once
(``ExecutablePlan.from_specs`` or the legacy ``search.compat.plan_from_flat_cfg``
/ ``plan_from_counts``) and hands it in, so the invocation layer no longer
re-derives ``n_type*`` counts.
"""
from __future__ import annotations


def build_bee_colony_kwargs(cfg, *, plan, mutation_counts_out=None) -> dict:
    """Build the ``run_bee_colony_plan`` kwargs from a composed search cfg.

    ``cfg`` is the flexible OmegaConf node a BCO run carries (``n_bees``,
    ``n_iterations``, the worse-accept / worse-selection schedule,
    ``trim_grace_period``, ``early_stop_*`` and the ``adjustment_degree_*``
    block). Optional fields fall back to the exact historical defaults.

    ``plan`` is the ``ExecutablePlan`` that owns the bee taxonomy (counts, the
    construction / edit models, halt and max-len flags) -- it is passed straight
    through. ``mutation_counts_out`` is an optional dict the executor fills in.
    """
    return dict(
        n_bees=cfg.n_bees,
        n_iterations=cfg.n_iterations,
        plan=plan,
        force_linking_unlinked=cfg.get("force_linking_unlinked", False),
        adjustment_degree_weight=cfg.get("adjustment_degree_weight", 0.0),
        adjustment_degree_gap=cfg.get("adjustment_degree_gap", 0.1),
        adjustment_degree_mode=cfg.get("adjustment_degree_mode", "current"),
        adjustment_degree_objective=cfg.get("adjustment_degree_objective", "raw"),
        adjustment_degree_target=cfg.get("adjustment_degree_target", 0.2),
        use_demand_weighted_route_selection=cfg.get(
            "use_demand_weighted_route_selection", False),
        worse_accept_temperature=cfg.get("worse_accept_temperature", 0.0),
        worse_accept_decay=cfg.get("worse_accept_decay", 0.995),
        worse_accept_min_temperature=cfg.get("worse_accept_min_temperature", 0.001),
        worse_selection_temperature=cfg.get("worse_selection_temperature", 0.0),
        worse_selection_decay=cfg.get("worse_selection_decay", 0.995),
        worse_selection_min_temperature=cfg.get(
            "worse_selection_min_temperature", 0.001),
        worse_selection_uniform_mix=cfg.get("worse_selection_uniform_mix", 0.05),
        worse_selection_elite_count=cfg.get("worse_selection_elite_count", 1),
        trim_grace_period=cfg.get("trim_grace_period", 0),
        process_neural_bees_sequentially=cfg.get(
            "process_neural_bees_sequentially", False),
        early_stop_patience=cfg.get("early_stop_patience", None),
        early_stop_min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
        mutation_counts_out={} if mutation_counts_out is None else mutation_counts_out,
    )
