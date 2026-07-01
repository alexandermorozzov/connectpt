"""The bee-colony invocation contract, owned by the library.

``build_bee_colony_kwargs`` translates a composed BCO config (the ``n_bees`` /
``n_type*_bees`` / worse-accept schedule / halt flags / adjustment block a
search run carries) into the exact keyword arguments the ``bee_colony`` executor
expects. Adjustment defaults come from the unified objective (single source),
not from scattered literals.

This centralises the parameter threading that used to live inline in the
notebook-support ``run_bco`` helper, so the library -- not eval_lib -- owns how a
bee-colony search is parameterised. Behaviour is byte-identical to that inline
dict: the fallback defaults match the historical ``cfg.get(key, <literal>)``
values exactly, so callers whose cfg omits a key are unaffected.
"""
from __future__ import annotations


def build_bee_colony_kwargs(cfg, *, bee_model=None, edit_model=None,
                            mutation_counts_out=None) -> dict:
    """Build the ``bee_colony`` kwargs from a composed search cfg.

    ``cfg`` is the flexible OmegaConf node a BCO run carries (``n_bees``,
    ``n_iterations``, ``n_type*_bees``, the worse-accept / worse-selection
    schedule, ``type*_allow_halt``, ``ignore_type*_max_route_len``,
    ``trim_grace_period``, ``early_stop_*`` and the ``adjustment_degree_*``
    block). Optional fields fall back to the exact defaults ``run_bco`` used.

    ``bee_model`` (neural rebuild/construction) and ``edit_model`` (trim/extend)
    are passed straight through -- their state lives in the loaded model, not
    here. ``mutation_counts_out`` is an optional dict the executor fills in.
    """
    return dict(
        n_bees=cfg.n_bees,
        n_iterations=cfg.n_iterations,
        n_type1_bees=cfg.get("n_type1_bees", None),
        n_type2_bees=cfg.get("n_type2_bees", None),
        n_type4_bees=cfg.get("n_type4_bees", 0),
        n_type5_bees=int(cfg.get("n_type5_bees", 0)),
        n_type6_bees=int(cfg.get("n_type6_bees", 0)),
        n_type7_bees=int(cfg.get("n_type7_bees", 0)),
        bee_model=bee_model,
        edit_model=edit_model,
        force_linking_unlinked=cfg.get("force_linking_unlinked", False),
        adjustment_degree_weight=cfg.get("adjustment_degree_weight", 0.0),
        adjustment_degree_gap=cfg.get("adjustment_degree_gap", 0.1),
        adjustment_degree_mode=cfg.get("adjustment_degree_mode", "current"),
        adjustment_degree_objective=cfg.get("adjustment_degree_objective", "raw"),
        adjustment_degree_target=cfg.get("adjustment_degree_target", 0.2),
        ignore_type4_max_route_len=cfg.get("ignore_type4_max_route_len", False),
        ignore_type5_max_route_len=cfg.get("ignore_type5_max_route_len", False),
        type4_allow_halt=cfg.get("type4_allow_halt", True),
        type5_allow_halt=cfg.get("type5_allow_halt", True),
        type6_allow_halt=cfg.get("type6_allow_halt", True),
        type7_allow_halt=cfg.get("type7_allow_halt", True),
        ignore_type6_max_route_len=cfg.get("ignore_type6_max_route_len", False),
        ignore_type7_max_route_len=cfg.get("ignore_type7_max_route_len", False),
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
