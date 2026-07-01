"""The bee-colony invocation contract lives in the library.

``build_bee_colony_kwargs`` is the single source for translating a composed BCO
config into ``bee_colony`` kwargs (it replaced the inline dict in eval_lib's
``run_bco``). These tests pin the fallback defaults and the override behaviour.
"""
from omegaconf import OmegaConf

from connectpt.routes_generator.search.bco_invocation import build_bee_colony_kwargs


def test_defaults_match_historical_run_bco():
    cfg = OmegaConf.create({"n_bees": 10, "n_iterations": 3})
    kw = build_bee_colony_kwargs(cfg)
    # adjustment fallbacks (were literal cfg.get(..., <default>) in run_bco)
    assert kw["adjustment_degree_weight"] == 0.0
    assert kw["adjustment_degree_gap"] == 0.1
    assert kw["adjustment_degree_mode"] == "current"
    assert kw["adjustment_degree_objective"] == "raw"
    assert kw["adjustment_degree_target"] == 0.2
    # worse-accept / worse-selection schedule defaults
    assert kw["worse_accept_decay"] == 0.995
    assert kw["worse_selection_uniform_mix"] == 0.05
    assert kw["worse_selection_elite_count"] == 1
    # halt flags default on; ignore-max-len default off
    assert all(kw[f"type{i}_allow_halt"] for i in (4, 5, 6, 7))
    assert not any(kw[f"ignore_type{i}_max_route_len"] for i in (4, 5, 6, 7))
    # models default to None; mutation_counts_out defaults to a fresh dict
    assert kw["bee_model"] is None and kw["edit_model"] is None
    assert kw["mutation_counts_out"] == {}


def test_cfg_values_override_defaults():
    cfg = OmegaConf.create({
        "n_bees": 20, "n_iterations": 100,
        "n_type1_bees": 5, "n_type5_bees": 5,
        "adjustment_degree_weight": 10.0, "adjustment_degree_target": 0.3,
        "adjustment_degree_objective": "target", "adjustment_degree_mode": "paper",
        "worse_accept_temperature": 0.02, "early_stop_patience": 7,
    })
    edit = object()
    counts = {}
    kw = build_bee_colony_kwargs(cfg, edit_model=edit, mutation_counts_out=counts)
    assert kw["n_bees"] == 20 and kw["n_iterations"] == 100
    assert kw["n_type1_bees"] == 5 and kw["n_type5_bees"] == 5
    assert kw["adjustment_degree_weight"] == 10.0
    assert kw["adjustment_degree_target"] == 0.3
    assert kw["adjustment_degree_objective"] == "target"
    assert kw["adjustment_degree_mode"] == "paper"
    assert kw["worse_accept_temperature"] == 0.02
    assert kw["early_stop_patience"] == 7
    assert kw["edit_model"] is edit
    assert kw["mutation_counts_out"] is counts
