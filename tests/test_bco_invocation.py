"""The bee-colony invocation contract lives in the library.

``build_bee_colony_kwargs`` translates a composed BCO config + a prebuilt
``ExecutablePlan`` into ``run_bee_colony_plan`` kwargs: the plan owns the bee
taxonomy (counts, models, halt / max-len flags), the cfg owns the run-level
schedule + adjustment block. These tests pin the fallback defaults, the
override behaviour, and that the plan is threaded straight through.
"""
from omegaconf import OmegaConf

from connectpt.routes_generator.search.bco_invocation import build_bee_colony_kwargs
from connectpt.routes_generator.search.executable_plan import ExecutablePlan


def test_defaults_match_historical_run_bco():
    cfg = OmegaConf.create({"n_bees": 10, "n_iterations": 3})
    plan = ExecutablePlan.from_flat_cfg(cfg)
    kw = build_bee_colony_kwargs(cfg, plan=plan)
    # the plan owns the taxonomy and is threaded through untouched
    assert kw["plan"] is plan
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
    # mutation_counts_out defaults to a fresh dict
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
    plan = ExecutablePlan.from_flat_cfg(cfg, edit_model=edit)
    counts = {}
    kw = build_bee_colony_kwargs(cfg, plan=plan, mutation_counts_out=counts)
    assert kw["n_bees"] == 20 and kw["n_iterations"] == 100
    assert kw["plan"] is plan
    assert kw["adjustment_degree_weight"] == 10.0
    assert kw["adjustment_degree_target"] == 0.3
    assert kw["adjustment_degree_objective"] == "target"
    assert kw["adjustment_degree_mode"] == "paper"
    assert kw["worse_accept_temperature"] == 0.02
    assert kw["early_stop_patience"] == 7
    assert kw["mutation_counts_out"] is counts
    # the plan captured the taxonomy from the flat cfg
    assert plan.attempted_type_counts()["n_type1"] == 5
    assert plan.attempted_type_counts()["n_type5"] == 5
    assert plan.needs_edit
