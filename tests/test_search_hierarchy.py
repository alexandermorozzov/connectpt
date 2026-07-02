"""U3: reusable search config groups (acceptance) compose + reach the runner.

The worse-accept schedule is a composable `search/acceptance` group instead of a
per-experiment baked block: `greedy` (default, matches the BCO algo-config) and
`mandl_tuned` (the small annealing schedule for Mandl/MACSA). plan_to_search_cfg
threads it into the bee_colony kwargs; omitting it falls back to the algo-config
defaults (so existing callers are unaffected).
"""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from connectpt.routes_generator.search.bee_plan import BeeColonyPlan
from connectpt.routes_generator.search.bee_specs import parse_bee_specs
from connectpt.routes_generator.search.plan_kwargs import plan_to_search_cfg
from connectpt.routes_generator.search.search_policies import (
    ConstructionSearchPolicy, EditSearchPolicy)

LIB_CFG = Path(__file__).resolve().parents[1] / "connectpt" / "routes_generator" / "cfg"


def _plan():
    policies = {"construction": ConstructionSearchPolicy(model=None, name="construction"),
                "edit": EditSearchPolicy(model=None, name="edit")}
    specs = parse_bee_specs(OmegaConf.load(LIB_CFG / "search" / "bee_sets" / "our_nbco.yaml").bees)
    return BeeColonyPlan.from_specs(specs, policies)


def test_acceptance_default_matches_algo_config():
    cfg = plan_to_search_cfg(_plan(), n_bees=10, n_iterations=3, acceptance=None)
    # BCO algo-config (cfg/bco_mumford.yaml) greedy defaults
    assert cfg.worse_accept_temperature == 0.0
    assert cfg.worse_selection_uniform_mix == 0.05
    assert cfg.worse_selection_elite_count == 1


def test_acceptance_mandl_tuned_overrides_schedule():
    mandl = OmegaConf.load(LIB_CFG / "search" / "acceptance" / "mandl_tuned.yaml").search.acceptance
    cfg = plan_to_search_cfg(_plan(), n_bees=10, n_iterations=3, acceptance=dict(mandl))
    assert cfg.worse_accept_temperature == 0.02
    assert cfg.worse_accept_decay == 0.985
    assert cfg.worse_selection_uniform_mix == 0.10
    assert cfg.worse_selection_elite_count == 2


def test_bee_colony_base_composes_greedy_acceptance_by_default():
    with initialize_config_dir(config_dir=str(LIB_CFG), version_base=None):
        base = compose(config_name="search/bee_colony_base")
        tuned = compose(config_name="search/bee_colony_base",
                        overrides=["search/acceptance=mandl_tuned"])
    assert base.search.acceptance.worse_accept_temperature == 0.0
    assert tuned.search.acceptance.worse_accept_temperature == 0.02
