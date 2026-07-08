"""Moved to the library: ``connectpt.routes_generator.paper_experiments.cfg_compose``.

Re-exported so the eval_lib / notebook / test callers keep the same names (one
implementation now lives in the library).
"""
from connectpt.routes_generator.paper_experiments.cfg_compose import (  # noqa: F401
    bco_cfg_set, compose_experiment_cfg, load_experiment_cfg, scoring_cfg,
    set_cfg_value, unify_weights)
