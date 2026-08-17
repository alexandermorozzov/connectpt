"""Legacy ``n_type1..n_type7`` taxonomy: minimal plan-from-counts builder.

The legacy flat-cfg orchestration (``compose_bco_cfg`` / ``plan_from_flat_cfg`` /
``run_bco_from_cfg`` + the numeric ``bee_colony`` engine) was removed in M010 --
the canonical path is declarative (``ExecutablePlan.from_specs``). Only
``plan_from_counts`` survives, as a plan builder for the engine unit tests
(``test_bee_colony``) that exercise ``get_mutants`` over specific type mixes.
"""
from .plans import plan_from_counts

__all__ = ["plan_from_counts"]
