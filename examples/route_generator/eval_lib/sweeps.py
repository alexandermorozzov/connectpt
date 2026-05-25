"""Seed-sweep configuration constants and a few MACSA helpers.

The three bespoke sweep drivers that used to live here
(``run_worse_accept_seed_sweep`` / ``run_nx_dataset_bco_seed_sweep`` /
``run_macsa_seeded_comparison``) have been replaced by the single unified
``run_seed_sweep`` (eval_lib.sweep), which wraps the shared ``run_method``.

What remains: the accept-experiment / seed / reuse-flag constants still read by
the notebook sections, and ``macsa_eval_bounds`` (per-scenario eval bounds,
used by the §11 MACSA section).
"""
from .context import *  # noqa: F401,F403
from .params import *  # noqa: F401,F403
from .helpers import *  # noqa: F401,F403
from .baselines import *  # noqa: F401,F403


# === section 8 (Worse Acceptance Seed Sweep) ===
WORSE_ACCEPT_SEEDS = list(range(1))  # seed 0 only
WORSE_ACCEPT_ON_TEMPERATURE = 2
WORSE_SELECTION_ON_TEMPERATURE = 2
WORSE_ACCEPT_DUMP_ROUTES = False
REUSE_EXISTING_SWEEP_RESULTS = True

WORSE_ACCEPT_EXPERIMENTS = [
    {
        "key": "without_worse",
        "label": "without worse acceptance",
        "temperature": 0.0,
        "decay": BCO_WORSE_ACCEPT_DECAY,
        "min_temperature": BCO_WORSE_ACCEPT_MIN_TEMPERATURE,
        "selection_temperature": 0.0,
        "selection_decay": BCO_WORSE_SELECTION_DECAY,
        "selection_min_temperature": BCO_WORSE_SELECTION_MIN_TEMPERATURE,
        "selection_uniform_mix": BCO_WORSE_SELECTION_UNIFORM_MIX,
        "selection_elite_count": BCO_WORSE_SELECTION_ELITE_COUNT,
        "trim_grace_period": 0,
    },
    {
        "key": "with_worse",
        "label": f"with worse acceptance + soft selection (T0={WORSE_ACCEPT_ON_TEMPERATURE})",
        "temperature": WORSE_ACCEPT_ON_TEMPERATURE,
        "decay": BCO_WORSE_ACCEPT_DECAY,
        "min_temperature": BCO_WORSE_ACCEPT_MIN_TEMPERATURE,
        "selection_temperature": WORSE_SELECTION_ON_TEMPERATURE,
        "selection_decay": BCO_WORSE_SELECTION_DECAY,
        "selection_min_temperature": BCO_WORSE_SELECTION_MIN_TEMPERATURE,
        "selection_uniform_mix": BCO_WORSE_SELECTION_UNIFORM_MIX,
        "selection_elite_count": BCO_WORSE_SELECTION_ELITE_COUNT,
        "trim_grace_period": BCO_TRIM_GRACE_PERIOD,
    },
]


# === section 8b (NX Heuristic Dataset BCO Seed Sweep) ===
NX_DATASET_SWEEP_SEEDS = WORSE_ACCEPT_SEEDS
NX_DATASET_SWEEP_ACCEPT_EXPERIMENTS = WORSE_ACCEPT_EXPERIMENTS
NX_DATASET_SWEEP_VARIANT_KEYS = [variant["key"] for variant in BCO_VARIANTS]
NX_DATASET_SWEEP_DUMP_ROUTES = False
REUSE_EXISTING_NX_SWEEP_RESULTS = True


# === section 11 (MACSA Seeded BCO/RL Comparison) ===
MACSA_SEEDS = WORSE_ACCEPT_SEEDS
MACSA_ACCEPT_EXPERIMENTS = WORSE_ACCEPT_EXPERIMENTS
MACSA_VARIANT_KEYS = [variant["key"] for variant in BCO_VARIANTS]
MACSA_DUMP_ROUTES = False
MACSA_RUN_RL_IMPROVEMENT = RUN_RL_ONLY_BASELINE
REUSE_EXISTING_MACSA_COMPARISON = True


def macsa_eval_bounds(scenario):
    n_nodes = int(scenario["tensors"]["node_locs"].shape[0])
    n_routes = int(scenario["routes"].shape[1])
    seed_route_lens = (scenario["routes"] > -1).sum(dim=-1)
    longest_seed_route = int(seed_route_lens.max().item()) if seed_route_lens.numel() else MIN_ROUTE_LEN
    max_route_len = min(n_nodes, max(MAX_ROUTE_LEN, longest_seed_route))
    return n_routes, MIN_ROUTE_LEN, max_route_len
