"""Tunable experiment parameters for the route-evaluation notebook.

Repo paths live in :mod:`eval_lib.context`. The BCO algorithm constants are
sourced from the algorithm YAML ``cfg/bco_mumford.yaml`` -- edit that YAML to
tune them; this module just re-exports the values under their historical
constant names so the notebooks / helpers keep working. The remaining knobs
(route-problem size, cost weights, RL-improvement settings,
run-orchestration flags) stay here.
"""
from omegaconf import OmegaConf as _OmegaConf

from .context import CFG_DIR

CITY_NAME = "Mumford0"

N_ROUTES = 10
MIN_ROUTE_LEN = 2
MAX_ROUTE_LEN = 12

LC_SAMPLES = 100
USE_NEURAL_BCO = False

# === BCO algorithm constants -- sourced from cfg/bco_mumford.yaml ===
# The algorithm YAML is the single source of truth; the BCO_* names below are
# re-exports so existing notebook / helper code keeps working.
_BCO_CFG = _OmegaConf.load(CFG_DIR / "bco_mumford.yaml")

BCO_N_BEES = int(_BCO_CFG.n_bees)
BCO_N_ITERATIONS = int(_BCO_CFG.n_iterations)
BCO_N_TYPE1_BEES = int(_BCO_CFG.n_type1_bees)
BCO_CONSTRUCTION_IGNORE_MAX_ROUTE_LEN = bool(_BCO_CFG.ignore_type4_max_route_len)
BCO_EDIT_IGNORE_MAX_ROUTE_LEN = bool(_BCO_CFG.ignore_type5_max_route_len)
BCO_TRIM_IGNORE_MAX_ROUTE_LEN = bool(_BCO_CFG.ignore_type6_max_route_len)
BCO_TRIM_EXTEND_IGNORE_MAX_ROUTE_LEN = bool(_BCO_CFG.ignore_type7_max_route_len)
BCO_WORSE_ACCEPT_TEMPERATURE = float(_BCO_CFG.worse_accept_temperature)
BCO_WORSE_ACCEPT_DECAY = float(_BCO_CFG.worse_accept_decay)
BCO_WORSE_ACCEPT_MIN_TEMPERATURE = float(_BCO_CFG.worse_accept_min_temperature)
BCO_WORSE_SELECTION_TEMPERATURE = float(_BCO_CFG.worse_selection_temperature)
BCO_WORSE_SELECTION_DECAY = float(_BCO_CFG.worse_selection_decay)
BCO_WORSE_SELECTION_MIN_TEMPERATURE = float(_BCO_CFG.worse_selection_min_temperature)
BCO_WORSE_SELECTION_UNIFORM_MIX = float(_BCO_CFG.worse_selection_uniform_mix)
BCO_WORSE_SELECTION_ELITE_COUNT = int(_BCO_CFG.worse_selection_elite_count)
# Trim-grace selection: a route-shrinking (trim) mutation is force-accepted as
# a setup move and the bee is shielded from cost-based selection for this many
# population-selection rounds, so a follow-up extend can build on the trim.
# Applied only to the worse-accept experiment.
BCO_TRIM_GRACE_PERIOD = int(_BCO_CFG.worse_accept_trim_grace_period)

RUN_RL_ONLY_BASELINE = True  # Set False to skip RL-only eval/table/plot rows.
RL_IMPROVEMENT_RUN_NAME = "rl_improvement_only_from_lc_mumford0"
RL_MAX_ROUTE_EDIT_STEPS = 8
RL_MAX_TRIM_ACTIONS_PER_ROUTE = 1
RL_FORCE_NONHALT_FIRST_STEP = False
RL_RETURN_BEST_ROUTES = False

DEMAND_TIME_WEIGHT = 0.33
ROUTE_TIME_WEIGHT = 0.33
MEDIAN_CONNECTIVITY_WEIGHT = 0.33

# Disable individual cost components (demand / route / connectivity). A
# disabled component is dropped from the weighted cost everywhere -- BCO, LC
# and RL-improvement runs all optimize and are scored without it -- and its
# columns are removed from every comparison table. Leave the list empty to
# keep all three. Example: DISABLED_COST_COMPONENTS = ["connectivity"].
DISABLED_COST_COMPONENTS = ["connectivity"]
ENABLED_COST_COMPONENTS = [
    c for c in ("demand", "route", "connectivity")
    if c not in set(DISABLED_COST_COMPONENTS)
]
