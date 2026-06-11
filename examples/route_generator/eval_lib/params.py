"""Tunable experiment parameters -- the single source of truth.

Repo paths live in :mod:`eval_lib.context`. The BCO algorithm constants are
sourced from the algorithm YAML ``cfg/bco_mumford.yaml`` -- edit that YAML to
tune them; this module just re-exports the values under their historical
constant names so the notebook / helpers keep working.

The **unified paper objective** block below is read by BOTH parts of
``paper_combined.ipynb``: PART 1 trains the edit model on exactly the same
objective the PART 2 experiments (our model + all BCO baselines) optimize.
Change a value here and training, eval-lib config builders and the notebook
all follow.
"""
from omegaconf import OmegaConf as _OmegaConf

from .context import CFG_DIR

# === Unified paper objective (paper_combined.ipynb) =========================
# Training (PART 1) and experiments (PART 2) both optimize
#     0.5 * RTT  +  0.5 * WMC  +  ADJ_WEIGHT * |adj - ADJ_TARGET|
# with the demand component disabled and WMC = the demand-weighted median
# connectivity (modified-Cp, ``median_weighted``).
CONNECTIVITY_MODE = "median_weighted"
DISABLED_COST_COMPONENTS = ["demand"]

DEMAND_TIME_WEIGHT = 0.0
ROUTE_TIME_WEIGHT = 0.5
MEDIAN_CONNECTIVITY_WEIGHT = 0.5
UNIFIED_COST_WEIGHTS = dict(
    demand_time_weight=DEMAND_TIME_WEIGHT,
    route_time_weight=ROUTE_TIME_WEIGHT,
    median_connectivity_weight=MEDIAN_CONNECTIVITY_WEIGHT,
)

# Adjustment-degree penalty. The SEARCH (BCO acceptance, E1u/E2 experiments)
# uses the two-sided "target" objective ADJ_WEIGHT * |adj - ADJ_TARGET|: the
# front is anchored at the prescribed modification budget. TRAINING reward
# shaping uses the one-sided "cap" max(0, adj - ADJ_TARGET) instead -- a
# two-sided term as a *reward* pays the agent for arbitrary changes up to the
# target (and rewards corrupting clean networks), drowning the RTT/WMC signal;
# as a budget upper bound it leaves the improvement reward untouched below the
# target. Both penalize the NETWORK-mean degree.
ADJ_WEIGHT = 10.0
ADJ_TARGET = 0.2
ADJ_OBJECTIVE = "target"        # eval / BCO search acceptance
ADJ_TRAIN_OBJECTIVE = "cap"     # PPO reward shaping (training only)
ADJ_GAP = 0.1
ADJ_MODE = "paper"

# === Problem-size defaults (config builders) ================================
CITY_NAME = "Mumford0"

MIN_ROUTE_LEN = 2
MAX_ROUTE_LEN = 12

LC_SAMPLES = 100
USE_NEURAL_BCO = False

# === BCO algorithm constants -- sourced from cfg/bco_mumford.yaml ===========
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
