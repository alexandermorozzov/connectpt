"""
Learning package for the Transit Learning project.

This package provides tools for:
    - generating synthetic city graphs with streets, stops, and demand,
    - dynamic dataset creation for training models on transit networks,
    - estimating transit times, costs, and route planning,
    - computing cost objectives for route optimization (e.g., demand time, route time, connectivity),
    - handling heterogeneous graph data for stops, streets, and demand,
    - executing transformations like scaling, flipping, and rotating graphs,
    - neural network models for inductive route generation,
    - heuristic initialization methods (John et al., Nikolić & Teodorović),
    - Bee Colony Optimization (BCO),
    - training loops (PPO / D3PO inductive learning),
    - evaluation utilities for route generators,
    - model-building helpers and common utilities.

Typical usage:
---------------
>>> from transit_learning import CityGraphData, RouteGenBatchState, build_cost
>>> from transit_learning import get_dynamic_training_set, init_from_cfg, bee_colony
>>> from transit_learning import PathCombiningRouteGenerator, test_method

# Load or generate a dataset
>>> dataset = get_dynamic_training_set(min_nodes=10, max_nodes=100)

# Build a model and cost function
>>> cost_obj = build_cost(cfg.cost)
>>> model = build_model_from_cfg(cfg.model, cfg.experiment)

# Run BCO with initialization
>>> state = RouteGenBatchState(data_list, cost_obj, n_routes=10)
>>> init_routes = init_from_cfg(state, cfg.init)
>>> final_routes = bee_colony(state, cost_obj, init_routes, n_bees=50)

# Evaluate any method
>>> cost, unserved, metrics = test_method(my_method, dataloader, cfg.eval, cfg.init, cost_obj)
"""

import importlib

__version__ = importlib.metadata.version("connectpt")

# === Dataset & Graph Utilities ===
from .city.synthetic import CityGraphDataset, DynamicCityGraphDataset, get_default_train_and_eval_split, get_dynamic_training_set
from .city.graph import InsertPosFeatures, RandomFlipCity, SpaceScaleTransform, DemandScaleTransform, CityGraphData

# === Transit Time & Cost Estimation ===
from .env.actions import ROUTE_ACTION_EXTEND, ROUTE_ACTION_HALT, ROUTE_ACTION_TRIM_END, ROUTE_ACTION_TRIM_START
from .env.state import ExtraStateData, RouteGenBatchState
from .objectives.cost import CostHelperOutput, CostModule, MyCostModule, build_cost
from .objectives.preferences import COST_WEIGHT_KEY_ORDER, COST_COMPONENT_NAMES

# === Initialization Methods ===
from .search.initial import (
    john_init,
    nikolic_init,
)

# === Optimization & Search ===
# The bee-colony engine is ``bee_colony.run_bee_colony_plan`` (plan-driven);
# the search application wraps it via ``search.BeeColonySearchRun``.

from .policy.generate import generate, evaluate_policy
from .policy.build import load_policy

# === Models & Architectures ===
from .policy.models import (
    PathCombiningRouteGenerator,
    TrimPathCombiningRouteGenerator,
    RandomPathCombiningRouteGenerator,
    RandomTrimExtendRouteGenerator,
    UnbiasedPathCombiner,
    NodeWalker,
    GraphEncoder,
    KoolGraphEncoder,
    KoolNextNodeScorer,
    FeatureNorm,
    get_mlp,
    PlanResults,
    RouteGenResults,
    RouteChoiceResults,
    FreqChoiceResults,
)

# === Utilities & Helpers ===
from .city.shortest_paths import yen_k_shortest_paths

# === Torch Utilities ===
from .run.artifacts import load_routes, save_routes
from .network.routes import to_routes
from .common.tensors import cat_var_size_tensors, get_update_at_mask, get_variable_slice_mask, unravel_indices
from .policy.models import get_indices_from_mask
from .city.shortest_paths import AllShortestPaths

# === Experiment orchestration API (M010 stage 6) ===
# The narrow public surface the notebook uses: name an experiment, run it,
# render it. Everything above is the legacy/library internals surface (narrowed
# in stage 7 once eval_lib + baselines dissolve).
from .run.factory import ExperimentRunFactory
from .run.config import build_experiment, load_experiment
from .reports import render_report

# Author & license
# __author__ = "Andrew Holliday"
# __license__ = "GNU General Public License v3"
