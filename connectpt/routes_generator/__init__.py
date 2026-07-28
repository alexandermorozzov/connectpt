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
    - training loops (PPO / REINFORCE-style inductive learning),
    - evaluation utilities for route generators,
    - replay buffers (standard & prioritized),
    - model-building helpers and common utilities.

Typical usage:
---------------
>>> from transit_learning import CityGraphData, RouteGenBatchState, get_cost_module_from_cfg
>>> from transit_learning import get_dynamic_training_set, init_from_cfg, bee_colony
>>> from transit_learning import PathCombiningRouteGenerator, test_method

# Load or generate a dataset
>>> dataset = get_dynamic_training_set(min_nodes=10, max_nodes=100)

# Build a model and cost function
>>> cost_obj = get_cost_module_from_cfg(cfg.cost)
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
from .citygraph_dataset import (
    CityGraphDataset,
    DynamicCityGraphDataset,
    InsertPosFeatures,
    RandomFlipCity,
    SpaceScaleTransform,
    DemandScaleTransform,
    CityGraphData,
    get_default_train_and_eval_split,
    get_dataset_from_config,
    get_dynamic_training_set,
)

# === Transit Time & Cost Estimation ===
from .transit_time_estimator import (
    ExtraStateData,
    RouteGenBatchState,
    CostHelperOutput,
    CostModule,
    MyCostModule,
    MultiObjectiveCostModule,
    NikolicCostModule,
    ROUTE_ACTION_EXTEND,
    ROUTE_ACTION_HALT,
    ROUTE_ACTION_TRIM_END,
    ROUTE_ACTION_TRIM_START,
    COST_WEIGHT_KEY_ORDER,
    COST_COMPONENT_NAMES,
    resolve_cost_component_index,
    resolve_enabled_cost_components,
    get_cost_module_from_cfg,
)

# === Initialization Methods ===
from .initialization import (
    init_from_cfg,
    john_init,
    nikolic_init,
)

# === Optimization & Search ===
# The bee-colony engine is ``bee_colony.run_bee_colony_plan`` (plan-driven);
# the search application wraps it via ``search.BeeColonySearchRun``.

# === Comparison Baselines (ported from AHolliday/transit_learning) ===
from .simulated_annealing import simulated_annealing_with_reheating
from .genetic_algorithm import run as genetic_algorithm
from .hyperheuristics import hyperheuristic
from .nsgaii import NSGAII, husselmann_init
from . import heuristics
from .nx_heuristic import build_nx_heuristic_routes, generate_initial_routes

# === Evaluation & Sampling ===
from .eval_route_generator import (
    sample_from_model,
    eval_model,
)

# === Inductive Learning ===
from .inductive_route_learning import (
    Baseline,
    FixedBaseline,
    RollingBaseline,
    NNBaseline,
    BLMODE_NONE,
    BLMODE_GREEDY,
    BLMODE_ROLL,
    BLMODE_NN,
)

# === Models & Architectures ===
from .models import (
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
from .utils import (
    build_model_from_cfg,
    get_graphnet_from_cfg,
    get_random_path_combiner,
    process_standard_experiment_cfg,
    test_method,
    yen_k_shortest_paths,
    log_config,
)

# === Replay Buffers ===
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# === Torch Utilities ===
from .torch_utils import (
    load_routes_tensor,
    dump_routes,
    get_batch_tensor_from_routes,
    cat_var_size_tensors,
    get_update_at_mask,
    get_indices_from_mask,
    get_variable_slice_mask,
    unravel_indices,
    AllShortestPaths,
)

# === Build Dataset (Simulation-Based) ===
from .build_dataset import build_dataset

# === Experiment orchestration API (M010 stage 6) ===
# The narrow public surface the notebook uses: name an experiment, run it,
# render it. Everything above is the legacy/library internals surface (narrowed
# in stage 7 once eval_lib + baselines dissolve).
from .core import (
    ExperimentRunFactory,
    ExperimentBatch,
    build_experiment,
    load_experiment,
    load_suite,
)
from .reports import render_report

# Author & license
# __author__ = "Andrew Holliday"
# __license__ = "GNU General Public License v3"
