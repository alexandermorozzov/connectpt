"""SA / GA / hyper-heuristic / NSGA-II runners and the cross-benchmark sweep.

Extracted verbatim from the notebook's §9b / §9c / §12 cells -- the function
and config definitions only; the execution that drives each section stays in
the notebook. Helper names resolve via ``from .helpers import *``; each section
keeps its own connectpt-library imports.
"""
from .context import *  # noqa: F401,F403
from .params import *  # noqa: F401,F403
from .helpers import *  # noqa: F401,F403
from . import plots as _plots


def load_benchmark_tensors(city_name: str = CITY_NAME) -> dict:
    """Load a benchmark city's coords / travel-times / demand text files."""
    node_locs = torch.tensor(
        np.genfromtxt(BENCHMARK_DIR / f"{city_name}Coords.txt", skip_header=1),
        dtype=torch.float32,
    )
    street_adj = torch.tensor(
        np.genfromtxt(BENCHMARK_DIR / f"{city_name}TravelTimes.txt"),
        dtype=torch.float32,
    ) * 60
    demand = torch.tensor(
        np.genfromtxt(BENCHMARK_DIR / f"{city_name}Demand.txt"),
        dtype=torch.float32,
    )
    return {"node_locs": node_locs, "street_adj": street_adj, "demand": demand}


# === from the notebook's section 9b (Baseline Optimizers) ===
# Baseline optimizers ported from AHolliday/transit_learning:
# simulated annealing, a genetic algorithm and a selection hyper-heuristic.
# All three are purely heuristic (no neural model). They are run on Mumford0
# from the same LC-seeded initial routes as the BCO experiments above, via the
# shared `test_method` contract, so the table is directly comparable.
import pandas as pd
from IPython.display import display
from omegaconf import OmegaConf

from connectpt.routes_generator import (
    simulated_annealing_with_reheating,
    genetic_algorithm,
    hyperheuristic,
)

# Compute budgets. SA / HH counts (and the NSGA-II budget below) are the
# AHolliday/transit_learning reference values -- cfg/sa_linear.yaml (40000),
# cfg/hyperheuristic.yaml (40000), cfg/nsgaii_husselmann.yaml (2000 / pop 400)
# -- halved. GA has no reference cfg in that repo, so it keeps a notebook
# default.
SA_N_ITERATIONS = 20000
GA_N_ITERATIONS = 200
GA_POP_SIZE = 10
HH_N_ITERATIONS = 20000


def _baseline_cfg_overrides(run_name, n_routes, min_route_len, max_route_len,
                            connectivity_mode="median_weighted"):
    return [
        "+eval=mumford0",
        "++eval.dataset.type=tensor",
        "++experiment.logdir=null",  # no empty TensorBoard run dir for eval runs
        f"++eval.n_routes={n_routes}",
        f"++eval.min_route_len={min_route_len}",
        f"++eval.max_route_len={max_route_len}",
        f"++experiment.cost_function.kwargs.demand_time_weight={DEMAND_TIME_WEIGHT}",
        f"++experiment.cost_function.kwargs.route_time_weight={ROUTE_TIME_WEIGHT}",
        f"++experiment.cost_function.kwargs.median_connectivity_weight={MEDIAN_CONNECTIVITY_WEIGHT}",
        f"++experiment.cost_function.kwargs.connectivity_mode={connectivity_mode}",
        f"++run_name={safe_run_name(run_name)}",
    ]


def _compose_baseline_cfg(config_name, overrides):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    cfg.batch_size = 1
    return cfg


def _early_stop_overrides(early_stop_patience, early_stop_min_delta):
    """Emit ``++early_stop_*`` overrides shared by all baseline cfgs.

    `early_stop_patience=None` disables early-stopping (the algorithm short-
    circuits on `cfg.get("early_stop_patience", None)`); `min_delta` is always
    written so the run_* helper picks it up without a default.
    """
    overrides = [f"++early_stop_min_delta={float(early_stop_min_delta)}"]
    if early_stop_patience is not None:
        overrides.append(f"++early_stop_patience={int(early_stop_patience)}")
    return overrides


def build_sa_cfg(run_name, n_routes, min_route_len, max_route_len,
                 n_iterations=SA_N_ITERATIONS,
                 early_stop_patience=None, early_stop_min_delta=0.0,
                 connectivity_mode="median_weighted"):
    overrides = _baseline_cfg_overrides(
        run_name, n_routes, min_route_len, max_route_len, connectivity_mode)
    overrides.append(f"++alg_args.n_iterations={n_iterations}")
    overrides += _early_stop_overrides(early_stop_patience, early_stop_min_delta)
    return _compose_baseline_cfg("sa_mumford", overrides)


def build_ga_cfg(run_name, n_routes, min_route_len, max_route_len,
                 n_iterations=GA_N_ITERATIONS, population_size=GA_POP_SIZE,
                 early_stop_patience=None, early_stop_min_delta=0.0,
                 connectivity_mode="median_weighted"):
    overrides = _baseline_cfg_overrides(
        run_name, n_routes, min_route_len, max_route_len, connectivity_mode)
    overrides.append(f"++n_iterations={n_iterations}")
    overrides.append(f"++population_size={population_size}")
    overrides += _early_stop_overrides(early_stop_patience, early_stop_min_delta)
    return _compose_baseline_cfg("ga_mumford", overrides)


def build_hh_cfg(run_name, n_routes, min_route_len, max_route_len,
                 n_iterations=HH_N_ITERATIONS, max_repair_iters=None,
                 early_stop_patience=None, early_stop_min_delta=0.0,
                 connectivity_mode="median_weighted"):
    overrides = _baseline_cfg_overrides(
        run_name, n_routes, min_route_len, max_route_len, connectivity_mode)
    overrides.append(f"++n_iterations={n_iterations}")
    if max_repair_iters is not None:
        overrides.append(f"++max_repair_iters={int(max_repair_iters)}")
    overrides += _early_stop_overrides(early_stop_patience, early_stop_min_delta)
    return _compose_baseline_cfg("hh_mumford", overrides)


def _run_baseline(method_fn, cfg, init_routes, prefix, method_kwargs, *,
                  tensors=None, return_history=False, use_weighted_connectivity=None,
                  connectivity_mode=None,
                  adjustment_seed_routes=None,
                  adjustment_degree_weight=0.0, adjustment_degree_target=0.2,
                  adjustment_degree_objective='raw', adjustment_degree_gap=0.1,
                  adjustment_degree_mode='paper'):
    # tensors=None -> Mumford0 dataloader; tensors=<dict> -> explicit tensor
    # dataset (mirrors the run_bco signature). The initial routes are passed
    # through test_method's `routes_tensor` + a "tensor" init config.
    # return_history=True also returns the cost_histories list from test_method
    # as an extra element of the result tuple; default False keeps the legacy
    # 4-tuple contract.
    if tensors is None:
        dataloader = make_test_dataloader(cfg.eval.dataset)
    else:
        dataloader = make_tensor_dataloader(cfg.eval.dataset, tensors)
    device, run_name, _, cost_obj, _ = lrnu.process_standard_experiment_cfg(
        cfg, run_name_prefix=prefix, weights_required=False)
    # Optional: optimize the WEIGHTED mean connectivity (WMC) instead of plain
    # median connectivity, so SA/GA/HH match the unified RTT+WMC+adj objective.
    if use_weighted_connectivity is not None:
        cost_obj.use_weighted_connectivity = bool(use_weighted_connectivity)
    if connectivity_mode is not None:
        cost_obj.connectivity_mode = connectivity_mode
    # Unified adjustment-degree penalty for metaheuristics (SA/GA/HH): the
    # cost module penalizes deviation of the candidate routes from the initial
    # network (same term BCO uses). Gated; off unless weight>0 + a seed.
    # ``adjustment_seed_routes`` decouples the adj reference from the evaluated
    # routes: for an evaluate-only pass (method_fn=None on an external solution,
    # e.g. NSGA-II's chosen front member) ``init_routes`` is the routes being
    # scored, so pass the real seed network here to score adj-vs-seed correctly.
    _adj_seed_src = adjustment_seed_routes if adjustment_seed_routes is not None else init_routes
    _adj_on = bool(adjustment_degree_weight) and adjustment_degree_weight > 0 \
        and _adj_seed_src is not None
    if _adj_on:
        cost_obj.adjustment_degree_weight = float(adjustment_degree_weight)
        cost_obj.adjustment_degree_target = float(adjustment_degree_target)
        cost_obj.adjustment_degree_objective = adjustment_degree_objective
        cost_obj.adjustment_degree_gap = float(adjustment_degree_gap)
        cost_obj.adjustment_degree_mode = adjustment_degree_mode
        _seed = as_route_tensor(_adj_seed_src)
        cost_obj.adjustment_seed = (_seed[None] if _seed.dim() == 2 else _seed).to(device)
    output = lrnu.test_method(
        method_fn,
        dataloader,
        cfg.eval,
        OmegaConf.create({"method": "tensor"}),
        cost_obj,
        silent=False,
        device=device,
        return_routes=True,
        return_histories=return_history,
        routes_tensor=init_routes,
        **method_kwargs,
    )
    if _adj_on:
        cost_obj.adjustment_seed = None
        cost_obj.adjustment_degree_weight = 0.0
    if return_history:
        _, _, unserved_demand, metrics, routes, cost_histories = output
    else:
        _, _, unserved_demand, metrics, routes = output
        cost_histories = None
    routes_tensor = as_route_tensor(routes)
    metrics = add_cost_breakdown_to_metrics(
        metrics, dataloader, cfg.eval, cost_obj, routes_tensor, device)
    result = (run_name, metrics, unserved_demand, routes_tensor)
    if return_history:
        result = result + (cost_histories,)
    return result


def _early_stop_kwargs(cfg):
    """Pull ``early_stop_patience`` / ``early_stop_min_delta`` out of the cfg
    if either was set as a Hydra ++override. Both default to None / 0.0, which
    disables early-stopping inside the algorithm."""
    return dict(
        early_stop_patience=cfg.get("early_stop_patience", None),
        early_stop_min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
    )


def run_sa(cfg, init_routes, *, tensors=None, run_name_scope="",
           return_history=False, **adj):
    method_kwargs = OmegaConf.to_container(cfg.alg_args, resolve=True)
    method_kwargs.update(_early_stop_kwargs(cfg))
    return _run_baseline(
        simulated_annealing_with_reheating, cfg, init_routes,
        f"{run_name_scope}sa_",
        method_kwargs,
        tensors=tensors, return_history=return_history, **adj)


def run_ga(cfg, init_routes, *, tensors=None, run_name_scope="",
           return_history=False, **adj):
    method_kwargs = dict(
        pop_size=int(cfg.population_size),
        n_iterations=int(cfg.n_iterations),
        shorten_prob=float(cfg.get("shorten_prob", 0.2)),
        force_linking_unlinked=bool(cfg.get("force_linking_unlinked", False)),
    )
    method_kwargs.update(_early_stop_kwargs(cfg))
    return _run_baseline(
        genetic_algorithm, cfg, init_routes, f"{run_name_scope}ga_",
        method_kwargs,
        tensors=tensors, return_history=return_history, **adj)


def run_hh(cfg, init_routes, *, tensors=None, run_name_scope="",
           return_history=False, **adj):
    method_kwargs = dict(f_0=float(cfg.f_0), n_steps=int(cfg.n_iterations))
    _mri = cfg.get("max_repair_iters", None)
    if _mri is not None:
        method_kwargs["max_repair_iters"] = int(_mri)
    method_kwargs.update(_early_stop_kwargs(cfg))
    return _run_baseline(
        hyperheuristic, cfg, init_routes, f"{run_name_scope}hh_",
        method_kwargs,
        tensors=tensors, return_history=return_history, **adj)


# === from the notebook's section 9c (NSGA-II) ===
# NSGA-II multi-objective optimizer ported from AHolliday/transit_learning.
# Unlike SA/GA/HH it does not fit the test_method (state, cost, init) ->
# (state, history) contract: it is a class whose .run() returns a Pareto front.
# We run it in the pure-heuristic Husselmann configuration, reduce the Pareto
# front to one solution (minimum weighted-sum cost under the table weights),
# and evaluate that solution with the same MyCostModule the other rows use.
import numpy as np
import matplotlib.pyplot as plt

import connectpt.routes_generator.nsgaii as _nsgaii_mod
from connectpt.routes_generator import NSGAII, RouteGenBatchState
from connectpt.routes_generator import heuristics as _hs

# NSGA-II budget: AHolliday cfg/nsgaii_husselmann.yaml (2000 iters / 400 pop)
# halved. NSGA-II is the heaviest baseline -- the Husselmann initialisation
# does K-shortest-paths over every node pair and each iteration evaluates
# pop_size networks -- so this is slow on Mumford2/3.
NSGAII_N_ITERATIONS = 1000
NSGAII_POP_SIZE = 200


def build_nsgaii_cfg(run_name, n_routes, min_route_len, max_route_len,
                     n_iterations=NSGAII_N_ITERATIONS, pop_size=NSGAII_POP_SIZE,
                     connectivity_mode="median_weighted"):
    overrides = [
        "+eval=mumford0",
        "++eval.dataset.type=tensor",
        "++experiment.logdir=null",  # no empty TensorBoard run dir for eval runs
        f"++eval.n_routes={n_routes}",
        f"++eval.min_route_len={min_route_len}",
        f"++eval.max_route_len={max_route_len}",
        f"++experiment.cost_function.kwargs.connectivity_mode={connectivity_mode}",
        f"++run_name={safe_run_name(run_name)}",
        f"++n_iterations={n_iterations}",
        f"++pop_size={pop_size}",
    ]
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="nsgaii_mumford", overrides=overrides)
    cfg.batch_size = 1
    return cfg


def run_nsgaii(cfg, *, tensors=None, init_routes=None, run_name_scope="",
               use_weighted_connectivity=False,
               connectivity_mode=None,
               adjustment_degree_weight=0.0, adjustment_degree_target=0.2,
               adjustment_degree_objective='raw', adjustment_degree_gap=0.1,
               adjustment_degree_mode='paper'):
    """Run NSGA-II and return ``(run_name, output)``.

    ``init_routes`` (optional) is forwarded as the NSGA-II seed network: it is
    injected into the initial population as one explicit member, with the
    remaining ``pop_size - 1`` slots still filled via ``cfg.init_mode``
    (default ``husselmann``). Pass the same benchmark init the other
    benchmark methods start from to make NSGA-II seeded comparably.

    ``use_weighted_connectivity`` selects the Pareto objectives: ``False``
    (legacy paper baseline) optimizes (mean demand time / ATT, total route time
    / RTT); ``True`` (unified regime) optimizes (RTT, weighted mean
    connectivity / WMC) so the front matches the unified
    route_time + connectivity + adj objective used by the other methods.
    """
    if tensors is None:
        dataloader = make_test_dataloader(cfg.eval.dataset)
    else:
        dataloader = make_tensor_dataloader(cfg.eval.dataset, tensors)
    device, run_name, _, cost_obj, _ = lrnu.process_standard_experiment_cfg(
        cfg, run_name_prefix=f"{run_name_scope}nsgaii_", weights_required=False)
    # Objective set: legacy (ATT, RTT) vs unified (RTT, WMC). The cost module
    # computes median_connectivity_weighted unconditionally, so toggling this
    # only changes which components get_cost stacks into the Pareto objectives.
    cost_obj.use_weighted_connectivity = bool(use_weighted_connectivity)
    if connectivity_mode is not None:
        cost_obj.connectivity_mode = connectivity_mode
    # Unified adjustment-degree penalty: NSGA-II uses MultiObjectiveCostModule
    # (a MyCostModule subclass), so setting the seed + params makes the adj term
    # shift both objectives -> NSGA-II balances deviation like the other methods.
    if adjustment_degree_weight and adjustment_degree_weight > 0 and init_routes is not None:
        cost_obj.adjustment_degree_weight = float(adjustment_degree_weight)
        cost_obj.adjustment_degree_target = float(adjustment_degree_target)
        cost_obj.adjustment_degree_objective = adjustment_degree_objective
        cost_obj.adjustment_degree_gap = float(adjustment_degree_gap)
        cost_obj.adjustment_degree_mode = adjustment_degree_mode
        _seed = as_route_tensor(init_routes)
        cost_obj.adjustment_seed = (_seed[None] if _seed.dim() == 2 else _seed).to(device)
    # NSGA-II reads its device from a module global (the ancestor set it in
    # the Hydra main()); point it at the device we just resolved.
    _nsgaii_mod.DEVICE = device
    data = next(iter(dataloader))
    if device.type != "cpu":
        data = data.cuda()
    state = RouteGenBatchState(data, cost_obj, cfg.eval.n_routes,
                               cfg.eval.min_route_len, cfg.eval.max_route_len)
    mutators = [_hs.add_terminal, _hs.delete_terminal, _hs.add_inside,
                _hs.delete_inside, _hs.invert_nodes, _hs.exchange_routes,
                _hs.replace_node, _hs.donate_node]
    if cfg.get("use_cost_based_heuristics", True):
        mutators += [_hs.cost_based_grow, _hs.cost_based_trim]
    optimizer = NSGAII(
        cost_obj, init_models=[], mutators=mutators,
        n_iterations=int(cfg.n_iterations), pop_size=int(cfg.pop_size),
        p_crossover=float(cfg.p_crossover), p_mutation=float(cfg.p_mutation),
        mutator_p_t=float(cfg.mutator_p_t),
        batch_size=int(cfg.get("gen_batch_size", cfg.pop_size)))
    with torch.no_grad():
        output = optimizer.run(state, cfg.get("init_mode", "husselmann"),
                               sum_writer=None, seed_routes=init_routes)
    return run_name, output


def reduce_pareto_front(output, demand_weight, route_weight):
    """Collapse the Pareto front to its lowest weighted-sum member.

    NSGA-II objectives are (mean demand time, total route time)."""
    pareto = output["pareto_pop"]
    return min(pareto, key=lambda m: (demand_weight * float(m["cost"][0])
                                      + route_weight * float(m["cost"][1])))


# === from the notebook's section 12 (Benchmark Sweep) ===
# Benchmark sweep: RPC init -> BCO variants / RL / SA / GA / HH / NSGA-II.
import gc

from connectpt.routes_generator import CityGraphData, build_nx_heuristic_routes

BENCHMARK_SPECS = [
    {"city": "Mandl",    "n_routes": 6,  "min_route_len": 2,  "max_route_len": 8},
    {"city": "Mumford0", "n_routes": 12, "min_route_len": 2,  "max_route_len": 15},
    {"city": "Mumford1", "n_routes": 15, "min_route_len": 10, "max_route_len": 30},
    {"city": "Mumford2", "n_routes": 56, "min_route_len": 10, "max_route_len": 22},
    {"city": "Mumford3", "n_routes": 60, "min_route_len": 12, "max_route_len": 25},
]
BENCHMARK_NX_SEED = 0
BENCHMARK_INIT_MODE = "rpc"

# requested metrics only: Cp (ATT) | Co (RTT) | d0 | d1 | d2 | d_un | cost


def load_benchmark_graph(spec, init_mode=None):
    """Load a benchmark city and build its benchmark initial route set."""
    init_mode = BENCHMARK_INIT_MODE if init_mode is None else init_mode
    tensors = load_benchmark_tensors(spec["city"])
    if init_mode == "rpc":
        init_routes = build_rpc_routes(
            spec, tensors, run_name=f"benchmark_rpc_init_{spec['city']}",
            n_samples=1)
    elif init_mode == "nx":
        graph = CityGraphData.from_tensors(
            tensors["node_locs"], tensors["street_adj"], tensors["demand"],
            pos_only=False)
        init_routes = build_nx_heuristic_routes(
            graph, num_routes=spec["n_routes"], min_len=spec["min_route_len"],
            max_len=spec["max_route_len"], seed=BENCHMARK_NX_SEED)
    else:
        raise ValueError(f"Unknown BENCHMARK_INIT_MODE: {init_mode!r}")
    return tensors, init_routes


