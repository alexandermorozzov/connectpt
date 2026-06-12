"""Helper functions for the route-evaluation notebook.

Extracted verbatim from the notebook's "Helper Functions" cell so the notebook
stays readable. Free names the helpers used to resolve against the notebook
namespace are provided here: repo paths via :mod:`eval_lib.context`, tunable
experiment constants via :mod:`eval_lib.params`, route-plotting helpers via
:mod:`eval_lib.plots`, and the benchmark tensors via the module-level
``INPUT_TENSORS`` (registered once by the notebook with
:func:`set_input_tensors`, replacing the old ``input_tensors`` notebook global).
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from connectpt.routes_generator.citygraph_dataset import (
    get_dataset_from_config, load_macsa_scenarios)
from connectpt.routes_generator.improvement_learning import (
    load_raw_graphs_and_lc_routes, rollout_lc_improvement,
    summarize_route_action_stats)
from connectpt.routes_generator.utils import get_eval_cfg
from connectpt.routes_generator.eval_route_generator import eval_model
from connectpt.routes_generator.bee_colony import bee_colony
from connectpt.routes_generator.torch_utils import (
    dump_routes, get_batch_tensor_from_routes)
from connectpt.routes_generator.transit_time_estimator import RouteGenBatchState
import connectpt.routes_generator.utils as lrnu

from .context import (ROOT_DIR, CFG_DIR, BENCHMARK_DIR, MACSA_DATA_DIR,
                      MODEL_WEIGHTS_PATH, EDIT_MODEL_WEIGHTS_PATH,
                      OUTPUT_ROUTES_DIR)
from .params import *  # noqa: F401,F403  (notebook experiment constants)
from . import plots as _plots


# Benchmark tensors used by make_test_dataloader. The notebook registers them
# once via set_input_tensors(); this replaces the old `input_tensors` global.
INPUT_TENSORS = None

# Adjustment-conditioning feature count of the edit-model checkpoint pointed
# to by EDIT_MODEL_WEIGHTS_PATH (0 = unconditioned legacy checkpoints).
EDIT_MODEL_N_ADJ_COND_FEATS = 0


def set_input_tensors(tensors):
    """Register the default benchmark tensors used by make_test_dataloader."""
    global INPUT_TENSORS
    INPUT_TENSORS = tensors


def make_test_dataloader(dataset_cfg):
    dataset = get_dataset_from_config(dataset_cfg, tensors=INPUT_TENSORS)
    return DataLoader(dataset, batch_size=1)


def make_tensor_dataloader(dataset_cfg, tensors):
    dataset = get_dataset_from_config(dataset_cfg, tensors=tensors)
    return DataLoader(dataset, batch_size=1)


def safe_run_name(run_name: str) -> str:
    run_name = str(run_name)
    safe_chars = []
    for char in run_name:
        if char.isascii() and (char.isalnum() or char in "._-"):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe = "_".join(part for part in "".join(safe_chars).split("_") if part)
    return safe.strip("._-") or "run"


def _apply_disabled_components_to_cfg(cfg):
    """Inject DISABLED_COST_COMPONENTS into the cost-function kwargs so
    every cost module built from this cfg drops those components from the
    weighted cost (and renormalizes the survivors)."""
    if not DISABLED_COST_COMPONENTS:
        return cfg
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.cost_function.kwargs.disabled_components = list(
        DISABLED_COST_COMPONENTS
    )
    OmegaConf.set_struct(cfg, True)
    return cfg


def build_lc_cfg(
    run_name: str,
    n_routes: int,
    min_route_len: int,
    max_route_len: int,
    demand_time_weight: float = DEMAND_TIME_WEIGHT,
    route_time_weight: float = ROUTE_TIME_WEIGHT,
    median_connectivity_weight: float = MEDIAN_CONNECTIVITY_WEIGHT,
    connectivity_mode: str = "median_weighted",
):
    run_name = safe_run_name(run_name)
    params = {
        "dataset_name": "tensor",
        "n_routes": n_routes,
        "min_route_len": min_route_len,
        "max_route_len": max_route_len,
        "demand_time_weight": demand_time_weight,
        "route_time_weight": route_time_weight,
        "median_connectivity_weight": median_connectivity_weight,
        "connectivity_mode": connectivity_mode,
        "run_name": run_name,
        "model_weights": str(MODEL_WEIGHTS_PATH),
    }
    cfg = get_eval_cfg(str(CFG_DIR), "eval_model_mumford", params)
    cfg.batch_size = 1
    return _apply_disabled_components_to_cfg(cfg)


def build_rpc_cfg(
    run_name: str,
    n_routes: int,
    min_route_len: int,
    max_route_len: int,
    demand_time_weight: float = DEMAND_TIME_WEIGHT,
    route_time_weight: float = ROUTE_TIME_WEIGHT,
    median_connectivity_weight: float = MEDIAN_CONNECTIVITY_WEIGHT,
    connectivity_mode: str = "median_weighted",
):
    """Build an LC-style eval cfg backed by RPC/pi_random instead of weights.

    This mirrors the ``benchmark_init_visualize.ipynb`` RPC panel: reuse the
    LC evaluation plumbing, but override the model with
    ``random_path_combiner`` so routes are composed from random shortest paths
    without trained construction weights.
    """
    run_name = safe_run_name(run_name)
    overrides = [
        "+eval=mumford0",
        "++eval.dataset.type=tensor",
        "++experiment.logdir=null",
        f"++eval.n_routes={n_routes}",
        f"++eval.min_route_len={min_route_len}",
        f"++eval.max_route_len={max_route_len}",
        f"++experiment.cost_function.kwargs.demand_time_weight={demand_time_weight}",
        f"++experiment.cost_function.kwargs.route_time_weight={route_time_weight}",
        f"++experiment.cost_function.kwargs.median_connectivity_weight={median_connectivity_weight}",
        f"++experiment.cost_function.kwargs.connectivity_mode={connectivity_mode}",
        f"++run_name={run_name}",
        "model=random_path_combiner",
    ]
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="eval_model_mumford", overrides=overrides)
    cfg.batch_size = 1
    return _apply_disabled_components_to_cfg(cfg)


def build_bco_cfg(
    run_name: str,
    n_routes: int,
    min_route_len: int,
    max_route_len: int,
    use_neural_bees: bool = USE_NEURAL_BCO,
    n_bees: int = BCO_N_BEES,
    n_type1_bees: int = BCO_N_TYPE1_BEES,
    n_type2_bees: int | None = None,
    n_type4_bees: int = 0,
    n_type5_bees: int = 0,
    n_type6_bees: int = 0,
    n_type7_bees: int = 0,
    ignore_type4_max_route_len: bool = BCO_CONSTRUCTION_IGNORE_MAX_ROUTE_LEN,
    ignore_type5_max_route_len: bool = BCO_EDIT_IGNORE_MAX_ROUTE_LEN,
    ignore_type6_max_route_len: bool = BCO_TRIM_IGNORE_MAX_ROUTE_LEN,
    ignore_type7_max_route_len: bool = BCO_TRIM_EXTEND_IGNORE_MAX_ROUTE_LEN,
    type4_allow_halt: bool = True,
    type5_allow_halt: bool = True,
    type6_allow_halt: bool = True,
    type7_allow_halt: bool = True,
    use_demand_weighted_route_selection: bool = False,
    demand_time_weight: float = DEMAND_TIME_WEIGHT,
    route_time_weight: float = ROUTE_TIME_WEIGHT,
    median_connectivity_weight: float = MEDIAN_CONNECTIVITY_WEIGHT,
    connectivity_mode: str = "median_weighted",
    worse_accept_temperature: float = BCO_WORSE_ACCEPT_TEMPERATURE,
    worse_accept_decay: float = BCO_WORSE_ACCEPT_DECAY,
    worse_accept_min_temperature: float = BCO_WORSE_ACCEPT_MIN_TEMPERATURE,
    worse_selection_temperature: float = BCO_WORSE_SELECTION_TEMPERATURE,
    worse_selection_decay: float = BCO_WORSE_SELECTION_DECAY,
    worse_selection_min_temperature: float = BCO_WORSE_SELECTION_MIN_TEMPERATURE,
    worse_selection_uniform_mix: float = BCO_WORSE_SELECTION_UNIFORM_MIX,
    worse_selection_elite_count: int = BCO_WORSE_SELECTION_ELITE_COUNT,
    trim_grace_period: int = 0,
    process_neural_bees_sequentially: bool = False,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    bee_model_arch: str | None = None,
    bee_model_weights=None,
    force_cpu: bool = False,
):
    """Build a BCO config.

    ``bee_model_arch`` / ``bee_model_weights`` (optional) override the neural
    *rebuild/construction* bee model. By default the type-1 rebuild bee uses the
    construction model (``bestsofar_feb2023``); set
    ``bee_model_arch="bestsofar_feb2023_trim"`` + ``bee_model_weights=<edit ckpt>``
    to drive the rebuild bee with the trim/edit model instead (rebuild via the
    edit model's RL rollout).

    User-facing mutation names:
      heuristic_rebuild      - rebuild the chosen route
      local_endpoint_edit    - local extend/shorten by one stop
      path_mix_rebuild       - random path-combiner rebuild
      construction_extend    - one construction-model extension/halt step
      extend_trim_edit       - one edit-model extend/trim/halt step
      trim_only              - one edit-model trim/halt step
      trim_then_extend       - trim/halt, then construction extend/halt
    """
    run_name = safe_run_name(run_name)
    base_cfg_name = "neural_bco_mumford" if use_neural_bees else "bco_mumford"
    effective_type2 = (
        n_bees - n_type1_bees - n_type4_bees - n_type5_bees - n_type6_bees - n_type7_bees
        if n_type2_bees is None
        else n_type2_bees
    )
    overrides = [
        "+eval=mumford0",
        "++eval.dataset.type=tensor",
        "++experiment.logdir=null",  # no empty TensorBoard run dir for eval runs
        f"++experiment.cpu={str(force_cpu).lower()}",
        f"++eval.n_routes={n_routes}",
        f"++eval.min_route_len={min_route_len}",
        f"++eval.max_route_len={max_route_len}",
        f"++experiment.cost_function.kwargs.demand_time_weight={demand_time_weight}",
        f"++experiment.cost_function.kwargs.route_time_weight={route_time_weight}",
        f"++experiment.cost_function.kwargs.median_connectivity_weight={median_connectivity_weight}",
        f"++experiment.cost_function.kwargs.connectivity_mode={connectivity_mode}",
        f"++run_name={run_name}",
        f"++n_bees={n_bees}",
        f"++n_iterations={BCO_N_ITERATIONS}",
        f"++n_type1_bees={n_type1_bees}",
        f"++n_type2_bees={effective_type2}",
        f"++n_type4_bees={n_type4_bees}",
        f"++n_type5_bees={n_type5_bees}",
        f"++n_type6_bees={n_type6_bees}",
        f"++n_type7_bees={n_type7_bees}",
        f"++ignore_type4_max_route_len={str(ignore_type4_max_route_len).lower()}",
        f"++ignore_type5_max_route_len={str(ignore_type5_max_route_len).lower()}",
        f"++ignore_type6_max_route_len={str(ignore_type6_max_route_len).lower()}",
        f"++ignore_type7_max_route_len={str(ignore_type7_max_route_len).lower()}",
        f"++type4_allow_halt={str(type4_allow_halt).lower()}",
        f"++type5_allow_halt={str(type5_allow_halt).lower()}",
        f"++type6_allow_halt={str(type6_allow_halt).lower()}",
        f"++type7_allow_halt={str(type7_allow_halt).lower()}",
        f"++use_demand_weighted_route_selection={str(use_demand_weighted_route_selection).lower()}",
        f"++worse_accept_temperature={worse_accept_temperature}",
        f"++worse_accept_decay={worse_accept_decay}",
        f"++worse_accept_min_temperature={worse_accept_min_temperature}",
        f"++worse_selection_temperature={worse_selection_temperature}",
        f"++worse_selection_decay={worse_selection_decay}",
        f"++worse_selection_min_temperature={worse_selection_min_temperature}",
        f"++worse_selection_uniform_mix={worse_selection_uniform_mix}",
        f"++worse_selection_elite_count={worse_selection_elite_count}",
        f"++trim_grace_period={trim_grace_period}",
        "++process_neural_bees_sequentially="
        f"{str(process_neural_bees_sequentially).lower()}",
        f"++early_stop_min_delta={float(early_stop_min_delta)}",
    ]
    if bee_model_arch is not None:
        # rebuild/construction bee uses a custom model arch (e.g. our trim edit
        # model) -- compose-time defaults-group override + serial halting.
        overrides = [f"model={bee_model_arch}",
                     "model.route_generator.kwargs.serial_halting=True"] + overrides
    if early_stop_patience is not None:
        overrides.append(f"++early_stop_patience={int(early_stop_patience)}")
    if use_neural_bees:
        _bee_w = bee_model_weights if bee_model_weights is not None else MODEL_WEIGHTS_PATH
        overrides.append(f"+model.weights='{_bee_w}'")
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=base_cfg_name, overrides=overrides)
    cfg.batch_size = 1
    n_type3 = (n_bees - n_type1_bees - effective_type2 - n_type4_bees -
               n_type5_bees - n_type6_bees - n_type7_bees)
    route_selection_label = "weighted" if use_demand_weighted_route_selection else "uniform-random"
    _rebuild_name = "neural_rebuild" if use_neural_bees else "heuristic_rebuild"
    print(
        f"[{run_name}] route_selection={route_selection_label} | "
        f"bee split: {_rebuild_name}={n_type1_bees} "
        f"local_endpoint_edit={effective_type2} path_mix_rebuild={n_type3} "
        f"construction_extend={n_type4_bees} extend_trim_edit={n_type5_bees} "
        f"trim_only={n_type6_bees} trim_then_extend={n_type7_bees} "
        f"(total={n_bees})"
    )
    return _apply_disabled_components_to_cfg(cfg)


def as_route_tensor(routes):
    if isinstance(routes, torch.Tensor):
        return routes.detach().cpu()
    return get_batch_tensor_from_routes(routes).detach().cpu()


def _pad_routes_to_spec(routes, n_routes, max_route_len):
    routes = as_route_tensor(routes).long()
    if routes.ndim == 2:
        routes = routes[None]
    if routes.shape[1] != n_routes:
        raise ValueError(
            f"Expected {n_routes} routes, got tensor shape {tuple(routes.shape)}")
    current_len = routes.shape[-1]
    if current_len > max_route_len:
        raise ValueError(
            f"RPC init produced route tensor length {current_len}, "
            f"but max_route_len={max_route_len}")
    if current_len < max_route_len:
        routes = torch.nn.functional.pad(
            routes, (0, max_route_len - current_len), value=-1)
    return routes


def metric_value(metrics, key, default=np.nan):
    if key not in metrics or metrics[key] is None:
        return default
    value = metrics[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 0:
            return default
        return float(value.float().mean().item())
    return float(value)


def _expand_batch_value(value, batch_size: int, device):
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.float32)
    else:
        value = torch.tensor(value, device=device, dtype=torch.float32)
    if value.numel() == 1:
        value = value.expand(batch_size)
    elif value.shape[0] > batch_size:
        value = value[:batch_size]
    return value


def _cost_weight(state, cost_obj, name: str):
    fallback = getattr(cost_obj, name)
    return _expand_batch_value(state.cost_weights.get(name, fallback), state.batch_size, state.device)


def compute_cost_breakdown(dataloader, eval_cfg, cost_obj, routes_tensor, device):
    data = next(iter(dataloader))
    if device is not None and device.type != "cpu":
        data = data.cuda()
    state = RouteGenBatchState(
        data,
        cost_obj,
        eval_cfg.n_routes,
        eval_cfg.min_route_len,
        eval_cfg.max_route_len,
    )
    routes_tensor = as_route_tensor(routes_tensor).to(state.device)
    state.add_new_routes(routes_tensor)

    cho = cost_obj._cost_helper(state)
    demand_time_weight = _cost_weight(state, cost_obj, "demand_time_weight")
    route_time_weight = _cost_weight(state, cost_obj, "route_time_weight")
    median_connectivity_weight = _cost_weight(state, cost_obj, "median_connectivity_weight")
    constraint_weight = _expand_batch_value(
        getattr(cost_obj, "constraint_violation_weight", 0.0),
        state.batch_size,
        state.device,
    )

    time_normalizer = state.drive_times.flatten(1, 2).max(1).values
    n_routes = state.n_routes_to_plan
    served_demand = cho.total_demand - cho.unserved_demand
    demand_component = (
        cho.mean_demand_time * served_demand + cho.unserved_demand * time_normalizer * 2
    ) / cho.total_demand
    route_component = cho.total_route_time
    connectivity_component = cho.median_connectivity_weighted if cost_obj.use_weighted_connectivity else cho.median_connectivity

    demand_component = demand_component / time_normalizer
    route_component = route_component / (time_normalizer * n_routes + 1e-6)
    connectivity_component = connectivity_component / time_normalizer

    demand_term = demand_component * demand_time_weight
    route_term = route_component * route_time_weight
    connectivity_term = connectivity_component * median_connectivity_weight

    frac_uncovered = cho.n_disconnected_demand_edges / state.n_demand_edges
    penalty_component = frac_uncovered + 0.1 * (frac_uncovered > 0)
    if not cost_obj.ignore_stops_oob:
        denom = n_routes * state.min_route_len
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        frac_stops_oob = cho.n_stops_oob / denom
        penalty_component = penalty_component + frac_stops_oob + 0.1 * (frac_stops_oob > 0)
    penalty_term = penalty_component * constraint_weight
    cost_sum = demand_term + route_term + connectivity_term + penalty_term

    def mean_cpu(x):
        return x.float().mean().detach().cpu()

    # Both demand-weighted WMC variants (modified-Cp), reported alongside the
    # optimized WMC. /60 to match the WMC column scale (get_metrics divides too).
    _wmean = getattr(cho, "mean_weighted_connectivity", None)
    _wmed = getattr(cho, "median_weighted_connectivity", None)
    return {
        "WMC_mean": mean_cpu(_wmean / 60) if _wmean is not None else float("nan"),
        "WMC_median": mean_cpu(_wmed / 60) if _wmed is not None else float("nan"),
        "cost_demand_component": mean_cpu(demand_component),
        "cost_route_component": mean_cpu(route_component),
        "cost_connectivity_component": mean_cpu(connectivity_component),
        "cost_penalty_component": mean_cpu(penalty_component),
        "cost_demand_term": mean_cpu(demand_term),
        "cost_route_term": mean_cpu(route_term),
        "cost_connectivity_term": mean_cpu(connectivity_term),
        "cost_penalty_term": mean_cpu(penalty_term),
        "cost_sum_check": mean_cpu(cost_sum),
        "cost_demand_weight": mean_cpu(demand_time_weight),
        "cost_route_weight": mean_cpu(route_time_weight),
        "cost_connectivity_weight": mean_cpu(median_connectivity_weight),
        "cost_constraint_weight": mean_cpu(constraint_weight),
    }


def add_cost_breakdown_to_metrics(metrics, dataloader, eval_cfg, cost_obj, routes_tensor, device):
    metrics = dict(metrics)
    metrics.update(compute_cost_breakdown(dataloader, eval_cfg, cost_obj, routes_tensor, device))
    return metrics


def run_lc(cfg, init_routes=None, revisit_routes=None, *,
           tensors=None, run_name_prefix="lc_", n_samples=None):
    # tensors=None -> Mumford0 dataloader; tensors=<dict> -> explicit
    # tensor dataset (unifies the former run_lc_on_tensors).
    # n_samples=None -> use the default LC_SAMPLES; pass a smaller integer
    # (e.g. 1) for a single-sample diagnostic run.
    if tensors is None:
        dataloader = make_test_dataloader(cfg.eval.dataset)
    else:
        dataloader = make_tensor_dataloader(cfg.eval.dataset, tensors)
    device, run_name, _, cost_obj, model = lrnu.process_standard_experiment_cfg(
        cfg,
        run_name_prefix=run_name_prefix,
        weights_required=True,
    )
    init_cfg = OmegaConf.create({"method": "tensor"}) if init_routes is not None else None
    if hasattr(model, "clear_step_counts_log"):
        model.clear_step_counts_log()
    effective_n_samples = LC_SAMPLES if n_samples is None else int(n_samples)
    _, unserved_demand, metrics, routes = eval_model(
        model,
        dataloader,
        cfg.eval,
        cost_obj,
        n_samples=effective_n_samples,
        return_routes=True,
        silent=True,
        device=device,
        init_cfg=init_cfg,
        routes_tensor=init_routes,
        revisit_routes_tensor=revisit_routes,
    )
    routes_tensor = as_route_tensor(routes)
    metrics = add_cost_breakdown_to_metrics(metrics, dataloader, cfg.eval, cost_obj, routes_tensor, device)
    step_counts = getattr(model, "route_step_counts_log", [])
    return run_name, metrics, unserved_demand, routes_tensor, list(step_counts)


def build_rpc_routes(spec, tensors, run_name=None, n_samples=1):
    """Generate benchmark initial routes with RPC/pi_random.

    ``spec`` is one row from ``BENCHMARK_SPECS``. The returned tensor matches
    the same ``[1, n_routes, max_route_len]`` contract as the old NX init.
    """
    run_name = run_name or f"rpc_init_{spec['city']}"
    cfg = build_rpc_cfg(
        run_name=run_name,
        n_routes=spec["n_routes"],
        min_route_len=spec["min_route_len"],
        max_route_len=spec["max_route_len"],
    )
    _, _metrics, _unserved, routes, _step_counts = run_lc(
        cfg, tensors=tensors, run_name_prefix="rpc_", n_samples=n_samples)
    return _pad_routes_to_spec(
        routes, spec["n_routes"], spec["max_route_len"])


def build_edit_model(device, weights_path=None, load_weights=True):
    """Load the trim-capable edit model used by edit/trim BCO mutations.

    ``weights_path`` defaults to ``EDIT_MODEL_WEIGHTS_PATH`` but can be pointed
    at any compatible trim-model checkpoint (e.g. a freshly trained
    route+conn(+adj) model from the lc_redundancy notebook).

    ``load_weights=False`` returns a randomly-initialised model of the same
    architecture (untrained policy) -- used for the RL-ablation baseline that
    isolates the value of training vs the structure alone.
    """
    weights_path = weights_path or EDIT_MODEL_WEIGHTS_PATH
    overrides = [
        "model=bestsofar_feb2023_trim",
        "model.route_generator.kwargs.serial_halting=True",
        "++run_name=eval_seeded_edit_model",
        "++experiment.logdir=null",
    ]
    # Checkpoints trained with adjustment conditioning carry 1-2 extra global
    # features; the architecture must match or load_state_dict fails. Set
    # eval_lib.helpers.EDIT_MODEL_N_ADJ_COND_FEATS alongside
    # EDIT_MODEL_WEIGHTS_PATH when pointing at a conditioned checkpoint.
    if EDIT_MODEL_N_ADJ_COND_FEATS:
        overrides.append(
            "++model.route_generator.kwargs.n_adjustment_cond_feats="
            f"{int(EDIT_MODEL_N_ADJ_COND_FEATS)}")
    if load_weights:
        overrides.append(f"+model.weights='{weights_path}'")
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        edit_cfg = compose(config_name="ppo_50nodes.yaml", overrides=overrides)
    edit_model = lrnu.build_model_from_cfg(edit_cfg.model, edit_cfg.experiment)
    if load_weights:
        edit_model.load_state_dict(torch.load(weights_path, map_location=device))
    edit_model.to(device).eval()
    return edit_model


def run_bco(cfg, init_routes, mutation_counts_out=None, *,
            tensors=None, run_name_scope="", cost_history_out=None):
    # tensors=None -> Mumford0 dataloader; tensors=<dict> -> explicit
    # tensor dataset. run_name_scope prefixes the run name so the
    # benchmark paths can keep their dataset/run-name labels.
    # Unifies the former run_bco / run_bco_on_nx_tensors /
    # run_bco_on_tensors trio.
    if tensors is None:
        dataloader = make_test_dataloader(cfg.eval.dataset)
    else:
        dataloader = make_tensor_dataloader(cfg.eval.dataset, tensors)
    use_neural_bees = cfg.get("neural_bees", False)
    prefix = f"{run_name_scope}"
    prefix += "neural_bco_" if use_neural_bees else "bco_"
    device, run_name, _, cost_obj, bee_model = lrnu.process_standard_experiment_cfg(
        cfg,
        run_name_prefix=prefix,
        weights_required=use_neural_bees,
    )
    force_linking_unlinked = cfg.get("force_linking_unlinked", False)
    if not use_neural_bees:
        bee_model = None
    elif bee_model is not None:
        bee_model.force_linking_unlinked = force_linking_unlinked
        bee_model.eval()

    n_type5_bees = int(cfg.get("n_type5_bees", 0))
    n_type6_bees = int(cfg.get("n_type6_bees", 0))
    n_type7_bees = int(cfg.get("n_type7_bees", 0))
    edit_model = build_edit_model(device) if (n_type5_bees > 0 or n_type6_bees > 0 or n_type7_bees > 0) else None
    mutation_counts_out = {} if mutation_counts_out is None else mutation_counts_out

    output = lrnu.test_method(
        bee_colony,
        dataloader,
        cfg.eval,
        OmegaConf.create({"method": "tensor"}),
        cost_obj,
        silent=False,   # show bee_colony's per-iteration tqdm (outer 1-sample bar is auto-hidden)
        device=device,
        return_routes=True,
        return_histories=cost_history_out is not None,
        routes_tensor=init_routes,
        n_bees=cfg.n_bees,
        n_iterations=cfg.n_iterations,
        n_type1_bees=cfg.get("n_type1_bees", None),
        n_type2_bees=cfg.get("n_type2_bees", None),
        n_type4_bees=cfg.get("n_type4_bees", 0),
        n_type5_bees=n_type5_bees,
        n_type6_bees=n_type6_bees,
        n_type7_bees=n_type7_bees,
        bee_model=bee_model,
        edit_model=edit_model,
        force_linking_unlinked=force_linking_unlinked,
        adjustment_degree_weight=cfg.get("adjustment_degree_weight", 0.0),
        adjustment_degree_gap=cfg.get("adjustment_degree_gap", 0.1),
        adjustment_degree_mode=cfg.get("adjustment_degree_mode", "current"),
        adjustment_degree_objective=cfg.get("adjustment_degree_objective", "raw"),
        adjustment_degree_target=cfg.get("adjustment_degree_target", 0.2),
        ignore_type4_max_route_len=cfg.get("ignore_type4_max_route_len", False),
        ignore_type5_max_route_len=cfg.get("ignore_type5_max_route_len", False),
        type4_allow_halt=cfg.get("type4_allow_halt", True),
        type5_allow_halt=cfg.get("type5_allow_halt", True),
        type6_allow_halt=cfg.get("type6_allow_halt", True),
        type7_allow_halt=cfg.get("type7_allow_halt", True),
        ignore_type6_max_route_len=cfg.get("ignore_type6_max_route_len", False),
        ignore_type7_max_route_len=cfg.get("ignore_type7_max_route_len", False),
        use_demand_weighted_route_selection=cfg.get("use_demand_weighted_route_selection", False),
        worse_accept_temperature=cfg.get("worse_accept_temperature", 0.0),
        worse_accept_decay=cfg.get("worse_accept_decay", 0.995),
        worse_accept_min_temperature=cfg.get("worse_accept_min_temperature", 0.001),
        worse_selection_temperature=cfg.get("worse_selection_temperature", 0.0),
        worse_selection_decay=cfg.get("worse_selection_decay", 0.995),
        worse_selection_min_temperature=cfg.get("worse_selection_min_temperature", 0.001),
        worse_selection_uniform_mix=cfg.get("worse_selection_uniform_mix", 0.05),
        worse_selection_elite_count=cfg.get("worse_selection_elite_count", 1),
        trim_grace_period=cfg.get("trim_grace_period", 0),
        process_neural_bees_sequentially=cfg.get(
            "process_neural_bees_sequentially", False),
        early_stop_patience=cfg.get("early_stop_patience", None),
        early_stop_min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
        mutation_counts_out=mutation_counts_out,
    )
    if cost_history_out is not None:
        _, _, unserved_demand, metrics, routes, cost_histories = output
        if cost_histories:
            _h = cost_histories[0]
            cost_history_out["history"] = (
                _h.detach().cpu().clone() if hasattr(_h, "detach") else _h)
    else:
        _, _, unserved_demand, metrics, routes = output
    routes_tensor = as_route_tensor(routes)
    metrics = add_cost_breakdown_to_metrics(metrics, dataloader, cfg.eval, cost_obj, routes_tensor, device)
    return run_name, metrics, unserved_demand, routes_tensor, mutation_counts_out


def build_default_bco_variants():
    return [
        {
            "key": "seeded_heuristic",
            "summary_label": "BCO",
            "run_name": "seeded_bco_heuristic_from_lc_mumford0",
            "use_neural_bees": False,
            "n_type1_bees": BCO_N_TYPE1_BEES,
            "n_type2_bees": None,
            "n_type4_bees": 0,
            "n_type5_bees": 0,
            "n_type6_bees": 0,
            "n_type7_bees": 0,
        },
        {
            # Same bee split as seeded_heuristic but the type-1 rebuild bees
            # use the neural route-construction model (the old "neural BCO").
            "key": "seeded_neural",
            "summary_label": "neural BCO",
            "run_name": "seeded_neural_bco_from_lc_mumford0",
            "use_neural_bees": True,
            "n_type1_bees": BCO_N_TYPE1_BEES,
            "n_type2_bees": None,
            "n_type4_bees": 0,
            "n_type5_bees": 0,
            "n_type6_bees": 0,
            "n_type7_bees": 0,
        },
        {
            "key": "heuristic_extend_trim_split_5_5",
            "summary_label": "Heuristic rebuild + extend/trim edit BCO (5+5)",
            "run_name": "seeded_bco_heuristic_extend_trim_split_5_5_from_lc_mumford0",
            "use_neural_bees": False,
            "n_type1_bees": BCO_N_TYPE1_BEES,
            "n_type2_bees": 0,
            "n_type4_bees": 0,
            "n_type5_bees": BCO_N_BEES - BCO_N_TYPE1_BEES,
            "n_type6_bees": 0,
            "n_type7_bees": 0,
        },
        # {
        #     "key": "construction_only",
        #     "summary_label": f"Construction-only BCO (all {BCO_N_BEES} bees)",
        #     "run_name": "seeded_bco_construction_only_from_lc_mumford0",
        #     "use_neural_bees": True,
        #     "n_type1_bees": 0,
        #     "n_type2_bees": 0,
        #     "n_type4_bees": BCO_N_BEES,
        #     "n_type5_bees": 0,
        #     "n_type6_bees": 0,
        #     "n_type7_bees": 0,
        # },
        {
            "key": "extend_trim_edit_only",
            "summary_label": f"Extend/trim edit-only BCO (all {BCO_N_BEES} bees)",
            "run_name": "seeded_bco_extend_trim_edit_only_from_lc_mumford0",
            "use_neural_bees": False,
            "n_type1_bees": 0,
            "n_type2_bees": 0,
            "n_type4_bees": 0,
            "n_type5_bees": BCO_N_BEES,
            "n_type6_bees": 0,
            "n_type7_bees": 0,
        },
        # {
        #     "key": "trim_then_extend_only",
        #     "summary_label": f"Trim-then-extend BCO (all {BCO_N_BEES} bees)",
        #     "run_name": "seeded_bco_trim_then_extend_only_from_lc_mumford0",
        #     "use_neural_bees": True,
        #     "n_type1_bees": 0,
        #     "n_type2_bees": 0,
        #     "n_type4_bees": 0,
        #     "n_type5_bees": 0,
        #     "n_type6_bees": 0,
        #     "n_type7_bees": BCO_N_BEES,
        # },
        {
            # "Construct" + trim, but BOTH bees use the trained LC-improvement
            # edit_model -- type5 (extend/trim/halt action set) for the
            # "construct" half, type6 (trim/halt only) for the trim half.
            # type4 (bestsofar_feb2023 construction model) is NOT used here.
            "key": "construction_trim_split_5_5",
            "summary_label": "Construction + trim BCO (5+5)",
            "run_name": "seeded_bco_construction_trim_split_5_5_from_lc_mumford0",
            "use_neural_bees": False,
            "n_type1_bees": 0,
            "n_type2_bees": 0,
            "n_type4_bees": 0,
            "n_type5_bees": 5,
            "n_type6_bees": 5,
            "n_type7_bees": 0,
        },
        {
            # Same idea as construction_trim_split_5_5 but with an 8+2 split:
            # 8 type5 (edit-model extend/trim/halt) + 2 type6 (edit-model
            # trim/halt). No bestsofar_feb2023 construction bee.
            "key": "construction_trim_split_8_2",
            "summary_label": "Construction + trim BCO (8+2)",
            "run_name": "seeded_bco_construction_trim_split_8_2_from_lc_mumford0",
            "use_neural_bees": False,
            "n_type1_bees": 0,
            "n_type2_bees": 0,
            "n_type4_bees": 0,
            "n_type5_bees": 8,
            "n_type6_bees": 2,
            "n_type7_bees": 0,
        },
        {
            "key": "trim12_extend12",
            "summary_label": "Trim 12 + extend 12 BCO",
            "run_name": "seeded_bco_trim12_extend12_from_lc_mumford0",
            "use_neural_bees": True,
            "n_bees": 24,
            "n_type1_bees": 0,
            "n_type2_bees": 0,
            "n_type4_bees": 12,
            "n_type5_bees": 0,
            "n_type6_bees": 12,
            "n_type7_bees": 0,
        },
        # Note: Figure-5-style ablation variants (neural construction vs
        # random-path-combiner construction, both paired with trained edit/trim
        # bees) are intentionally NOT in the default list -- they exist only
        # for the Pareto-alpha experiment in pareto_alpha_sweep.ipynb and
        # would otherwise pollute the §12 benchmark sweep / MACSA / worse-accept
        # tables with rows that are not meaningful outside the alpha sweep.
        # See pareto_alpha_sweep.ipynb config cell for the inline definitions.
    ]


BCO_VARIANTS = build_default_bco_variants()


BCO_VARIANTS
