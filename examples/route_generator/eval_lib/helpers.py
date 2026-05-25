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


BCO_ROUTE_SELECTION_MODES = {
    "uniform": {
        "label": "uniform random",
        "use_demand_weighted_route_selection": False,
    },
}

MUTATION_TYPES = (
    {"key": "type1", "slug": "heuristic_rebuild", "label": "heuristic rebuild"},
    {"key": "type2", "slug": "local_endpoint_edit", "label": "local endpoint edit"},
    {"key": "type3", "slug": "path_mix_rebuild", "label": "path mix rebuild"},
    {"key": "type4", "slug": "construction_extend", "label": "construction extend"},
    {"key": "type5", "slug": "extend_trim_edit", "label": "extend/trim edit"},
    {"key": "type6", "slug": "trim_only", "label": "trim only"},
    {"key": "type7", "slug": "trim_then_extend", "label": "trim then extend"},
)
MUTATION_TYPE_KEYS = tuple(item["key"] for item in MUTATION_TYPES)
MUTATION_TYPE_SLUGS = {item["key"]: item["slug"] for item in MUTATION_TYPES}
MUTATION_TYPE_LABELS = {item["key"]: item["label"] for item in MUTATION_TYPES}


def mutation_slug(type_name: str) -> str:
    return MUTATION_TYPE_SLUGS[type_name]


def mutation_label(type_name: str) -> str:
    return MUTATION_TYPE_LABELS[type_name]


def route_selection_mode_label(mode_key: str) -> str:
    return BCO_ROUTE_SELECTION_MODES[mode_key]["label"]


def route_selection_mode_suffix(mode_key: str) -> str:
    return mode_key


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
        "run_name": run_name,
        "model_weights": str(MODEL_WEIGHTS_PATH),
    }
    cfg = get_eval_cfg(str(CFG_DIR), "eval_model_mumford", params)
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
    use_demand_weighted_route_selection: bool = False,
    demand_time_weight: float = DEMAND_TIME_WEIGHT,
    route_time_weight: float = ROUTE_TIME_WEIGHT,
    median_connectivity_weight: float = MEDIAN_CONNECTIVITY_WEIGHT,
    worse_accept_temperature: float = BCO_WORSE_ACCEPT_TEMPERATURE,
    worse_accept_decay: float = BCO_WORSE_ACCEPT_DECAY,
    worse_accept_min_temperature: float = BCO_WORSE_ACCEPT_MIN_TEMPERATURE,
    worse_selection_temperature: float = BCO_WORSE_SELECTION_TEMPERATURE,
    worse_selection_decay: float = BCO_WORSE_SELECTION_DECAY,
    worse_selection_min_temperature: float = BCO_WORSE_SELECTION_MIN_TEMPERATURE,
    worse_selection_uniform_mix: float = BCO_WORSE_SELECTION_UNIFORM_MIX,
    worse_selection_elite_count: int = BCO_WORSE_SELECTION_ELITE_COUNT,
    trim_grace_period: int = 0,
):
    """Build a BCO config.

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
        f"++eval.n_routes={n_routes}",
        f"++eval.min_route_len={min_route_len}",
        f"++eval.max_route_len={max_route_len}",
        f"++experiment.cost_function.kwargs.demand_time_weight={demand_time_weight}",
        f"++experiment.cost_function.kwargs.route_time_weight={route_time_weight}",
        f"++experiment.cost_function.kwargs.median_connectivity_weight={median_connectivity_weight}",
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
    ]
    if use_neural_bees:
        overrides.append(f"+model.weights='{MODEL_WEIGHTS_PATH}'")
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=base_cfg_name, overrides=overrides)
    cfg.batch_size = 1
    n_type3 = (n_bees - n_type1_bees - effective_type2 - n_type4_bees -
               n_type5_bees - n_type6_bees - n_type7_bees)
    route_selection_label = "weighted" if use_demand_weighted_route_selection else "uniform-random"
    print(
        f"[{run_name}] route_selection={route_selection_label} | "
        f"bee split: heuristic_rebuild={n_type1_bees} "
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

    return {
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


def format_cost_terms(metrics):
    return (
        f"cost={metric_value(metrics, 'cost'):.4f} "
        f"[d={metric_value(metrics, 'cost_demand_term'):.3f}, "
        f"r={metric_value(metrics, 'cost_route_term'):.3f}, "
        f"c={metric_value(metrics, 'cost_connectivity_term'):.3f}, "
        f"pen={metric_value(metrics, 'cost_penalty_term'):.3f}]"
    )


def format_absolute_metrics(metrics):
    mc_value = metric_value(metrics, "median_connectivity", default=np.nan)
    mc_part = f", MC={mc_value:.3f}" if np.isfinite(mc_value) else ""
    return (
        f"ATT={metric_value(metrics, 'ATT'):.3f}, "
        f"RTT={metric_value(metrics, 'RTT'):.3f}{mc_part}, "
        f"d_un={metric_value(metrics, '$d_{un}$'):.2f}%"
    )


def run_lc(cfg, init_routes=None, revisit_routes=None, *,
           tensors=None, run_name_prefix="lc_"):
    # tensors=None -> Mumford0 dataloader; tensors=<dict> -> explicit
    # tensor dataset (unifies the former run_lc_on_tensors).
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
    _, unserved_demand, metrics, routes = eval_model(
        model,
        dataloader,
        cfg.eval,
        cost_obj,
        n_samples=LC_SAMPLES,
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


def build_edit_model(device):
    """Load the trim-capable edit model used by edit/trim BCO mutations."""
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        edit_cfg = compose(
            config_name="ppo_50nodes.yaml",
            overrides=[
                "model=bestsofar_feb2023_trim",
                "model.route_generator.kwargs.serial_halting=True",
                "++run_name=eval_seeded_edit_model",
                "++experiment.logdir=null",
                f"+model.weights='{EDIT_MODEL_WEIGHTS_PATH}'",
            ],
        )
    edit_model = lrnu.build_model_from_cfg(edit_cfg.model, edit_cfg.experiment)
    edit_model.load_state_dict(
        torch.load(EDIT_MODEL_WEIGHTS_PATH, map_location=device)
    )
    edit_model.to(device).eval()
    return edit_model


def run_rl_improvement(
    init_routes,
    run_name: str = RL_IMPROVEMENT_RUN_NAME,
    n_routes: int = N_ROUTES,
    min_route_len: int = MIN_ROUTE_LEN,
    max_route_len: int = MAX_ROUTE_LEN,
    max_route_edit_steps: int | None = RL_MAX_ROUTE_EDIT_STEPS,
    max_trim_actions_per_route: int | None = RL_MAX_TRIM_ACTIONS_PER_ROUTE,
    force_nonhalt_first_step: bool = RL_FORCE_NONHALT_FIRST_STEP,
    return_best_routes: bool = RL_RETURN_BEST_ROUTES,
    *,
    tensors=None,
    weights=None,
):
    """Run the trained edit/RL agent once over all seeded routes, without BCO.

    tensors=None -> Mumford0 dataloader; tensors=<dict> -> explicit tensor
    dataset. weights=<dict with demand/route/connectivity weights> overrides
    the cost-objective weights (used for the objective-weight sweep).
    """
    cfg = build_bco_cfg(
        run_name=run_name,
        n_routes=n_routes,
        min_route_len=min_route_len,
        max_route_len=max_route_len,
        use_neural_bees=False,
        n_type1_bees=0,
        n_type2_bees=0,
        n_type4_bees=0,
        n_type5_bees=0,
        n_type6_bees=0,
        n_type7_bees=0,
        **(weights or {}),
    )

    def _make_loader():
        if tensors is None:
            return make_test_dataloader(cfg.eval.dataset)
        return make_tensor_dataloader(cfg.eval.dataset, tensors)

    dataloader = _make_loader()
    device, resolved_run_name, _, cost_obj, _ = lrnu.process_standard_experiment_cfg(
        cfg,
        run_name_prefix="rl_improvement_",
        weights_required=False,
    )
    edit_model = build_edit_model(device)
    graph_batch = next(iter(dataloader))
    if device is not None and device.type != "cpu":
        graph_batch = graph_batch.cuda()
    route_batch = as_route_tensor(init_routes).to(device=device, dtype=torch.long)
    if route_batch.ndim == 2:
        route_batch = route_batch.unsqueeze(0)
    eval_weights = cost_obj.get_weights(device)
    # Pure-evaluation rollout: run under no_grad. Without it PyTorch builds a
    # full autograd graph -- every edit step over every route keeps its GNN /
    # attention activations alive for a backward pass that never happens. On
    # the large benchmark graphs (Mumford2 / Mumford3: ~60 routes over many
    # nodes) that autograd graph is what OOMs. The batch is already a single
    # graph (cfg.batch_size = 1), so shrinking the batch would not help -- the
    # memory is the per-graph rollout graph, not multiple graphs.
    with torch.no_grad():
        rollout_output = rollout_lc_improvement(
            edit_model,
            cost_obj,
            graph_batch,
            route_batch,
            cfg.eval.min_route_len,
            cfg.eval.max_route_len,
            greedy=True,
            cost_weights=eval_weights,
            return_actions=True,
            force_nonhalt_first_step=force_nonhalt_first_step,
            max_route_edit_steps=max_route_edit_steps,
            max_trim_actions_per_route=max_trim_actions_per_route,
            return_best_routes=return_best_routes,
        )
    final_state = rollout_output[0]
    routes_tensor = as_route_tensor(final_state.routes)
    # rollout_output[5:7] = (route_actions, route_action_kinds) when
    # return_actions=True -- feed them into the shared action-stat summary so
    # the RunResult carries the ext/trim counters the route figures annotate.
    action_stats = summarize_route_action_stats(
        rollout_output[5], rollout_output[6])

    eval_dataloader = _make_loader()
    output = lrnu.test_method(
        None,
        eval_dataloader,
        cfg.eval,
        OmegaConf.create({"method": "tensor"}),
        cost_obj,
        silent=True,
        device=device,
        return_routes=True,
        routes_tensor=routes_tensor,
    )
    _, _, unserved_demand, metrics, eval_routes = output
    routes_tensor = as_route_tensor(eval_routes)
    metrics = add_cost_breakdown_to_metrics(metrics, eval_dataloader, cfg.eval, cost_obj, routes_tensor, device)
    return (resolved_run_name, metrics, unserved_demand, routes_tensor, cfg,
            action_stats)


def run_bco(cfg, init_routes, mutation_counts_out=None, *,
            tensors=None, run_name_scope=""):
    # tensors=None -> Mumford0 dataloader; tensors=<dict> -> explicit
    # tensor dataset. run_name_scope prefixes the run name so the
    # NX paths can keep their "nx_dataset_" / "dataset_" labels.
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
        silent=True,
        device=device,
        return_routes=True,
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
        mutation_counts_out=mutation_counts_out,
    )
    _, _, unserved_demand, metrics, routes = output
    routes_tensor = as_route_tensor(routes)
    metrics = add_cost_breakdown_to_metrics(metrics, dataloader, cfg.eval, cost_obj, routes_tensor, device)
    return run_name, metrics, unserved_demand, routes_tensor, mutation_counts_out


def summarize_run(label: str, metrics: dict, routes):
    routes = as_route_tensor(routes)
    n_nonempty_routes = int(((routes > -1).sum(dim=-1) > 1).sum().item())
    return {
        "method": label,
        "cost": metric_value(metrics, "cost"),
        "cost_demand_term": metric_value(metrics, "cost_demand_term"),
        "cost_route_term": metric_value(metrics, "cost_route_term"),
        "cost_connectivity_term": metric_value(metrics, "cost_connectivity_term"),
        "cost_penalty_term": metric_value(metrics, "cost_penalty_term"),
        "ATT": metric_value(metrics, "ATT"),
        "RTT": metric_value(metrics, "RTT"),
        "median_connectivity": metric_value(metrics, "median_connectivity"),
        "$d_{un}$": metric_value(metrics, "$d_{un}$"),
        "# disconnected node pairs": metric_value(metrics, "# disconnected node pairs"),
        "# stops out of bounds": metric_value(metrics, "# stops out of bounds"),
        "# routes": n_nonempty_routes,
    }


def mutation_stat_bucket(mutation_stats: dict, bucket: str) -> dict:
    nested = mutation_stats.get(bucket, {}) if mutation_stats else {}
    if bucket == "attempted" and not nested:
        nested = {name: mutation_stats.get(name, 0) for name in MUTATION_TYPE_KEYS} if mutation_stats else {}
    return {name: int(nested.get(name, 0)) for name in MUTATION_TYPE_KEYS}


def selection_stat_bucket(mutation_stats: dict) -> dict:
    nested = mutation_stats.get("selection", {}) if mutation_stats else {}
    return {
        "parent_copies": int(nested.get("parent_copies", 0)),
        "nonbest_parent_copies": int(nested.get("nonbest_parent_copies", 0)),
        "worse_parent_copies": int(nested.get("worse_parent_copies", 0)),
        "trim_grace_forced_accepts": int(nested.get("trim_grace_forced_accepts", 0)),
        "trim_grace_protected": int(nested.get("trim_grace_protected", 0)),
    }


def summarize_mutation_stats(mutation_stats: dict) -> dict:
    return {
        "attempted": {mutation_slug(k): v for k, v in mutation_stat_bucket(mutation_stats, "attempted").items()},
        "accepted": {mutation_slug(k): v for k, v in mutation_stat_bucket(mutation_stats, "accepted").items()},
        "worse_accepted": {mutation_slug(k): v for k, v in mutation_stat_bucket(mutation_stats, "worse_accepted").items()},
        "selection": selection_stat_bucket(mutation_stats),
    }


def plot_mutation_histogram(mutation_stats, title, ax=None):
    """Unified per-mutation-type histogram for one BCO run.

    Shows attempted / accepted / worse-accepted counts per mutation type and
    annotates trim-grace activity in the title when present. ``mutation_stats``
    is either a raw mutation_counts dict from one BCO run or a seed-averaged
    dict from ``aggregate_mutation_stats`` (both work — the buckets are read
    through mutation_stat_bucket / selection_stat_bucket).
    """
    attempted = mutation_stat_bucket(mutation_stats, "attempted")
    accepted = mutation_stat_bucket(mutation_stats, "accepted")
    worse_accepted = mutation_stat_bucket(mutation_stats, "worse_accepted")
    selection = selection_stat_bucket(mutation_stats)
    forced = selection.get("trim_grace_forced_accepts", 0)
    protected = selection.get("trim_grace_protected", 0)

    show = ax is None
    if show:
        fig, ax = plt.subplots(figsize=(6.5, 4))

    labels = list(MUTATION_TYPE_KEYS)
    display_labels = [mutation_label(type_name) for type_name in labels]
    xs = np.arange(len(labels))
    series = [
        ("attempted", attempted, "tab:blue", 0.75),
        ("accepted", accepted, "tab:orange", 0.9),
    ]
    if any(worse_accepted.values()):
        series.append(("worse accepted", worse_accepted, "tab:red", 0.85))

    width = min(0.8 / len(series), 0.32)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    all_vals = []
    bar_groups = []
    for offset, (label, values, color, alpha) in zip(offsets, series):
        vals = [values[type_name] for type_name in labels]
        all_vals.extend(vals)
        bar_groups.append(
            ax.bar(xs + offset, vals, width, color=color, alpha=alpha,
                   label=label)
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    full_title = title
    if forced or protected:
        full_title = (f"{title}\ntrim-grace: forced-accepts={forced:g}, "
                      f"protected={protected:g}")
    ax.set_title(full_title)
    ax.set_ylabel("count")
    ax.legend()

    ymax = max(all_vals + [1])
    for bars in bar_groups:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + ymax * 0.015,
                    f"{height:.0f}",
                    ha="center", va="bottom", fontsize=8)

    if show:
        plt.tight_layout()
        plt.show()


def aggregate_mutation_stats(mutation_counts_list):
    """Seed-average a list of per-run mutation_counts dicts into one
    normalized dict that plot_mutation_histogram can render."""
    runs = [mc for mc in mutation_counts_list if mc]
    n = max(len(runs), 1)
    agg = {"attempted": {}, "accepted": {}, "worse_accepted": {},
           "selection": {}}
    for bucket in ("attempted", "accepted", "worse_accepted"):
        for type_name in MUTATION_TYPE_KEYS:
            agg[bucket][type_name] = sum(
                mutation_stat_bucket(mc, bucket)[type_name]
                for mc in runs) / n
    for key in ("parent_copies", "nonbest_parent_copies",
                "worse_parent_copies", "trim_grace_forced_accepts",
                "trim_grace_protected"):
        agg["selection"][key] = sum(
            selection_stat_bucket(mc).get(key, 0) for mc in runs) / n
    return agg


def print_lc_result(run_name: str, metrics: dict, routes, step_counts=None):
    routes_tensor = as_route_tensor(routes)
    print(f"Run name: {run_name}")
    print(format_cost_terms(metrics))
    print(format_absolute_metrics(metrics))
    print(f"Routes shape: {tuple(routes_tensor.shape)}")
    if step_counts is not None:
        print(f"Step counts: {step_counts}")


def route_selection_run_name(base_run_name: str, mode_key: str) -> str:
    return f"{base_run_name}_{route_selection_mode_suffix(mode_key)}"


BCO_N_CONSTRUCTION_BEES = 3
BCO_N_EDIT_BEES = 3
BCO_N_TRIM_BEES = 3
BCO_N_TRIM_EXTEND_BEES = 3


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
    ]


BCO_VARIANTS = build_default_bco_variants()

DEFAULT_BCO_PASSES_PER_IT = 5
DEFAULT_BCO_MOD_STEPS_PER_PASS = 2
MUTATION_ATTEMPTS_PER_BEE = (
    BCO_N_ITERATIONS * DEFAULT_BCO_PASSES_PER_IT * DEFAULT_BCO_MOD_STEPS_PER_PASS
)
DEFAULT_SEED_SWEEP_CONSTRUCTION_BEES = BCO_N_CONSTRUCTION_BEES


def target_attempts_to_bees(target_attempts: int, attempts_per_bee: int = MUTATION_ATTEMPTS_PER_BEE) -> int:
    import math

    lower = max(1, target_attempts // attempts_per_bee)
    upper = max(1, math.ceil(target_attempts / attempts_per_bee))
    candidates = sorted({lower, upper})
    return min(
        candidates,
        key=lambda bee_count: (
            abs(bee_count * attempts_per_bee - target_attempts),
            bee_count,
        ),
    )


def run_lc_base():
    cfg = build_lc_cfg(
        run_name="lc_base_mumford0_10r_len12",
        n_routes=N_ROUTES,
        min_route_len=MIN_ROUTE_LEN,
        max_route_len=MAX_ROUTE_LEN,
    )
    run_name, metrics, unserved, routes, step_counts = run_lc(cfg)
    dump_routes(run_name, routes.cpu(), out_dir=OUTPUT_ROUTES_DIR)
    return {
        "cfg": cfg,
        "run_name": run_name,
        "metrics": metrics,
        "unserved_demand": unserved,
        "routes": routes,
        "step_counts": step_counts,
    }


def run_lc_seeded(seed_routes):
    cfg = build_lc_cfg(
        run_name="lc_seeded_second_pass_mumford0_10r_len12",
        n_routes=N_ROUTES,
        min_route_len=MIN_ROUTE_LEN,
        max_route_len=MAX_ROUTE_LEN,
    )
    run_name, metrics, unserved, routes, step_counts = run_lc(
        cfg,
        revisit_routes=seed_routes,
    )
    dump_routes(run_name, routes.cpu(), out_dir=OUTPUT_ROUTES_DIR)
    return {
        "cfg": cfg,
        "run_name": run_name,
        "metrics": metrics,
        "unserved_demand": unserved,
        "routes": routes,
        "step_counts": step_counts,
    }


def run_seed_sweep(
    init_routes,
    mode_key="uniform",
    seeds=None,
    target_heuristic_rebuild_attempts=1000,
    target_local_endpoint_edit_attempts=1000,
    construction_bees=None,
):
    if seeds is None:
        seeds = list(range(1))
    if construction_bees is None:
        construction_bees = DEFAULT_SEED_SWEEP_CONSTRUCTION_BEES

    n_type1_bees = target_attempts_to_bees(target_heuristic_rebuild_attempts)
    n_type2_bees = target_attempts_to_bees(target_local_endpoint_edit_attempts)
    n_bees = n_type1_bees + n_type2_bees + construction_bees
    mode_cfg = BCO_ROUTE_SELECTION_MODES[mode_key]

    rows = []
    results = {}
    for seed in [int(seed) for seed in seeds]:
        cfg = build_bco_cfg(
            run_name=route_selection_run_name(
                f"seeded_bco_seed_sweep_rebuild_{target_heuristic_rebuild_attempts}_local_{target_local_endpoint_edit_attempts}_seed{seed}",
                mode_key,
            ),
            n_routes=N_ROUTES,
            min_route_len=MIN_ROUTE_LEN,
            max_route_len=MAX_ROUTE_LEN,
            use_neural_bees=True,
            n_bees=n_bees,
            n_type1_bees=n_type1_bees,
            n_type2_bees=n_type2_bees,
            n_type4_bees=construction_bees,
            ignore_type4_max_route_len=BCO_CONSTRUCTION_IGNORE_MAX_ROUTE_LEN,
            use_demand_weighted_route_selection=mode_cfg["use_demand_weighted_route_selection"],
        )
        cfg.experiment.seed = seed
        run_name, metrics, unserved_demand, routes, mutation_stats = run_bco(
            cfg,
            init_routes,
        )
        attempted = mutation_stat_bucket(mutation_stats, "attempted")
        accepted = mutation_stat_bucket(mutation_stats, "accepted")
        row = summarize_run(
            f"seed={seed} [{mode_cfg['label']}]",
            metrics,
            routes,
        )
        row.update({
            "seed": seed,
            "route_selection": mode_cfg["label"],
            "target_heuristic_rebuild_attempts": target_heuristic_rebuild_attempts,
            "target_local_endpoint_edit_attempts": target_local_endpoint_edit_attempts,
            "heuristic_rebuild_bees": n_type1_bees,
            "local_endpoint_edit_bees": n_type2_bees,
            "construction_extend_bees": construction_bees,
            "total_bees": n_bees,
            "attempted_heuristic_rebuild": attempted["type1"],
            "attempted_local_endpoint_edit": attempted["type2"],
            "attempted_construction_extend": attempted["type4"],
            "accepted_heuristic_rebuild": accepted["type1"],
            "accepted_local_endpoint_edit": accepted["type2"],
            "accepted_path_mix_rebuild": accepted["type3"],
            "accepted_construction_extend": accepted["type4"],
            "accepted_total": sum(accepted.values()),
        })
        rows.append(row)
        results[seed] = {
            "run_name": run_name,
            "metrics": metrics,
            "unserved_demand": unserved_demand,
            "routes": routes,
            "routes_tensor": routes,
            "mutation_stats": mutation_stats,
        }

    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary_cols = _plots.filter_component_columns([
        "cost",
        "cost_demand_term",
        "cost_route_term",
        "cost_connectivity_term",
        "cost_penalty_term",
        "ATT",
        "RTT",
        "$d_{un}$",
        "# routes",
        "accepted_heuristic_rebuild",
        "accepted_local_endpoint_edit",
        "accepted_path_mix_rebuild",
        "accepted_construction_extend",
        "accepted_total",
    ], ENABLED_COST_COMPONENTS)
    from .tables import TABLE_DECIMALS as _TABLE_DECIMALS
    summary_df = pd.DataFrame({
        "mean": df[summary_cols].mean(),
        "std": df[summary_cols].std(ddof=0),
        "min": df[summary_cols].min(),
        "max": df[summary_cols].max(),
    }).round(_TABLE_DECIMALS)
    return {
        "mode_key": mode_key,
        "mode_label": mode_cfg["label"],
        "seeds": list(seeds),
        "target_heuristic_rebuild_attempts": target_heuristic_rebuild_attempts,
        "target_local_endpoint_edit_attempts": target_local_endpoint_edit_attempts,
        "construction_bees": construction_bees,
        "n_type1_bees": n_type1_bees,
        "n_type2_bees": n_type2_bees,
        "n_bees": n_bees,
        "rows": rows,
        "results": results,
        "df": df,
        "summary_df": summary_df,
    }


BCO_VARIANTS
