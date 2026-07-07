"""Helper functions for the route-evaluation notebook.

Extracted verbatim from the notebook's "Helper Functions" cell so the notebook
stays readable. Free names the helpers used to resolve against the notebook
namespace are provided here: repo paths via :mod:`eval_lib.context`, the unified
objective read from the single YAML source via
``connectpt.routes_generator.objectives.load_unified_objective``, and
route-plotting helpers via :mod:`eval_lib.plots`. Runtime knobs (edit-model
checkpoint, output prefix) are NOT module state anymore -- they travel
explicitly via :class:`eval_lib.run_context.RunContext`.
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
from connectpt.routes_generator.search.cfg_run import run_bco_from_cfg
from connectpt.routes_generator.search.compat.bco_config import (
    compose_bco_cfg as build_bco_cfg, safe_run_name,
    apply_disabled_components as _apply_disabled_components_to_cfg)
from connectpt.routes_generator.torch_utils import (
    dump_routes, get_batch_tensor_from_routes)
from connectpt.routes_generator.data.routes import as_route_tensor
import connectpt.routes_generator.utils as lrnu

from .context import (ROOT_DIR, CFG_DIR, BENCHMARK_DIR, MACSA_DATA_DIR,
                      MODEL_WEIGHTS_PATH, EDIT_MODEL_WEIGHTS_PATH,
                      OUTPUT_ROUTES_DIR)
# Unified objective read from the single source (cfg/objective YAML) via the
# library factory -- no eval_lib.params constants pile. The builders below take
# these as default args (they are transitional: build_bco_cfg / build_lc_cfg get
# replaced by BeeColonySearchRun in the config-first migration).
from connectpt.routes_generator.objectives import (
    load_bco_algo_config as bco_config, load_unified_objective as _load_objective)
from . import plots as _plots

_OBJ = _load_objective()
DEMAND_TIME_WEIGHT = _OBJ.demand_time_weight
ROUTE_TIME_WEIGHT = _OBJ.route_time_weight
MEDIAN_CONNECTIVITY_WEIGHT = _OBJ.median_connectivity_weight
DISABLED_COST_COMPONENTS = list(_OBJ.disabled_components)
# Non-objective run defaults (were eval_lib.params knobs, not part of the cost).
LC_SAMPLES = 100
# FROZEN-NOTEBOOK COMPAT: referenced by evaluation.ipynb; not used by new code.
USE_NEURAL_BCO = False


def make_test_dataloader(dataset_cfg):
    """Dataloader for a dataset described entirely by ``dataset_cfg``.

    For explicit in-memory tensors use :func:`make_tensor_dataloader`.
    """
    dataset = get_dataset_from_config(dataset_cfg)
    return DataLoader(dataset, batch_size=1)


def make_tensor_dataloader(dataset_cfg, tensors):
    dataset = get_dataset_from_config(dataset_cfg, tensors=tensors)
    return DataLoader(dataset, batch_size=1)


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


# Cost breakdown now lives in the library (evaluation.cost_breakdown); re-exported
# here so the notebook / eval_lib callers keep the same names.
from connectpt.routes_generator.evaluation.cost_breakdown import (  # noqa: E402
    add_cost_breakdown_to_metrics, compute_cost_breakdown)


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


def run_lc_batch(cfg, tensors_list, *, run_name_prefix="lc_", n_samples=None,
                 batch_size=None):
    """Batched learned-construction over many graphs (one GPU forward per batch).

    Unlike :func:`run_lc` (one graph, ``batch_size=1``), this builds one
    ``CityGraphData`` per tensors dict, batches them through a single
    ``DataLoader``, and runs the construction model on the whole batch -- so the
    GPU does K graphs at once instead of K launch-overhead-bound singletons.

    ``tensors_list`` is a list of ``{node_locs, street_adj, demand}`` dicts (all
    with the same node count, so they collate into one batch). Returns a route
    tensor ``[K, n_routes, max_len]`` in input order.
    """
    from connectpt.routes_generator.citygraph_dataset import (
        get_dataset_from_config)
    device, run_name, _, cost_obj, model = lrnu.process_standard_experiment_cfg(
        cfg, run_name_prefix=run_name_prefix, weights_required=True)
    if hasattr(model, "clear_step_counts_log"):
        model.clear_step_counts_log()
    datalist = []
    for tn in tensors_list:
        ds = get_dataset_from_config(cfg.eval.dataset, tensors=tn)
        datalist.append(ds[0] if isinstance(ds, (list, tuple)) else ds)
    bs = int(batch_size or len(datalist))
    dataloader = DataLoader(datalist, batch_size=bs)
    effective_n_samples = LC_SAMPLES if n_samples is None else int(n_samples)
    _, _unserved, _metrics, routes = eval_model(
        model, dataloader, cfg.eval, cost_obj,
        n_samples=effective_n_samples, sample_batch_size=bs,
        return_routes=True, silent=True, device=device)
    return as_route_tensor(routes)


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


def run_bco(cfg, init_routes, mutation_counts_out=None, *,
            tensors=None, run_name_scope="", cost_history_out=None,
            iteration_callback=None):
    """FROZEN-NOTEBOOK COMPAT (experiment.ipynb) -- do not use in new code.

    Thin wrapper over the library-owned
    :func:`connectpt.routes_generator.search.cfg_run.run_bco_from_cfg`, pinned
    to the legacy default edit checkpoint
    (``eval_lib.context.EDIT_MODEL_WEIGHTS_PATH``). New code calls
    ``run_bco_from_cfg`` directly with an explicit ``edit_weights_path``
    (usually ``RunContext.edit_weights_path``).
    """
    return run_bco_from_cfg(
        cfg, init_routes, tensors,
        mutation_counts_out=mutation_counts_out, run_name_scope=run_name_scope,
        cost_history_out=cost_history_out, iteration_callback=iteration_callback,
        edit_weights_path=EDIT_MODEL_WEIGHTS_PATH,
        edit_n_adjustment_cond_feats=0)


def build_default_bco_variants():
    bco = bco_config()
    n_bees, n_type1 = int(bco.n_bees), int(bco.n_type1_bees)
    return [
        {
            "key": "seeded_heuristic",
            "summary_label": "BCO",
            "run_name": "seeded_bco_heuristic_from_lc_mumford0",
            "use_neural_bees": False,
            "n_type1_bees": n_type1,
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
            "n_type1_bees": n_type1,
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
            "n_type1_bees": n_type1,
            "n_type2_bees": 0,
            "n_type4_bees": 0,
            "n_type5_bees": n_bees - n_type1,
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
            "summary_label": f"Extend/trim edit-only BCO (all {n_bees} bees)",
            "run_name": "seeded_bco_extend_trim_edit_only_from_lc_mumford0",
            "use_neural_bees": False,
            "n_type1_bees": 0,
            "n_type2_bees": 0,
            "n_type4_bees": 0,
            "n_type5_bees": n_bees,
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
