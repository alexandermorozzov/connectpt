"""Seed-sweep configuration constants and a few MACSA helpers.

The three bespoke sweep drivers that used to live here
(``run_worse_accept_seed_sweep`` / ``run_nx_dataset_bco_seed_sweep`` /
``run_macsa_seeded_comparison``) have been replaced by the single unified
``run_seed_sweep`` (eval_lib.sweep), which wraps the shared ``run_method``.

What remains: the accept-experiment / seed / reuse-flag constants still read by
the notebook sections, ``macsa_eval_bounds`` (per-scenario eval bounds, used by
§11) and the seed-average / result-reshape companions kept for reference.
"""
from .context import *  # noqa: F401,F403
from .params import *  # noqa: F401,F403
from .helpers import *  # noqa: F401,F403
from .baselines import *  # noqa: F401,F403
from . import plots as _plots


# === section 8 (Worse Acceptance Seed Sweep) ===
from IPython.display import display

WORSE_ACCEPT_SEEDS = list(range(1))
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


def summarize_worse_accept_seed_sweep(rows_df: pd.DataFrame):
    """Seed-average a worse-accept / NX seed-sweep into a comparison table.

    summary_df carries the same metric columns as the cell-27
    comparison_rows table, each as mean_<metric> + std_<metric> over seeds
    (no selection / accepted-total counters — those live in the histograms).
    comparison_df pairs the with_worse vs without_worse means side by side.
    """
    group_cols = ["variant_key", "variant", "accept_key", "accept_mode"]
    metric_cols = _plots.filter_component_columns([
        "cost", "cost_demand_term", "cost_route_term",
        "cost_connectivity_term", "cost_penalty_term", "ATT", "RTT",
        "median_connectivity", "$d_{un}$", "# disconnected node pairs",
        "# stops out of bounds", "# routes",
    ], ENABLED_COST_COMPONENTS)
    grouped = rows_df.groupby(group_cols, sort=False)
    stats = grouped[metric_cols].agg(["mean", "std"])
    stats.columns = [f"{stat}_{col}" for col, stat in stats.columns]
    n_seeds = grouped["seed"].nunique().rename("n_seeds")
    summary_df = stats.join(n_seeds).reset_index()

    without_df = summary_df[summary_df["accept_key"] == "without_worse"]
    with_df = summary_df[summary_df["accept_key"] == "with_worse"]
    comparison_df = with_df.merge(
        without_df,
        on=["variant_key", "variant"],
        suffixes=("_with_worse", "_without_worse"),
    )
    keep = ["variant_key", "variant"]
    for col in metric_cols:
        keep += [f"mean_{col}_without_worse", f"mean_{col}_with_worse"]
    comparison_df = comparison_df[keep].copy()
    for col in ["cost", "ATT", "RTT"]:
        comparison_df[f"delta_{col}_with_minus_without"] = (
            comparison_df[f"mean_{col}_with_worse"]
            - comparison_df[f"mean_{col}_without_worse"]
        )
    return summary_df, comparison_df


def bco_experiment_results_from_seed_sweep(
    sweep_results: dict,
    accept_key: str = "without_worse",
    seed: int | None = None,
):
    """Recreate the old bco_experiment_results shape from one seed sweep.

    This keeps all downstream tables/plots working from the sweep as the
    single source of truth, without rerunning BCO variants separately.
    """
    if seed is None:
        seed = int(sweep_results["seeds"][0])

    mode_key = sweep_results.get("mode_key", "uniform")
    mode_label = sweep_results.get("mode_label", route_selection_mode_label(mode_key))
    mode_results = {}
    for variant in BCO_VARIANTS:
        variant_runs = sweep_results["results"][accept_key][variant["key"]]
        result = next(
            (run for run in variant_runs if int(run["seed"]) == int(seed)),
            variant_runs[0],
        )
        mode_results[variant["key"]] = {
            "mode_key": mode_key,
            "mode_label": mode_label,
            "variant_key": variant["key"],
            "variant_label": variant["summary_label"],
            "n_type1_bees": variant["n_type1_bees"],
            "n_type2_bees": variant["n_type2_bees"],
            "n_type4_bees": variant["n_type4_bees"],
            "n_type5_bees": variant.get("n_type5_bees", 0),
            "n_type6_bees": variant.get("n_type6_bees", 0),
            "n_type7_bees": variant.get("n_type7_bees", 0),
            "run_name": result["run_name"],
            "metrics": result["metrics"],
            "unserved_demand": result["unserved_demand"],
            "routes": result["routes"],
            "routes_tensor": result["routes_tensor"],
            "mutation_counts": result["mutation_counts"],
            "source_accept_key": accept_key,
            "source_seed": int(result["seed"]),
        }
    return {mode_key: mode_results}


# === section 8b (NX Heuristic Dataset BCO Seed Sweep) ===
NX_DATASET_SWEEP_SEEDS = WORSE_ACCEPT_SEEDS
NX_DATASET_SWEEP_ACCEPT_EXPERIMENTS = WORSE_ACCEPT_EXPERIMENTS
NX_DATASET_SWEEP_VARIANT_KEYS = [variant["key"] for variant in BCO_VARIANTS]
NX_DATASET_SWEEP_DUMP_ROUTES = False
REUSE_EXISTING_NX_SWEEP_RESULTS = True


# === section 11 (MACSA Seeded BCO/RL Comparison) ===
from IPython.display import display

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


def evaluate_seed_routes_on_tensors(cfg, tensors, routes_tensor, run_name_prefix="seed_eval_"):
    dataloader = make_tensor_dataloader(cfg.eval.dataset, tensors)
    device, run_name, _, cost_obj, _ = lrnu.process_standard_experiment_cfg(
        cfg,
        run_name_prefix=run_name_prefix,
        weights_required=True,
    )
    output = lrnu.test_method(
        None,
        dataloader,
        cfg.eval,
        OmegaConf.create({"method": "tensor"}),
        cost_obj,
        silent=True,
        device=device,
        return_routes=True,
        routes_tensor=routes_tensor,
    )
    _, _, unserved_demand, metrics, routes = output
    routes_tensor = as_route_tensor(routes)
    metrics = add_cost_breakdown_to_metrics(metrics, dataloader, cfg.eval, cost_obj, routes_tensor, device)
    return run_name, metrics, unserved_demand, routes_tensor


def _macsa_base_row(scenario, label, metrics, routes, method_group, **extra):
    n_routes, min_route_len, max_route_len = macsa_eval_bounds(scenario)
    row = summarize_run(label, metrics, routes)
    row.update({
        "scenario": scenario["name"],
        "scenario_path": str(scenario["path"].relative_to(ROOT_DIR)),
        "graph_source": "macsa",
        "n_nodes": int(scenario["tensors"]["node_locs"].shape[0]),
        "n_routes_target": n_routes,
        "min_route_len": min_route_len,
        "max_route_len": max_route_len,
        "method_group": method_group,
    })
    row.update(extra)
    return row
