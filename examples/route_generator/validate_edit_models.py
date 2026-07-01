"""Validate old/new LC edit checkpoints on a shared curriculum split.

This is a lightweight runner for comparing the trim-capable improvement
checkpoints without executing the full paper_combined notebook.  It mirrors the
notebook's balanced validation setup: demand off, 0.5 RTT + 0.5 weighted median
connectivity, and the training-time adjustment cap penalty.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from tqdm.auto import tqdm

import connectpt.routes_generator.utils as lrnu
from connectpt.routes_generator.bee_colony import (
    get_adjustment_degrees, get_adjustment_penalties)
from connectpt.routes_generator.improvement_learning import (
    RouteGenBatchState, evaluate_lc_improvement, load_raw_graphs_and_lc_routes,
    make_improvement_batch)
from connectpt.routes_generator.objectives import load_unified_objective
from eval_lib.context import CFG_DIR, DATASETS_DIR, EDIT_MODEL_WEIGHTS_DIR, ROOT_DIR

_OBJ = load_unified_objective()
CONNECTIVITY_MODE = _OBJ.connectivity_mode
DISABLED_COST_COMPONENTS = list(_OBJ.disabled_components)
ADJ_WEIGHT = _OBJ.adj_weight
ADJ_TARGET = _OBJ.adj_target
ADJ_TRAIN_OBJECTIVE = _OBJ.adj_train_objective
ADJ_GAP = _OBJ.adj_gap
ADJ_MODE = _OBJ.adj_mode


DEFAULT_DATASET = (
    DATASETS_DIR / "lc_copy_subcopy_curriculum_n500_n50_r12_len8_15_v4")
DEFAULT_MODELS = [
    {
        "label": "new_realistic_scratch_adj",
        "path": EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_realistic_scratch_rttwmc_adj_v1.pt",
        "n_adj_feats": 1,
    },
    {
        "label": "old_finetune100",
        "path": EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_rttconn_adj_w10_t02_finetune100.pt",
        "n_adj_feats": 0,
    },
]


def _repo_rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path)


def _build_model_and_cost(label: str, n_adj_feats: int, device_cpu: bool = True):
    overrides = [
        "model=bestsofar_feb2023_trim",
        "model.route_generator.kwargs.serial_halting=True",
        "model.route_generator.kwargs.allow_trim_below_min=true",
        f"++model.route_generator.kwargs.n_adjustment_cond_feats={int(n_adj_feats)}",
        f"++run_name=validate_{label}",
        "++experiment.logdir=null",
        f"++experiment.cpu={str(bool(device_cpu)).lower()}",
        f"++experiment.cost_function.kwargs.connectivity_mode={CONNECTIVITY_MODE}",
        "++experiment.cost_function.kwargs.use_weighted_connectivity=true",
    ]
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="ppo_50nodes.yaml", overrides=overrides)
    device, run_name, _, cost_obj, model = lrnu.process_standard_experiment_cfg(
        cfg, run_name_prefix="validation_")
    cost_obj.ignore_stops_oob = True
    cost_obj.set_enabled_components(disabled_components=DISABLED_COST_COMPONENTS)
    cost_obj.variable_weights = False
    cost_obj.demand_time_weight = 0.0
    cost_obj.route_time_weight = 0.5
    cost_obj.median_connectivity_weight = 0.5
    # Adj is added explicitly below so we can report raw and adj-aware costs.
    cost_obj.adjustment_degree_weight = 0.0
    return cfg, device, run_name, cost_obj, model


def _validation_indices(meta_df: pd.DataFrame, per_tier: int, seed: int,
                        train_fraction: float = 0.9) -> dict[str, torch.Tensor]:
    tiers = list(dict.fromkeys(meta_df["tier"].tolist()))
    tier_of = dict(zip(meta_df["graph_index"], meta_df["tier"]))
    perm = torch.randperm(len(meta_df), generator=torch.Generator().manual_seed(seed))
    by_tier = {tier: [] for tier in tiers}
    for gi in perm.tolist():
        by_tier[tier_of[gi]].append(gi)

    val_by_tier = {}
    for tier in tiers:
        idxs = by_tier[tier]
        n_val = max(per_tier, int(round((1.0 - train_fraction) * len(idxs))))
        n_val = min(n_val, max(0, len(idxs) - 1))
        val_by_tier[tier] = torch.tensor(idxs[:min(per_tier, n_val)], dtype=torch.long)
    return val_by_tier


def _pad_routes(routes: torch.Tensor, n_routes: int, width: int) -> torch.Tensor:
    if routes.ndim == 2:
        routes = routes.unsqueeze(0)
    out = torch.full((routes.shape[0], n_routes, width), -1,
                     dtype=routes.dtype, device=routes.device)
    nr = min(n_routes, routes.shape[1])
    w = min(width, routes.shape[2])
    out[:, :nr, :w] = routes[:, :nr, :w]
    return out


def _adj_penalty(final_routes: torch.Tensor, seed_routes: torch.Tensor,
                 cost_obj) -> tuple[torch.Tensor, torch.Tensor]:
    n_routes = max(int(final_routes.shape[1]), int(seed_routes.shape[1]))
    width = max(int(final_routes.shape[2]), int(seed_routes.shape[2]))
    final_routes = _pad_routes(final_routes.long(), n_routes, width)
    seed_routes = _pad_routes(seed_routes.long(), n_routes, width)
    degrees = get_adjustment_degrees(
        final_routes, seed_routes, cost_obj.symmetric_routes,
        gap=ADJ_GAP, mode=ADJ_MODE)
    network_degree = degrees.reshape(degrees.shape[0], -1).mean(dim=1)
    penalty = ADJ_WEIGHT * get_adjustment_penalties(
        network_degree, objective=ADJ_TRAIN_OBJECTIVE, target=ADJ_TARGET)
    return penalty.detach().cpu(), network_degree.detach().cpu()


def _raw_costs_for_routes(cost_obj, graphs, seed_routes, indices, final_routes,
                          device, min_route_len: int, max_route_len: int,
                          batch_size: int, target_n_routes: int):
    seed_costs, final_costs = [], []
    eval_weights = cost_obj.get_weights(device)
    cursor = 0
    for batch_indices in indices.split(batch_size):
        graph_batch, route_batch = make_improvement_batch(
            graphs, seed_routes, batch_indices, device, training=False,
            target_n_routes=target_n_routes)
        batch_final = final_routes[cursor: cursor + len(batch_indices)].to(device)
        cursor += len(batch_indices)

        seed_state = RouteGenBatchState(
            graph_batch, cost_obj, route_batch.shape[1],
            min_route_len, max_route_len, cost_weights=eval_weights)
        seed_state.add_new_routes(route_batch)
        final_state = RouteGenBatchState(
            graph_batch, cost_obj, batch_final.shape[1],
            min_route_len, max_route_len, cost_weights=eval_weights)
        final_state.add_new_routes(batch_final)

        seed_costs.append(cost_obj(seed_state).cost.detach().cpu())
        final_costs.append(cost_obj(final_state).cost.detach().cpu())
    return torch.cat(seed_costs), torch.cat(final_costs)


def _rollout_adj_kwargs(model):
    n_adj_feats = int(getattr(model, "n_adjustment_cond_feats", 0) or 0)
    if n_adj_feats <= 0:
        return {}
    kwargs = dict(adjustment_target=float(ADJ_TARGET),
                  adjustment_use_current=False,
                  adjustment_gap=ADJ_GAP,
                  adjustment_mode=ADJ_MODE)
    if n_adj_feats > 1:
        kwargs["adjustment_weight"] = float(ADJ_WEIGHT)
    return kwargs


def _evaluate_subset(model, cost_obj, graphs, seed_routes, indices, device,
                     min_route_len: int, max_route_len: int, batch_size: int,
                     target_n_routes: int):
    raw = evaluate_lc_improvement(
        model, cost_obj, graphs, seed_routes, indices, device,
        min_route_len, max_route_len, batch_size=batch_size,
        max_route_edit_steps=max_route_len,
        max_trim_actions_per_route=1,
        return_action_stats=True,
        target_n_routes=target_n_routes,
        return_best_routes=False,
        **_rollout_adj_kwargs(model))

    final_routes = torch.cat([
        _pad_routes(
            routes if routes.ndim == 3 else routes.unsqueeze(0),
            target_n_routes, max_route_len)
        for routes in raw["routes"]
    ], dim=0)
    seed_subset = seed_routes[indices]
    seed_raw, final_raw = _raw_costs_for_routes(
        cost_obj, graphs, seed_routes, indices, final_routes, device,
        min_route_len, max_route_len, batch_size, target_n_routes)
    final_adj_pen, final_adj = _adj_penalty(final_routes, seed_subset, cost_obj)
    seed_adj_pen = torch.zeros_like(final_adj_pen)
    seed_adj = torch.zeros_like(final_adj)

    seed_total = seed_raw + seed_adj_pen
    final_total = final_raw + final_adj_pen
    raw.update({
        "raw_seed_cost": float(seed_raw.mean()),
        "raw_final_cost": float(final_raw.mean()),
        "raw_delta": float((seed_raw - final_raw).mean()),
        "seed_adjustment_penalty": float(seed_adj_pen.mean()),
        "final_adjustment_penalty": float(final_adj_pen.mean()),
        "seed_adjustment_degree": float(seed_adj.mean()),
        "final_adjustment_degree": float(final_adj.mean()),
        "adjustment_penalty_delta": float((seed_adj_pen - final_adj_pen).mean()),
        "adjustment_degree_delta": float((seed_adj - final_adj).mean()),
        "seed_cost": float(seed_total.mean()),
        "final_cost": float(final_total.mean()),
        "delta": float((seed_total - final_total).mean()),
        "win_rate": float((final_total < seed_total).float().mean()),
    })
    raw.pop("routes", None)
    return raw


def _row_from_result(model_label: str, tier: str, n: int, result: dict,
                     seconds: float) -> dict:
    action_stats = result.get("action_stats") or {}
    return {
        "model": model_label,
        "tier": tier,
        "n": int(n),
        "seconds": round(float(seconds), 2),
        "seed_cost": result["seed_cost"],
        "final_cost": result["final_cost"],
        "delta": result["delta"],
        "win_rate": result["win_rate"],
        "raw_seed_cost": result["raw_seed_cost"],
        "raw_final_cost": result["raw_final_cost"],
        "raw_delta": result["raw_delta"],
        "final_adj_degree": result["final_adjustment_degree"],
        "final_adj_penalty": result["final_adjustment_penalty"],
        "changed_route_rate": result["changed_route_rate"],
        "changed_graph_rate": result["changed_graph_rate"],
        "component_delta_route": result["component_delta_route"],
        "component_delta_connectivity": result["component_delta_connectivity"],
        "avg_actions_per_route": action_stats.get("avg_actions_per_route"),
        "halt": action_stats.get("halt"),
        "extend": action_stats.get("extend"),
        "trim_start": action_stats.get("trim_start"),
        "trim_end": action_stats.get("trim_end"),
    }


def _summary_from_tier_rows(model_label: str, rows: list[dict]) -> dict:
    if not rows:
        raise ValueError(f"No tier rows for {model_label}")
    total_n = sum(int(row["n"]) for row in rows)
    summary = {"model": model_label, "tier": "ALL", "n": total_n,
               "seconds": round(sum(float(row["seconds"]) for row in rows), 2)}
    skip = {"model", "tier", "n", "seconds"}
    for key in rows[0]:
        if key in skip:
            continue
        vals, weights = [], []
        for row in rows:
            value = row.get(key)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            vals.append(float(value))
            weights.append(float(row["n"]))
        summary[key] = (float(np.average(vals, weights=weights))
                        if vals else None)
    return summary


def run_validation(args):
    dataset_dir = Path(args.dataset)
    meta_path = dataset_dir / "meta.csv"
    raw_path = dataset_dir / "raw_graphs_1000.pkl"
    if not meta_path.exists() or not raw_path.exists():
        raise FileNotFoundError(
            f"Expected meta.csv and raw_graphs_1000.pkl under {dataset_dir}")

    print(f"[validation] dataset={_repo_rel(dataset_dir)}")
    t0 = time.perf_counter()
    graphs, seed_routes = load_raw_graphs_and_lc_routes(raw_path, dataset_dir)
    meta_df = pd.read_csv(meta_path)
    print(f"[validation] loaded {len(graphs)} graphs, routes={tuple(seed_routes.shape)} "
          f"in {time.perf_counter() - t0:.1f}s")

    val_by_tier = _validation_indices(meta_df, args.per_tier, args.seed)
    print("[validation] split:", {tier: int(len(idx)) for tier, idx in val_by_tier.items()})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_tier_rows, summary_rows = [], []
    for spec in DEFAULT_MODELS:
        label = spec["label"]
        weights_path = Path(spec["path"])
        print(f"\n[validation] model={label} weights={_repo_rel(weights_path)} "
              f"n_adj_feats={spec['n_adj_feats']}")
        if not weights_path.exists():
            raise FileNotFoundError(weights_path)
        _, device, _, cost_obj, model = _build_model_and_cost(
            label, int(spec["n_adj_feats"]), device_cpu=not args.cuda)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()

        model_rows = []
        for tier, indices in tqdm(list(val_by_tier.items()), desc=f"{label} tiers"):
            if len(indices) == 0:
                continue
            t_tier = time.perf_counter()
            result = _evaluate_subset(
                model, cost_obj, graphs, seed_routes, indices, device,
                args.min_route_len, args.max_route_len, args.batch_size,
                args.target_n_routes)
            row = _row_from_result(
                label, tier, int(len(indices)), result,
                time.perf_counter() - t_tier)
            by_tier_rows.append(row)
            model_rows.append(row)
        summary_rows.append(_summary_from_tier_rows(label, model_rows))

    summary_df = pd.DataFrame(summary_rows)
    by_tier_df = pd.DataFrame(by_tier_rows)
    summary_path = out_dir / "edit_model_validation_summary.csv"
    by_tier_path = out_dir / "edit_model_validation_by_tier.csv"
    summary_df.round(6).to_csv(summary_path, index=False)
    by_tier_df.round(6).to_csv(by_tier_path, index=False)
    meta = {
        "dataset": _repo_rel(dataset_dir),
        "per_tier": int(args.per_tier),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "target_n_routes": int(args.target_n_routes),
        "min_route_len": int(args.min_route_len),
        "max_route_len": int(args.max_route_len),
        "adj_weight": ADJ_WEIGHT,
        "adj_target": ADJ_TARGET,
        "adj_objective": ADJ_TRAIN_OBJECTIVE,
        "adj_gap": ADJ_GAP,
        "adj_mode": ADJ_MODE,
        "models": [
            {**spec, "path": _repo_rel(Path(spec["path"]))}
            for spec in DEFAULT_MODELS
        ],
        "outputs": {
            "summary": _repo_rel(summary_path),
            "by_tier": _repo_rel(by_tier_path),
        },
    }
    meta_path = out_dir / "edit_model_validation_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n[validation] summary")
    print(summary_df.round(4).to_string(index=False))
    print(f"[validation] saved -> {_repo_rel(summary_path)}")
    print(f"[validation] saved -> {_repo_rel(by_tier_path)}")
    print(f"[validation] saved -> {_repo_rel(meta_path)}")
    return summary_df, by_tier_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("artifacts/model_validation"))
    parser.add_argument("--per-tier", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-n-routes", type=int, default=12)
    parser.add_argument("--min-route-len", type=int, default=8)
    parser.add_argument("--max-route-len", type=int, default=15)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())
