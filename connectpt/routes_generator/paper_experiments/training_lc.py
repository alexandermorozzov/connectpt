"""LC copy-tier training experiment: dataset generation, model/cost building,
baseline, history plotting, balanced evaluation and post-training convergence.

This is the one-off pipeline that produced the paper's edit checkpoint (now just
loaded by the experiments). The *reusable* training mechanics already live in
the library (``connectpt.routes_generator.training``: ``TrainingDataModule`` +
``EditTrainingRun`` driven by ``cfg/train/edit*.yaml``); this module only holds
the paper-specific scaffolding that used to sit inline in ``paper_combined.ipynb``
so the notebook stays a thin presentation layer.

Nothing here owns model state: ``build_edit_model_and_cost`` builds config-first
via the library factories, exactly like the dormant balanced-eval cell needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def load_train_config(name="edit_scratch", *, overrides=None, cfg_dir=None):
    """Compose a training config from ``cfg/train/<name>.yaml`` (the single
    source of truth for every training/report knob). Mirrors the reference
    ``load_experiment_config`` -- the notebook loads this once and passes the
    composed cfg to the factory helpers below; no constants in the notebook."""
    from hydra import compose, initialize_config_dir

    from connectpt.routes_generator.core.paths import CFG_DIR

    cfg_dir = cfg_dir or CFG_DIR
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        return compose(config_name=f"train/{name}", overrides=list(overrides or []))


def copytier_config(cfg) -> "CopyTierConfig":
    """Build the dataset-generation config from a composed train cfg (factory:
    reads cfg.dataset_gen / cfg.data / cfg.curriculum, not notebook constants)."""
    from omegaconf import OmegaConf

    from connectpt.routes_generator.core.paths import DATASETS_DIR
    from connectpt.routes_generator.data.route_copies import COPY_TIER_CFG

    dg, data = cfg.dataset_gen, cfg.data
    ddir = DATASETS_DIR / data.dataset_dirname
    tiers = list(cfg.curriculum.tiers) if cfg.get("curriculum") else list(COPY_TIER_CFG)
    combos = [list(c) for c in OmegaConf.to_container(dg.lc_combos, resolve=True)]
    corrupt_target = (int(data.target_n_routes) if bool(dg.get("corrupt_all_routes"))
                      else dg.get("n_routes_to_corrupt"))
    return CopyTierConfig(
        new_dataset_dir=ddir, subset_pkl=ddir / "raw_graphs_1000.pkl",
        meta_csv=ddir / "meta.csv", raw_graph_seed=int(dg.raw_graph_seed),
        raw_n_nodes=int(dg.raw_n_nodes), raw_graph_type=str(dg.raw_graph_type),
        n_graphs=int(dg.n_graphs), tiers=tiers, lc_combos=combos,
        target_n_routes=int(data.target_n_routes), min_route_len=int(data.min_route_len),
        max_route_len=int(data.max_route_len),
        connectivity_mode=str(cfg.experiment.cost_function.kwargs.connectivity_mode),
        lc_n_samples=int(dg.lc_n_samples), tier_cfg=COPY_TIER_CFG,
        corrupt_target=corrupt_target, force_regen=bool(dg.get("force_regen", False)))


def _pad_routes_to(routes, n_routes, max_route_len):
    """Pad/clip a route tensor to (n_routes, max_route_len) with -1 fill.
    Shared by dataset generation and the clean-LC baseline."""
    from ._common import pad_routes_to
    return pad_routes_to(routes, n_routes, max_route_len)


def _graph_tensors(g):
    """node_locs / street_adj / demand dict for one graph (shared helper)."""
    from connectpt.routes_generator.citygraph_dataset import STOP_KEY

    return {"node_locs": g[STOP_KEY].pos.detach().cpu().clone(),
            "street_adj": g.street_adj.detach().cpu().clone(),
            "demand": g.demand.detach().cpu().clone()}


@dataclass
class CopyTierConfig:
    """Geometry, paths and tier schedule for the copy-tier training dataset.

    Mirrors the inline notebook constants 1:1 so generation is byte-identical;
    the notebook builds this from its config cell and passes it to
    :func:`build_copytier_dataset`.
    """
    new_dataset_dir: Any
    subset_pkl: Any
    meta_csv: Any
    raw_graph_seed: int
    raw_n_nodes: int
    raw_graph_type: str
    n_graphs: int
    tiers: list
    lc_combos: list
    target_n_routes: int
    min_route_len: int
    max_route_len: int
    connectivity_mode: str
    lc_n_samples: int
    tier_cfg: dict
    corrupt_target: Any = None
    force_regen: bool = False


def build_copytier_dataset(cfg: CopyTierConfig):
    """Generate the four-tier route-copy training dataset (idempotent).

    Returns seconds/graph if it (re)generated, or ``None`` if a complete dataset
    already existed and was reused. Logic moved verbatim from the dormant
    notebook cell -- learned construction batched on the GPU per LC-objective
    combo, then per-graph copy/subcopy corruption + dump.
    """
    import json
    import pickle
    import random as _random
    import shutil
    import time as _time

    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    from connectpt.routes_generator.citygraph_dataset import DynamicCityGraphDataset
    from connectpt.routes_generator.lc_eval import build_lc_cfg, run_lc_batch
    from connectpt.routes_generator.torch_utils import dump_routes
    from connectpt.routes_generator.data.route_copies import (
        count_changed_routes, inject_route_copies, redundancy_stats,
        uncovered_demand_pct)

    def _to_fixed(routes):
        return _pad_routes_to(routes, cfg.target_n_routes, cfg.max_route_len)

    _tensors = _graph_tensors

    def generate_dataset():
        if cfg.new_dataset_dir.exists():
            shutil.rmtree(cfg.new_dataset_dir)
        cfg.new_dataset_dir.mkdir(parents=True, exist_ok=True)
        _random.seed(cfg.raw_graph_seed); torch.manual_seed(cfg.raw_graph_seed)
        ds = DynamicCityGraphDataset(min_nodes=cfg.raw_n_nodes, max_nodes=cfg.raw_n_nodes,
                                     data_type=cfg.raw_graph_type, mumford_style=True, pos_only=False)
        raw = [ds.generate_graph(n_nodes=cfg.raw_n_nodes) for _ in range(cfg.n_graphs)]
        per = cfg.n_graphs // len(cfg.tiers)
        tensors_all = [_tensors(g) for g in raw]

        # --- batched learned construction on the GPU --------------------------
        # Group graphs by LC objective combo (one cfg / one set of cost weights
        # per combo), then construct each group in GPU batches of LC_GEN_BATCH
        # via run_lc_batch (one forward per batch) instead of one graph at a
        # time -- this is what actually keeps the GPU busy during dataset gen.
        LC_GEN_BATCH = 64 if torch.cuda.is_available() else 8
        lc_routes = [None] * cfg.n_graphs
        for _ci, (d, rt, cn, ctag) in enumerate(cfg.lc_combos):
            combo_idxs = [gi for gi in range(cfg.n_graphs) if gi % len(cfg.lc_combos) == _ci]
            if not combo_idxs:
                continue
            c = build_lc_cfg(run_name=f"copy_cur_{ctag}", n_routes=cfg.target_n_routes,
                             min_route_len=cfg.min_route_len, max_route_len=cfg.max_route_len,
                             demand_time_weight=d, route_time_weight=rt,
                             median_connectivity_weight=cn,
                             connectivity_mode=cfg.connectivity_mode)
            for _s in tqdm(range(0, len(combo_idxs), LC_GEN_BATCH),
                           desc=f"LC construct [{ctag}]"):
                chunk = combo_idxs[_s:_s + LC_GEN_BATCH]
                routes_b = run_lc_batch(
                    c, [tensors_all[gi] for gi in chunk],
                    run_name_prefix="copy_cur_", n_samples=cfg.lc_n_samples,
                    batch_size=len(chunk))
                for _j, gi in enumerate(chunk):
                    lc_routes[gi] = routes_b[_j]

        # --- per-graph corruption + dump (CPU) --------------------------------
        subset, meta = [], []
        for gi, g in enumerate(tqdm(raw, desc="corrupt + dump")):
            tier = cfg.tiers[min(gi // per, len(cfg.tiers) - 1)]
            tier_cfg = cfg.tier_cfg[tier]
            rng = _random.Random(1000 + gi)
            tn = tensors_all[gi]
            ctag = cfg.lc_combos[gi % len(cfg.lc_combos)][3]
            routes = _to_fixed(lc_routes[gi])
            before = redundancy_stats(routes)
            clean_routes = routes.clone()
            routes, event_counts, _ = inject_route_copies(
                routes, tier_cfg, rng, cfg.min_route_len, cfg.max_route_len,
                demand=tn["demand"], n_nodes=cfg.raw_n_nodes,
                street_adj=tn["street_adj"])
            after = redundancy_stats(routes)
            n_corrupted = count_changed_routes(clean_routes, routes)
            gdir = cfg.new_dataset_dir / f"graph_{gi:04d}"; gdir.mkdir(parents=True, exist_ok=True)
            dump_routes(f"lc_copy_cur_graph_{gi:04d}_routes", routes, out_dir=gdir)
            subset.append(g)
            meta.append({
                "graph_index": gi, "tier": tier, "tier_kind": ("clean" if not tier_cfg["kinds"] else "copies"),
                "lc_combo": ctag,
                "applied_events": int(sum(event_counts.values())),
                "n_routes_to_corrupt": ("" if cfg.corrupt_target is None else int(cfg.corrupt_target)),
                "n_corrupted_routes": int(n_corrupted),
                "mutation_events": json.dumps(dict(event_counts), sort_keys=True),
                "d_un_after_pct": round(uncovered_demand_pct(routes, tn["demand"], cfg.raw_n_nodes), 2),
                "redun_before": round(before["redundancy"], 4),
                "redun_after": round(after["redundancy"], 4),
                "max_leg_use_before": before["max_leg_use"],
                "max_leg_use_after": after["max_leg_use"],
            })
        with cfg.subset_pkl.open("wb") as fh:
            pickle.dump(subset, fh)
        pd.DataFrame(meta).to_csv(cfg.meta_csv, index=False)
        print(f"Saved {len(subset)} graphs -> {cfg.new_dataset_dir}")

    _have = len(list(cfg.new_dataset_dir.glob("graph_*"))) if cfg.new_dataset_dir.exists() else 0
    if cfg.subset_pkl.exists() and _have == cfg.n_graphs and cfg.meta_csv.exists() and not cfg.force_regen:
        print(f"dataset already exists ({_have}) -> skip")
        return None
    if _have and _have != cfg.n_graphs:
        print(f"found {_have} graphs, expected {cfg.n_graphs} -> regenerate")
    _tg = _time.perf_counter()
    generate_dataset()
    return (_time.perf_counter() - _tg) / max(1, cfg.n_graphs)


def clean_lc_baseline(cfg, *, graphs, seed_routes, meta_df, device, baseline_path):
    """Per-curriculum-stage clean-LC baseline cost CSV for the history figure.

    Config-driven: geometry/combos/curriculum + baseline knobs all read from the
    composed train ``cfg`` (cfg.report.baseline, cfg.curriculum, cfg.dataset_gen).
    Returns the baseline DataFrame.
    """
    import pandas as pd
    import torch

    from connectpt.routes_generator.improvement_learning import (
        RouteGenBatchState, make_improvement_batch, _clone_cost_weights)
    from connectpt.routes_generator.objectives import CostFactory
    from connectpt.routes_generator.lc_eval import build_lc_cfg, run_lc_batch

    ct_cfg = copytier_config(cfg)
    batch = int(cfg.report.baseline.batch)
    adj_weight = float(cfg.report.baseline.adj_weight)
    n_per_tier = int(cfg.dataset_gen.n_graphs) // len(ct_cfg.tiers)
    n_iterations = int(cfg.train_loop.n_iterations)
    use_curriculum = cfg.get("curriculum") is not None
    curriculum = ([(round(float(frac) * n_iterations), list(t), label)
                   for frac, t, label in cfg.curriculum.schedule]
                  if use_curriculum else [])

    def _to_fixed(routes):
        return _pad_routes_to(routes, ct_cfg.target_n_routes, ct_cfg.max_route_len)

    _tensors = _graph_tensors

    tiers, lc_combos = ct_cfg.tiers, ct_cfg.lc_combos
    _baseline_pool_by_tier = {
        tier: meta_df.loc[meta_df["tier"] == tier, "graph_index"].astype(int).tolist()
        for tier in tiers
    }

    # Clean-LC baseline cost: the unified objective (RTT+WMC, demand off, fixed
    # 0.5/0.5 weights) with the adjustment penalty off, via the library
    # CostFactory + objective YAML.
    _cost_base = CostFactory.build_unified("rtt_wmc_no_demand", for_training=True)
    _cost_base.variable_weights = False
    _cost_base.adjustment_degree_weight = float(adj_weight)
    _cost_base.ignore_stops_oob = True
    _cost_base.to(device)
    _eval_weights = _cost_base.get_weights(device)

    _baseline_by_tier = {
        tier: [int(i) for i in _baseline_pool_by_tier[tier][:n_per_tier]]
        for tier in tiers
    }
    _baseline_indices = sorted({gi for _idxs in _baseline_by_tier.values() for gi in _idxs})
    print("clean LC baseline graphs per tier:", {tier: len(_idxs) for tier, _idxs in _baseline_by_tier.items()})
    _clean_routes = seed_routes.clone()
    for _ci, (_d, _rt, _cn, _ctag) in enumerate(lc_combos):
        _idxs = [gi for gi in _baseline_indices if gi % len(lc_combos) == _ci]
        if not _idxs:
            continue
        _lc_cfg = build_lc_cfg(
            run_name=f"clean_lc_baseline_{_ctag}", n_routes=ct_cfg.target_n_routes,
            min_route_len=ct_cfg.min_route_len, max_route_len=ct_cfg.max_route_len,
            demand_time_weight=_d, route_time_weight=_rt,
            median_connectivity_weight=_cn, connectivity_mode=ct_cfg.connectivity_mode)
        for _s in range(0, len(_idxs), batch):
            _chunk = _idxs[_s:_s + batch]
            _routes_b = run_lc_batch(
                _lc_cfg, [_tensors(graphs[gi]) for gi in _chunk],
                run_name_prefix="clean_lc_baseline_", n_samples=ct_cfg.lc_n_samples,
                batch_size=len(_chunk))
            for _j, gi in enumerate(_chunk):
                _clean_routes[gi] = _to_fixed(_routes_b[_j])

    def _mean_cost(routes_src, idxs):
        idxs = torch.as_tensor(list(map(int, idxs)), dtype=torch.long)
        if len(idxs) == 0:
            return float("nan")
        costs = []
        for _chunk in idxs.split(batch):
            _gb, _rb = make_improvement_batch(
                graphs, routes_src, _chunk, device, training=False,
                target_n_routes=ct_cfg.target_n_routes)
            _state = RouteGenBatchState(
                _gb, _cost_base, _rb.shape[1], ct_cfg.min_route_len, ct_cfg.max_route_len,
                cost_weights=_clone_cost_weights(_eval_weights))
            _state.add_new_routes(_rb)
            _res = _cost_base(_state)
            costs.append(_res.cost.detach().cpu())
        return torch.cat(costs).mean().item()

    _tier_of = dict(zip(meta_df["graph_index"].astype(int), meta_df["tier"]))
    if use_curriculum:
        _stage_rows = []
        for _until, _tiers, _label in curriculum:
            _idxs = [gi for tier in _tiers for gi in _baseline_by_tier.get(tier, [])]
            _stage_rows.append({
                "until_epoch": int(_until), "curriculum_stage": _label,
                "active_tiers": ";".join(_tiers), "n_graphs": len(_idxs),
                "n_per_tier_target": int(n_per_tier),
                "clean_lc_cost": _mean_cost(_clean_routes, _idxs),
                "seed_cost": _mean_cost(seed_routes, _idxs)})
    else:
        _idxs = _baseline_indices
        _stage_rows = [{
            "until_epoch": int(n_iterations), "curriculum_stage": "all",
            "active_tiers": ";".join(sorted({_tier_of[i] for i in _idxs})),
            "n_graphs": len(_idxs), "n_per_tier_target": int(n_per_tier),
            "clean_lc_cost": _mean_cost(_clean_routes, _idxs),
            "seed_cost": _mean_cost(seed_routes, _idxs)}]

    clean_lc_baseline_df = pd.DataFrame(_stage_rows)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    clean_lc_baseline_df.to_csv(baseline_path, index=False)
    print(f"clean LC baseline -> {baseline_path}")
    return clean_lc_baseline_df


def stitch_history(*, prior_history_files=(), history_df=None,
                   full_history_checkpoint=None):
    """Concatenate prior history part(s) + this run's continuation into one
    epoch-indexed frame, and derive the curriculum-stage spans. Returns (h, spans)."""
    from pathlib import Path

    import pandas as pd

    parts, labels = [], []
    for f in (prior_history_files or []):
        f = Path(f)
        if f.exists():
            parts.append(pd.read_csv(f)); labels.append(f"{f.name}({len(parts[-1])})")
    if history_df is not None:
        parts.append(history_df.copy()); labels.append(f"in-memory({len(history_df)})")
    elif full_history_checkpoint is not None and Path(full_history_checkpoint).exists():
        parts.append(pd.read_csv(full_history_checkpoint))
        labels.append(f"{Path(full_history_checkpoint).name}({len(parts[-1])})")
    if not parts:
        raise FileNotFoundError("No history found. Train at least one epoch first.")
    h = pd.concat(parts, ignore_index=True)
    h["epoch"] = range(1, len(h) + 1)  # continuous axis across stitched parts
    print(f"history stitched: {' + '.join(labels)} => {len(h)} epochs")

    spans = []
    if "curriculum_stage" in h.columns and h["curriculum_stage"].notna().any():
        for epoch, label in h[["epoch", "curriculum_stage"]].dropna().itertuples(index=False, name=None):
            epoch = int(epoch)
            if spans and spans[-1][2] == label:
                spans[-1] = (spans[-1][0], epoch, label)
            else:
                spans.append((epoch, epoch, label))
    return h, spans


def plot_training_history(h, spans, cfg, *, model_outputs_dir):
    """Actor curves + curriculum shading (inline figure) and mirror every scalar
    column to a TensorBoard run. Config-driven: run name + the scalar allow-list
    come from cfg (cfg.run.name, cfg.report.tensorboard_scalars)."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from torch.utils.tensorboard import SummaryWriter

    run_name = full_history_run = cfg.run.name
    _tbs = cfg.report.get("tensorboard_scalars") if cfg.get("report") else None
    tensorboard_scalars = list(_tbs) if _tbs else None

    def _num(col):
        return pd.to_numeric(h[col], errors="coerce") if col in h.columns else None

    colors = ["#eaf3ff", "#eafbea", "#fff6e6", "#fdeaea", "#f0eaff"]

    def _shade(ax):
        for k, (s, e, lab) in enumerate(spans):
            ax.axvspan(s, e, color=colors[k % len(colors)], alpha=0.6, zorder=0)
            ax.axvline(s, color="gray", lw=0.6, ls=":")

    fig, ax = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    panels = [("train_reward_mean", "train reward"), ("val_delta", "val cost delta"),
              ("val_win_rate", "val win rate"), ("train_action_avg_actions_per_route", "avg edits/route"),
              ("val_component_delta_route", "val route delta"),
              ("val_component_delta_connectivity", "val conn delta")]
    for a, (col, title) in zip(ax.flat, panels):
        _shade(a); y = _num(col)
        if y is not None and y.notna().any():
            a.plot(h["epoch"], y, marker="o", ms=2, color="tab:blue", zorder=3)
        a.axhline(0, color="k", lw=0.7); a.set_title(title); a.set_xlabel("epoch"); a.grid(alpha=0.2)
    for s, e, lab in spans:
        ax[0, 0].text((s + e) / 2, ax[0, 0].get_ylim()[1], lab, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Actor curves + curriculum stages", fontsize=13, fontweight="bold")
    plt.show()

    def _tb_tag(col):
        for pre, grp in (("train_ppo_", "ppo/"), ("train_critic_", "critic/"),
                         ("train_action_", "action/"), ("train_component_", "component/train_"),
                         ("val_component_", "component/val_"), ("train_", "train/"), ("val_", "val/")):
            if col.startswith(pre):
                return grp + col[len(pre):]
        return "misc/" + col

    tb_dir = model_outputs_dir / "tensorboard" / (full_history_run or run_name)
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    epochs = h["epoch"].astype(int).tolist()
    n_scalars = 0
    for col in h.columns:
        if col == "epoch" or (tensorboard_scalars is not None and col not in tensorboard_scalars):
            continue
        y = pd.to_numeric(h[col], errors="coerce")
        if not y.notna().any():
            continue
        for ep, v in zip(epochs, y.tolist()):
            if v == v:  # skip NaN
                writer.add_scalar(col, float(v), ep)
        n_scalars += 1
    writer.add_figure("actor_curves", fig, global_step=epochs[-1])
    writer.flush(); writer.close()
    plt.close(fig)
    print(f"[tensorboard] {n_scalars} scalar series ({len(epochs)} epochs) -> {tb_dir}")
    return _shade


def plot_critic_metrics(h, shade):
    """Critic diagnostic panels with curriculum shading (dormant cell)."""
    import matplotlib.pyplot as plt
    import pandas as pd

    crit_cols = [c for c in h.columns if "critic" in c.lower()]
    print("critic columns:", crit_cols)
    if not crit_cols:
        return
    n = len(crit_cols)
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False, constrained_layout=True)
    for a, col in zip(ax[0], crit_cols):
        shade(a)
        y = pd.to_numeric(h[col], errors="coerce") if col in h.columns else None
        if y is not None and y.notna().any():
            a.plot(h["epoch"], y, marker="o", ms=2, color="tab:orange", zorder=3)
        a.set_title(col, fontsize=9); a.set_xlabel("epoch"); a.grid(alpha=0.2)
        if "explained" in col:
            a.axhline(0, color="k", lw=0.7)
    fig.suptitle("Critic metrics + curriculum stages", fontsize=13, fontweight="bold")
    plt.show(); plt.close(fig)
    return h[["epoch", "curriculum_stage"] + crit_cols].iloc[::max(1, len(h) // 15)].round(4)


def balanced_eval_by_tier(cfg, *, model, cost_obj, device, graphs, seed_routes,
                          val_by_tier):
    """Greedy balanced-weight rollout per curriculum tier; returns
    ``(eval_df, visual_examples)``. Config-driven: tiers, eval count, balanced
    weights, geometry and adjustment gap/mode all read from the train ``cfg``."""
    import numpy as np
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    from connectpt.routes_generator.bee_colony import get_adjustment_degrees
    from connectpt.routes_generator.improvement_learning import (
        get_batch_tensor_from_routes, make_improvement_batch, rollout_lc_improvement)
    from connectpt.routes_generator.data.route_copies import (
        redundancy_stats as _redundancy_stats)

    tiers = list(cfg.curriculum.tiers)
    eval_n_per_tier = int(cfg.report.eval_n_per_tier)
    balanced_weights = tuple(cfg.report.balanced_eval_weights)
    target_n_routes = int(cfg.data.target_n_routes)
    min_route_len = int(cfg.data.min_route_len)
    max_route_len = int(cfg.data.max_route_len)
    max_route_edit_steps = max_route_len  # was MAX_ROUTE_EDIT_STEPS = MAX_ROUTE_LEN
    max_trim_actions_per_route = int(cfg.get("max_trim_actions_per_route", 1))
    adj_gap = float(cfg.adjustment_degree_gap)
    adj_mode = str(cfg.adjustment_degree_mode)

    def _redun_t(routes_2d):
        return _redundancy_stats(routes_2d)["redundancy"]

    def _mean_metric(result, key):
        return float(result.get_metrics()[key].detach().float().mean().item())

    val_by_tier = {tier: indices[:eval_n_per_tier] for tier, indices in val_by_tier.items()}

    base_w = cost_obj.get_weights(device)

    def mkw(route_weight, conn_weight):
        weights = {key: (value.clone() if torch.is_tensor(value) else value)
                   for key, value in base_w.items()}
        weights["demand_time_weight"] = torch.as_tensor(0.0, device=device)
        weights["route_time_weight"] = torch.as_tensor(float(route_weight), device=device)
        weights["median_connectivity_weight"] = torch.as_tensor(float(conn_weight), device=device)
        return weights

    rows, visual_examples = [], {}
    conn_metric_key = ("median_connectivity_weighted"
                       if cost_obj.use_weighted_connectivity else "median_connectivity")
    transfer_metric_keys = {"d0": "$d_0$", "d1": "$d_1$", "d2": "$d_2$", "d_un": "$d_{un}$"}
    model.eval()
    weights = mkw(*balanced_weights)
    rollout_kwargs = rollout_adjustment_kwargs(model)
    for tier in tiers:
        idxs = val_by_tier[tier]
        if not idxs:
            continue
        metrics = {f"{m}_{w}": [] for m in
                   ("redun", "ATT", "RTT", "CONN", "d0", "d1", "d2", "d_un")
                   for w in ("before", "after")}
        for gi in tqdm(idxs, desc=f"eval balanced/{tier}", leave=False):
            graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, torch.tensor([gi]), device,
                training=False, target_n_routes=target_n_routes)
            with torch.no_grad():
                output = rollout_lc_improvement(
                    model, cost_obj, graph_batch, route_batch,
                    min_route_len, max_route_len, greedy=True, cost_weights=weights,
                    max_route_edit_steps=max_route_edit_steps,
                    max_trim_actions_per_route=max_trim_actions_per_route,
                    **rollout_kwargs)
            final_state, seed_result, final_result = output[:3]
            improved = get_batch_tensor_from_routes(
                final_state.routes, device, max_route_len=route_batch.shape[-1])
            metrics["redun_before"].append(_redun_t(route_batch[0]))
            metrics["redun_after"].append(_redun_t(improved[0]))
            metrics["ATT_before"].append(_mean_metric(seed_result, "ATT"))
            metrics["ATT_after"].append(_mean_metric(final_result, "ATT"))
            metrics["RTT_before"].append(_mean_metric(seed_result, "RTT"))
            metrics["RTT_after"].append(_mean_metric(final_result, "RTT"))
            metrics["CONN_before"].append(_mean_metric(seed_result, conn_metric_key))
            metrics["CONN_after"].append(_mean_metric(final_result, conn_metric_key))
            for metric_name, metric_key in transfer_metric_keys.items():
                metrics[f"{metric_name}_before"].append(_mean_metric(seed_result, metric_key))
                metrics[f"{metric_name}_after"].append(_mean_metric(final_result, metric_key))

            if tier not in visual_examples:
                nr = min(improved.shape[1], route_batch.shape[1])
                width = min(improved.shape[-1], route_batch.shape[-1])
                adj = get_adjustment_degrees(
                    improved[:, :nr, :width], route_batch[:, :nr, :width],
                    cost_obj.symmetric_routes, gap=adj_gap, mode=adj_mode).mean().item()
                visual_examples[tier] = {
                    "graph_index": gi,
                    "seed": route_batch[0].detach().cpu(),
                    "improved": improved[0].detach().cpu(), "Adj": adj,
                    "cost_before": float(seed_result.cost.detach().float().mean().item()),
                    "cost_after": float(final_result.cost.detach().float().mean().item()),
                    "redun_before": metrics["redun_before"][-1],
                    "redun_after": metrics["redun_after"][-1],
                    "ATT_before": metrics["ATT_before"][-1], "ATT_after": metrics["ATT_after"][-1],
                    "RTT_before": metrics["RTT_before"][-1], "RTT_after": metrics["RTT_after"][-1],
                    "CONN_before": metrics["CONN_before"][-1], "CONN_after": metrics["CONN_after"][-1]}

        means = {key: float(np.mean(values)) for key, values in metrics.items()}
        rows.append({"tier": tier, "n": len(idxs), **means})

    return pd.DataFrame(rows).round(4), visual_examples


def plot_balanced_examples(visual_examples, graphs, tiers):
    """Per-tier seed vs edited-network diff panels (dormant viz cell)."""
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    from connectpt.routes_generator.reports import route_plots

    if not visual_examples:
        print("Run the evaluation cell first.")
        return
    tiers_to_plot = [tier for tier in tiers if tier in visual_examples]
    fig, axes = plt.subplots(len(tiers_to_plot), 2,
                             figsize=(18, 7 * len(tiers_to_plot)),
                             squeeze=False, constrained_layout=True)
    for row_idx, tier in enumerate(tqdm(tiers_to_plot, desc="render tiers")):
        example = visual_examples[tier]
        graph = graphs[example["graph_index"]]
        route_plots.plot_plain_route_set(
            axes[row_idx, 0], example["seed"], graph,
            title=f"{tier}: corrupted seed (graph {example['graph_index']})",
            subtitle=(f"cost={example.get('cost_before', float('nan')):.3f}; "
                      f"redun={example['redun_before']:.3f}; "
                      f"ATT={example['ATT_before']:.2f}; RTT={example['RTT_before']:.2f}; "
                      f"CONN={example['CONN_before']:.2f}"))
        route_plots.plot_route_diff(
            axes[row_idx, 1], example["improved"], example["seed"], graph,
            title=f"{tier}: edited network vs seed",
            subtitle=(f"cost {example.get('cost_before', float('nan')):.3f}->{example.get('cost_after', float('nan')):.3f}; "
                      f"redun {example['redun_before']:.3f}->{example['redun_after']:.3f}; "
                      f"Adj={example['Adj']:.3f}\n"
                      f"ATT {example['ATT_before']:.2f}->{example['ATT_after']:.2f}; "
                      f"RTT {example['RTT_before']:.2f}->{example['RTT_after']:.2f}; "
                      f"CONN {example['CONN_before']:.2f}->{example['CONN_after']:.2f}"))
    fig.suptitle("Balanced validation: copy/subcopy corruption repair",
                 fontsize=15, fontweight="bold")
    plt.show(); plt.close(fig)


def post_training_convergence(cfg, *, best_model_path, benchmark_specs,
                              load_benchmark_graph):
    """Two 5-model BCO variants (RPC+trim/extend, RPC+type2) on one benchmark
    city, driven by the just-trained edit checkpoint. Config-driven: city/iters/
    alpha from cfg.report.post_train. Returns ``(rows_df, convergence)``."""
    import time as _time

    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    from connectpt.routes_generator import load_experiment
    from connectpt.routes_generator.search import BeeColonySearchRun

    city = str(cfg.report.post_train.city)
    iters = int(cfg.report.post_train.iters)
    alpha = float(cfg.report.post_train.alpha)
    print(f"[post-train] edit bee model <- {best_model_path.name}")

    spec = next(s for s in benchmark_specs if s["city"] == city)
    tensors, init = load_benchmark_graph(spec)

    # Config-first, C-native: the two ablation bee sets live in declarative configs;
    # the RPC + trim/extend edit bees use the just-trained checkpoint (set after
    # compose to avoid hydra-override parsing of a Windows path).
    variants = [("RPC + trim/extend", "e2/5model/mumford1/rpc_trim_extend", True),
                ("RPC + type2", "e2/5model/mumford1/rpc_type2", False)]
    conv, rows = {}, []
    for label, cfg_name, needs_edit in variants:
        run_cfg = load_experiment(cfg_name, overrides=[f"search.n_iterations={int(iters)}"])
        if needs_edit:
            OmegaConf.set_struct(run_cfg, False)
            run_cfg.models.edit.checkpoint_path = str(best_model_path)
        run = BeeColonySearchRun(run_cfg)
        run.setup()
        t0 = _time.perf_counter()
        _routes, _unserved, _metrics, histories = run.runner.run_seeded(
            init, tensors, eval_dims=spec, alpha=alpha, n_iterations=int(iters),
            return_histories=True)
        dt = _time.perf_counter() - t0
        h = histories[0] if histories else None
        y = (np.asarray(h.numpy() if hasattr(h, "numpy") else h).reshape(-1)
             if h is not None else None)
        m = {"label": label}
        if y is not None and y.size:
            conv[m["label"]] = y
            final, best = float(y[-1]), float(np.min(y))
        else:
            final = best = float("nan")
        rows.append(dict(model=m["label"], final_cost=round(final, 4),
                         best_cost=round(best, 4), iters=int(iters), seconds=round(dt, 1)))
        print(f"[post-train] {m['label']:18} final={final:.4f} best={best:.4f} ({dt:.0f}s)")
    return pd.DataFrame(rows), conv


def build_edit_model_and_cost(run_name, *, device, vary_weights=True,
                              route_time_weight=None, adj_weight=0.0,
                              cfg_dir=None, weights_dir=None):
    """Build ``(cfg, cost_obj, model, run_name, best_path)`` config-first.

    Replaces the notebook's old ``build_edit_run``: there is no bespoke builder
    anymore -- the model comes from ``RouteModelFactory.build_edit_model`` and
    the cost from ``CostFactory.build_unified`` over ``train/edit`` +
    ``rtt_wmc_no_demand.yaml``. Used by the standalone balanced-eval cell when
    the training cell wasn't run, to rebuild the model from its checkpoint.
    """
    from hydra import compose, initialize_config_dir

    from connectpt.routes_generator.model_factory import RouteModelFactory
    from connectpt.routes_generator.objectives import CostFactory
    from connectpt.routes_generator.core.paths import CFG_DIR, EDIT_MODEL_WEIGHTS_DIR

    cfg_dir = cfg_dir or CFG_DIR
    weights_dir = weights_dir or EDIT_MODEL_WEIGHTS_DIR
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(config_name="train/edit", overrides=[f"++run.name={run_name}"])
    model = RouteModelFactory.build_edit_model(cfg.model, cfg.experiment).to(device)
    cost_obj = CostFactory.build_unified("rtt_wmc_no_demand", for_training=True)
    cost_obj.variable_weights = bool(vary_weights)
    if route_time_weight is not None:
        cost_obj.route_time_weight = float(route_time_weight)
    cost_obj.adjustment_degree_weight = float(adj_weight)
    cost_obj.ignore_stops_oob = True
    cost_obj.to(device)
    best_path = weights_dir / f"{run_name}.pt"
    return cfg, cost_obj, model, run_name, best_path


def rollout_adjustment_kwargs(model, *, target=None):
    """Adjustment-conditioning rollout kwargs (empty when the model has no
    adjustment-target conditioning feature, which is the paper default)."""
    from connectpt.routes_generator.objectives import load_unified_objective
    _obj = load_unified_objective()
    ADJ_TARGET, ADJ_WEIGHT = _obj.adj_target, _obj.adj_weight
    ADJ_GAP, ADJ_MODE = _obj.adj_gap, _obj.adj_mode

    n_adj_feats = int(getattr(model, "n_adjustment_cond_feats", 0) or 0)
    if n_adj_feats <= 0:
        return {}
    target = ADJ_TARGET if target is None else target
    kwargs = dict(adjustment_target=float(target), adjustment_use_current=False,
                  adjustment_gap=ADJ_GAP, adjustment_mode=ADJ_MODE)
    if n_adj_feats > 1:
        kwargs["adjustment_weight"] = float(ADJ_WEIGHT)
    return kwargs
