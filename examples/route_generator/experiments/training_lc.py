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

    from connectpt.routes_generator.citygraph_dataset import (
        STOP_KEY, DynamicCityGraphDataset)
    from eval_lib import as_route_tensor, build_lc_cfg, dump_routes, run_lc_batch
    from eval_lib.route_copies import (count_changed_routes, inject_route_copies,
                                       redundancy_stats, uncovered_demand_pct)

    def _to_fixed(routes):
        t = as_route_tensor(routes).long()
        if t.ndim == 3:
            t = t[0]
        if t.shape[0] < cfg.target_n_routes:
            t = torch.cat([t, torch.full((cfg.target_n_routes - t.shape[0], t.shape[1]), -1, dtype=t.dtype)], 0)
        else:
            t = t[:cfg.target_n_routes]
        if t.shape[1] < cfg.max_route_len:
            t = torch.cat([t, torch.full((t.shape[0], cfg.max_route_len - t.shape[1]), -1, dtype=t.dtype)], 1)
        elif t.shape[1] > cfg.max_route_len:
            t = t[:, :cfg.max_route_len]
        return t

    def _tensors(g):
        return {"node_locs": g[STOP_KEY].pos.detach().cpu().clone(),
                "street_adj": g.street_adj.detach().cpu().clone(),
                "demand": g.demand.detach().cpu().clone()}

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
    from eval_lib.context import CFG_DIR, EDIT_MODEL_WEIGHTS_DIR

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
    from eval_lib.params import ADJ_TARGET, ADJ_WEIGHT, ADJ_GAP, ADJ_MODE

    n_adj_feats = int(getattr(model, "n_adjustment_cond_feats", 0) or 0)
    if n_adj_feats <= 0:
        return {}
    target = ADJ_TARGET if target is None else target
    kwargs = dict(adjustment_target=float(target), adjustment_use_current=False,
                  adjustment_gap=ADJ_GAP, adjustment_mode=ADJ_MODE)
    if n_adj_feats > 1:
        kwargs["adjustment_weight"] = float(ADJ_WEIGHT)
    return kwargs
