"""Build the copy-tier training dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.runs import ExperimentRun, RunArtifact


def copytier_config(cfg) -> "CopyTierConfig":
    """Build the dataset-generation config from a composed train cfg
    (reads ``cfg.dataset_build`` and ``cfg.dataset``)."""
    from omegaconf import OmegaConf

    from connectpt.routes_generator.core.paths import DATASETS_DIR
    from connectpt.routes_generator.data.route_copies import COPY_TIER_CFG

    dg, data = cfg.dataset_build, cfg.dataset
    ddir = DATASETS_DIR / data.dataset_dirname
    tiers = list(dg.tiers) if dg.get("tiers") else list(COPY_TIER_CFG)
    combos = [list(c) for c in OmegaConf.to_container(dg.lc_combos, resolve=True)]
    return CopyTierConfig(
        new_dataset_dir=ddir, subset_pkl=ddir / "raw_graphs_1000.pkl",
        meta_csv=ddir / "meta.csv", raw_graph_seed=int(dg.raw_graph_seed),
        raw_n_nodes=int(dg.raw_n_nodes), raw_graph_type=str(dg.raw_graph_type),
        n_graphs=int(dg.n_graphs), tiers=tiers, lc_combos=combos,
        target_n_routes=int(data.target_n_routes), min_route_len=int(data.min_route_len),
        max_route_len=int(data.max_route_len),
        connectivity_mode=str(cfg.cost.kwargs.connectivity_mode),
        lc_n_samples=int(dg.lc_n_samples), tier_cfg=COPY_TIER_CFG,
        force_regen=bool(dg.get("force_regen", False)))


def _graph_tensors(g):
    """node_locs / street_adj / demand dict for one graph (shared helper)."""
    from connectpt.routes_generator.citygraph_dataset import STOP_KEY

    return {"node_locs": g[STOP_KEY].pos.detach().cpu().clone(),
            "street_adj": g.street_adj.detach().cpu().clone(),
            "demand": g.demand.detach().cpu().clone()}


@dataclass
class CopyTierConfig:
    """Geometry, paths and tier schedule for the copy-tier training dataset.

    Built by :func:`copytier_config` from the training config and passed to
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
    force_regen: bool = False


def build_copytier_dataset(cfg: CopyTierConfig):
    """Generate the four-tier route-copy training dataset (idempotent).

    Returns seconds/graph if it (re)generated, or ``None`` if a complete dataset
    already existed and was reused. Learned construction batched on the GPU per
    LC-objective combo, then per-graph copy/subcopy corruption + dump.
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
    from connectpt.routes_generator.data.routes import pad_routes_to
    from connectpt.routes_generator.data.route_copies import (
        count_changed_routes, inject_route_copies, redundancy_stats,
        uncovered_demand_pct)

    def _to_fixed(routes):
        return pad_routes_to(routes, cfg.target_n_routes, cfg.max_route_len)

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


class DatasetPreparationRun(ExperimentRun):
    """Generate the dataset described by a composed ``runs/prepare_dataset`` cfg."""

    def setup(self) -> None:
        self.dataset = copytier_config(self.cfg)

    def run(self, *, dry_run: bool = False):
        self.setup()
        seconds = None if dry_run else build_copytier_dataset(self.dataset)
        return RunArtifact(
            run_name=str(self.cfg.run.name),
            output_dir=Path(self.dataset.new_dataset_dir),
            metadata={"dry_run": dry_run, "seconds_per_graph": seconds,
                      "dataset": self.dataset},
        )
