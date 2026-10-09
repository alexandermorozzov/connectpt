"""TrainingDataModule -- load graphs / seed routes, split, batch, curriculum.

Encapsulates the data plumbing the training notebook did inline: loading the
raw graphs + LC seed routes, the tier-stratified train/val split, the
curriculum schedule functions, and the make_batch wrapper. City batching lives
in ``city.graph``; route loading and fitting use ``network.io`` and
``network.routes``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..city.graph import STOP_KEY
from ..city.synthetic import load_graphs
from ..network.io import load_routes
from ..network.routes import to_routes
from .params import EditDatasetParams
from .curriculum import prepare_curriculum
from .pipeline import GraphPipeline
from .split import SplitParams, graph_source_id, prepare_split


@dataclass(frozen=True)
class EditDataset:
    params: EditDatasetParams
    train_pipeline: GraphPipeline
    evaluation_pipeline: GraphPipeline

    def prepare(self, cost, batch_size, evaluation, device, env_params, curriculum, resample_preferences):
        folder = Path(self.params.dataset_dirname)
        if not folder.is_absolute():
            raise ValueError("edit dataset directory must be an absolute Path")
        module = TrainingDataModule(
            folder / "raw_graphs_1000.pkl", folder, device, self.params.min_route_len,
            self.params.max_route_len, self.params.target_n_routes, self.train_pipeline, self.evaluation_pipeline,
        )
        module, stages, split = load(module, self.params, curriculum)
        data = EditData(module, stages, split.train, split.validation, split.monitor, batch_size)
        data.training_count = len(split.train)
        data.validation_count = len(split.monitor)
        return data, cost


@dataclass
class TrainingDataModule:
    """Owns the training graphs/routes and how they are split, batched, staged."""

    raw_graphs_path: str | Path
    lc_results_dir: str | Path
    device: torch.device | str
    min_route_len: int | None
    max_route_len: int | None
    target_n_routes: int | None
    train_pipeline: GraphPipeline
    evaluation_pipeline: GraphPipeline

    graphs: list = field(default_factory=list, init=False)
    seed_routes: torch.Tensor | None = field(default=None, init=False)

    # --- loading ------------------------------------------------------------
    def setup(self) -> "TrainingDataModule":
        """Load graphs and LC seed routes into memory."""
        if not Path(self.raw_graphs_path).exists():
            raise FileNotFoundError(
                f"training dataset not found: {self.raw_graphs_path}. "
                f"Build it once with `python scripts/run.py --config-name runs/prepare_dataset` "
                f"(about an hour for the default 1000 graphs)."
            )
        self.graphs = load_graphs(self.raw_graphs_path)
        self.seed_routes = load_routes(self.lc_results_dir)
        if len(self.graphs) != self.seed_routes.shape[0]:
            raise ValueError("graph count does not match seed network count")
        self.seed_routes = to_routes(
            self.seed_routes,
            width=min(self.seed_routes.shape[-1], max(int(g[STOP_KEY].num_nodes) for g in self.graphs)),
        )
        return self

    def __len__(self) -> int:
        return len(self.graphs)

    # --- batching (delegates to the existing builder) -----------------------
    def make_batch(self, indices, *, training: bool = False):
        """Build a (graph_batch, route_batch) for ``indices``."""
        pipeline = self.train_pipeline if training else self.evaluation_pipeline
        graph_batch = pipeline.prepare([self.graphs[int(i)] for i in indices], self.device)
        node_capacity = int((graph_batch[STOP_KEY].ptr[1:] - graph_batch[STOP_KEY].ptr[:-1]).max())
        routes = to_routes(
            self.seed_routes[torch.as_tensor(indices, dtype=torch.long)],
            n_routes=self.target_n_routes,
            width=min(self.seed_routes.shape[-1], node_capacity),
        ).to(self.device)
        return graph_batch, routes


def _graph_tensors(g):
    """node_locs / street_adj / demand dict for one graph (shared helper)."""
    from connectpt.routes_generator.city.graph import STOP_KEY

    return {
        "node_locs": g[STOP_KEY].pos.detach().cpu().clone(),
        "street_adj": g.street_adj.detach().cpu().clone(),
        "demand": g.demand.detach().cpu().clone(),
    }


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
    lc_n_samples: int
    tier_cfg: dict
    force_regen: bool
    policy_profile: Any
    policy_cost: Any
    policy_seed: int
    batch_size_cpu: int
    batch_size_cuda: int
    corruption_seed: int


def load(module, dataset, curriculum_params):
    """Load seed routes and tier splits; the schedule contains data, not callbacks."""
    import pandas as pd

    module.setup()
    meta = pd.read_csv(Path(module.lc_results_dir) / "meta.csv")
    tier_of = dict(zip(meta.graph_index.astype(int), meta.tier))
    if set(tier_of) != set(range(len(module.graphs))) or len(meta) != len(module.graphs):
        raise ValueError('metadata must contain every graph exactly once')
    tiers = tuple(tier_of[i] for i in range(len(module.graphs)))
    split = prepare_split(
        [graph_source_id(g) for g in module.graphs], tiers,
        SplitParams('stratified', dataset.train_fraction, dataset.split_seed, dataset.n_val_per_tier),
    )
    return module, prepare_curriculum(split.train, tiers, curriculum_params), split


class EditData:
    def __init__(self, module, curriculum, train_indices, val_indices, monitor_indices, batch_size):
        self.module, self.curriculum = module, curriculum
        self.train_indices, self.val_indices = train_indices, val_indices
        self.all_val_indices = val_indices
        self.monitor_indices = monitor_indices
        self.batch_size = min(batch_size, len(train_indices))
        self.device = module.device
        self.indices = self.order = None
        self.label = "all"
        self.index_cursor = self.route_cursor = 0
        self.graphs = self.working = self.reference = self.preferences = None
        self.fresh = True
        self.graph_instances_loaded = self.graph_batches_loaded = 0
        self.episode_rewards = self.episode_lengths = None

    def initialize(self):
        self.indices = self.train_indices
        self.order = self.indices[torch.randperm(len(self.indices))]

    def warmup(self, count):
        for indices in list(self.train_indices.split(self.batch_size))[:count]:
            yield self.module.make_batch(indices, training=True)

    def get(self, progress):
        indices, label = self.curriculum.get(progress)
        if self.label != label:
            self.indices, self.label = indices, label
            self.order = indices[torch.randperm(len(indices))]
            self.index_cursor = 0
        self.fresh = True
        return self

    def start(self, env):
        self.prepare_step(env)

    def prepare_step(self, env):
        if not self.fresh and not env.is_done().all():
            return
        if not self.fresh:
            self.working = to_routes(env.updated_network(), width=self.reference.shape[-1])
        if self.fresh or self.route_cursor == 0:
            if self.index_cursor + self.batch_size > len(self.order):
                self.order = self.indices[torch.randperm(len(self.indices))]
                self.index_cursor = 0
            indices = self.order[self.index_cursor : self.index_cursor + self.batch_size]
            self.index_cursor += self.batch_size
            self.graphs, self.working = self.module.make_batch(indices, training=True)
            width = max(self.working.shape[-1], env.params.max_route_len)
            self.working = to_routes(self.working, width=width)
            self.reference = self.working.clone()
            self.preferences = env.cost.sample_variable_weights(self.graphs.num_graphs, self.device)
            self.route_cursor = 0
            self.graph_instances_loaded += self.graphs.num_graphs
            self.graph_batches_loaded += 1
        route_idx = self.route_cursor
        self.route_cursor = (route_idx + 1) % self.working.shape[1]
        env.reset(self.graphs, self.preferences, self.working, route_idx=route_idx, reference_networks=self.reference)
        self.episode_rewards = torch.zeros(self.graphs.num_graphs, device=self.device)
        self.episode_lengths = torch.zeros(self.graphs.num_graphs, dtype=torch.long, device=self.device)
        self.fresh = False

    def after_step(self, env, done):
        pass

    def bootstrap_mask(self, env):
        return ~env.is_done()

    def final_validation(self):
        graphs, networks = self.module.make_batch(self.all_val_indices)
        return graphs.to_data_list(), networks


def build(cfg: CopyTierConfig):
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

    from connectpt.routes_generator.city.synthetic import DynamicCityGraphDataset
    from torch_geometric.loader import DataLoader
    from ..city.graph import to_batch
    from ..env.base import EnvParams
    from ..env.construction import ConstructionEnv
    from ..policy.build import build_policy
    from ..policy.predict import predict
    from ..common.random import seed_everything
    from connectpt.routes_generator.network.io import save_routes
    from connectpt.routes_generator.network.routes import to_routes
    from connectpt.routes_generator.network.corruption import (
        count_changed_routes,
        corrupt,
        copy_tier,
        CorruptionCity,
        CorruptionLimits,
        redundancy_stats,
        uncovered_demand_pct,
    )

    def _to_fixed(routes):
        return to_routes(routes, n_routes=cfg.target_n_routes, width=cfg.max_route_len)[0]

    _tensors = _graph_tensors

    def generate_dataset():
        if cfg.new_dataset_dir.exists():
            shutil.rmtree(cfg.new_dataset_dir)
        cfg.new_dataset_dir.mkdir(parents=True, exist_ok=True)
        _random.seed(cfg.raw_graph_seed)
        torch.manual_seed(cfg.raw_graph_seed)
        ds = DynamicCityGraphDataset(
            min_nodes=cfg.raw_n_nodes,
            max_nodes=cfg.raw_n_nodes,
            data_type=cfg.raw_graph_type,
            mumford_style=True,
            pos_only=False,
        )
        raw = [ds.generate_graph(n_nodes=cfg.raw_n_nodes) for _ in range(cfg.n_graphs)]
        per = cfg.n_graphs // len(cfg.tiers)
        tensors_all = [_tensors(g) for g in raw]

        # --- batched learned construction on the GPU --------------------------
        # Group graphs by LC objective combo (one cfg / one set of cost weights
        # per combo), then construct each group in GPU batches of LC_GEN_BATCH
        # via generate (one forward per batch) instead of one graph at a
        # time -- this is what actually keeps the GPU busy during dataset gen.
        LC_GEN_BATCH = cfg.batch_size_cuda if cfg.policy_profile.device.type == "cuda" else cfg.batch_size_cpu
        lc_routes = [None] * cfg.n_graphs
        for _ci, (d, rt, cn, ctag) in enumerate(cfg.lc_combos):
            combo_idxs = [gi for gi in range(cfg.n_graphs) if gi % len(cfg.lc_combos) == _ci]
            if not combo_idxs:
                continue
            cost = cfg.policy_cost.with_params(
                demand_time_weight=d, route_time_weight=rt, median_connectivity_weight=cn
            )
            for _s in tqdm(range(0, len(combo_idxs), LC_GEN_BATCH), desc=f"LC construct [{ctag}]"):
                chunk = combo_idxs[_s : _s + LC_GEN_BATCH]
                seed_everything(cfg.policy_seed)
                policy = build_policy(cfg.policy_profile)
                graphs = [to_batch(tensors_all[gi]).to_data_list()[0] for gi in chunk]
                data = next(iter(DataLoader(graphs, batch_size=len(chunk)))).to(cfg.policy_profile.device)
                limits = EnvParams(cfg.target_n_routes, cfg.min_route_len, cfg.max_route_len,
                                   True, 1.0, policy.force_linking_unlinked, policy.max_act_len,
                                   None, None, False, False, False)
                routes_b = predict(policy, data, cost.get_weights(), cost, ConstructionEnv,
                                   limits, "sample", cfg.lc_n_samples,
                                   sample_batch_size=len(chunk)).routes
                for _j, gi in enumerate(chunk):
                    lc_routes[gi] = routes_b[_j]

        # --- per-graph corruption + dump (CPU) --------------------------------
        subset, meta = [], []
        for gi, g in enumerate(tqdm(raw, desc="corrupt + dump")):
            tier = cfg.tiers[min(gi // per, len(cfg.tiers) - 1)]
            tier_cfg = cfg.tier_cfg[tier]
            rng = _random.Random(cfg.corruption_seed + gi)
            tn = tensors_all[gi]
            ctag = cfg.lc_combos[gi % len(cfg.lc_combos)][3]
            routes = _to_fixed(lc_routes[gi])
            before = redundancy_stats(routes)
            clean_routes = routes.clone()
            corruption = corrupt(
                routes,
                rng,
                CorruptionCity(tn["street_adj"], tn["demand"], cfg.raw_n_nodes),
                CorruptionLimits(cfg.min_route_len, cfg.max_route_len),
                copy_tier(tier_cfg),
            )
            routes, event_counts = corruption.routes, corruption.applied_events
            after = redundancy_stats(routes)
            n_corrupted = count_changed_routes(clean_routes, routes)
            gdir = cfg.new_dataset_dir / f"graph_{gi:04d}"
            gdir.mkdir(parents=True, exist_ok=True)
            save_routes(gdir / f"lc_copy_cur_graph_{gi:04d}_routes_routes.pkl", routes)
            subset.append(g)
            meta.append(
                {
                    "graph_index": gi,
                    "tier": tier,
                    "tier_kind": ("clean" if not tier_cfg.kinds else "copies"),
                    "lc_combo": ctag,
                    "applied_events": int(sum(event_counts.values())),
                    "n_corrupted_routes": int(n_corrupted),
                    "mutation_events": json.dumps(dict(event_counts), sort_keys=True),
                    "d_un_after_pct": round(uncovered_demand_pct(routes, tn["demand"], cfg.raw_n_nodes), 2),
                    "redun_before": round(before["redundancy"], 4),
                    "redun_after": round(after["redundancy"], 4),
                    "max_leg_use_before": before["max_leg_use"],
                    "max_leg_use_after": after["max_leg_use"],
                }
            )
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
