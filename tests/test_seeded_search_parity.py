"""Seeded bee-colony parity: library-native config-first path == builder path.

paper_combined seeds BCO from a concrete initial network (never from scratch).
This test proves the library-native seeded path -- cost built by
``CostFactory.build_unified``, dataset via the library loader,
``run_seeded_bee_colony`` -- reproduces the eval_lib ``build_bco_cfg`` + ``run_bco``
builder path bit-for-bit, given the same seeded init routes, tensors and params.

Heuristic (no-model) variant on Mandl: with no neural model there is no
model-init RNG between seeding and the bee-colony loop, so the two paths must
match exactly. This is the parity gate for migrating the notebook off the
builders onto the library runner.
"""
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTE_EXAMPLES = REPO_ROOT / "examples" / "route_generator"
if str(ROUTE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROUTE_EXAMPLES))

from connectpt.routes_generator.citygraph_dataset import get_dataset_from_config
from connectpt.routes_generator.core.checkpoints import CheckpointStore
from connectpt.routes_generator.core.paths import CONSTRUCTION_MODEL_WEIGHTS_PATH
from connectpt.routes_generator.core.runtime import seed_everything
from connectpt.routes_generator.model_factory import RouteModelFactory
from connectpt.routes_generator.objectives import CostFactory, load_unified_objective
from connectpt.routes_generator.search.executable_plan import ExecutablePlan
from connectpt.routes_generator.search.seeded_search import run_seeded_bee_colony
from connectpt.routes_generator.torch_utils import get_batch_tensor_from_routes


def _as_tensor(routes):
    """Route list/tensor -> a comparable cpu batch tensor (like as_route_tensor)."""
    if isinstance(routes, torch.Tensor):
        return routes.detach().cpu()
    return get_batch_tensor_from_routes(routes).detach().cpu()

CITY = "Mandl"
N_ITERS = 2


def _init_routes(spec, tensors):
    """Deterministic seeded initial network (greedy random walk on the street graph)."""
    import random
    rng = random.Random(0)
    n = tensors["node_locs"].shape[0]
    adj = tensors["street_adj"]
    nbrs = {u: [v for v in range(n) if v != u and torch.isfinite(adj[u, v])]
            for u in range(n)}
    routes, uncov = [], set(range(n))
    for _ in range(spec["n_routes"]):
        start = rng.choice(sorted(uncov)) if uncov else rng.randrange(n)
        path = [start]
        while len(path) < spec["max_route_len"]:
            choices = [v for v in nbrs[path[-1]] if v not in path]
            if not choices:
                break
            path.append(rng.choice([v for v in choices if v in uncov] or choices))
        routes.append(path)
        uncov -= set(path)
    R = torch.full((1, spec["n_routes"], spec["max_route_len"]), -1, dtype=torch.long)
    for i, p in enumerate(routes):
        R[0, i, :len(p)] = torch.tensor(p)
    return R


def _run_builder_path(spec, tensors, R, use_neural):
    from eval_lib.helpers import build_bco_cfg, run_bco
    from eval_lib.experiments import bco_cfg_set, set_cfg_value
    from eval_lib.paper import UNIFIED_ADJ

    cfg = build_bco_cfg("parity_builder", spec["n_routes"], spec["min_route_len"],
                        spec["max_route_len"], use_neural_bees=use_neural, n_bees=10,
                        n_type1_bees=5, n_type2_bees=5,
                        connectivity_mode="median_weighted",
                        **load_unified_objective().weights)
    bco_cfg_set(cfg, n_iterations=N_ITERS, **UNIFIED_ADJ)
    set_cfg_value(cfg, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
    set_cfg_value(cfg, "experiment.seed", 0)
    seed_everything(0)
    _run, _metrics, unserved, routes, _mc = run_bco(cfg, R, tensors=tensors)
    return _as_tensor(routes)


def _run_library_native_path(spec, tensors, R, use_neural):
    obj = load_unified_objective()
    device = torch.device("cpu")
    # neural rebuild bee: build the construction model config-first (factory +
    # strict checkpoint load), matching seed->build->run order so model-init RNG
    # aligns with the builder path.
    seed_everything(0)
    bee_model = None
    if use_neural:
        bee_model = RouteModelFactory.build_construction_model_by_name("construction")
        CheckpointStore.load_model_weights(
            bee_model, CONSTRUCTION_MODEL_WEIGHTS_PATH, strict=True, map_location=device)
        bee_model.to(device).eval()
    cost = CostFactory.build_unified("rtt_wmc_no_demand", for_training=False)
    cost.to(device)
    dataset = get_dataset_from_config(OmegaConf.create({"type": "tensor"}), tensors=tensors)
    dataloader = DataLoader(dataset, batch_size=1)
    eval_cfg = OmegaConf.create({
        "n_routes": spec["n_routes"], "min_route_len": spec["min_route_len"],
        "max_route_len": spec["max_route_len"],
    })
    search_cfg = OmegaConf.create({
        "n_bees": 10, "n_iterations": N_ITERS,
        "n_type1_bees": 5, "n_type2_bees": 5,
        "neural_bees": use_neural,
        **obj.adj_kwargs,
    })
    plan = ExecutablePlan.from_flat_cfg(search_cfg, bee_model=bee_model)
    _mc, _sc, unserved, metrics, routes = run_seeded_bee_colony(
        dataloader, eval_cfg, cost, R, search_cfg=search_cfg,
        plan=plan, device=device, silent=True)
    return _as_tensor(routes)


@pytest.mark.parametrize("use_neural", [False, True])
def test_library_native_seeded_bco_matches_builder_path(use_neural):
    if use_neural and not CONSTRUCTION_MODEL_WEIGHTS_PATH.exists():
        pytest.skip(f"construction weights not present: {CONSTRUCTION_MODEL_WEIGHTS_PATH}")
    from eval_lib.baselines import load_benchmark_tensors, BENCHMARK_SPECS

    spec = next(s for s in BENCHMARK_SPECS if s["city"] == CITY)
    tensors = load_benchmark_tensors(CITY)
    R = _init_routes(spec, tensors)

    routes_builder = _run_builder_path(spec, tensors, R, use_neural)
    routes_library = _run_library_native_path(spec, tensors, R, use_neural)

    assert routes_builder.shape == routes_library.shape
    assert torch.equal(routes_builder, routes_library), (
        f"library-native seeded BCO diverged from the builder path (neural={use_neural})"
    )
