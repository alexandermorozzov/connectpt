"""U1 end-to-end: BeeColonySearchRun runs SEEDED from one composed cfg.

Proves the config-first rich path -- compose an experiment (base + objective +
golden models + our_nbco bee_set), build cost/models config-first, translate the
declarative plan to bee_colony kwargs, and improve an EXISTING network passed to
``run(init_routes=, tensors=)`` -- reproduces the flat/golden our_nbco run
bit-for-bit. Both paths reseed from seed 0 immediately before the bee-colony
loop, so the comparison isolates the cfg/plan translation + config-first model
building (construction via factory; edit via the edit_trim_seeded config, which
matches golden's bestsofar_feb2023_trim / allow_trim_below_min=false).
"""
import random
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
LIB_CFG = REPO / "connectpt" / "routes_generator" / "cfg"
ROUTE_EX = REPO / "examples" / "route_generator"
if str(ROUTE_EX) not in sys.path:
    sys.path.insert(0, str(ROUTE_EX))

from connectpt.routes_generator.citygraph_dataset import get_dataset_from_config
from connectpt.routes_generator.core.checkpoints import CheckpointStore
from connectpt.routes_generator.core.paths import (
    CONSTRUCTION_MODEL_WEIGHTS_PATH, EDIT_MODEL_WEIGHTS_DIR)
from connectpt.routes_generator.core.runtime import seed_everything
from connectpt.routes_generator.model_factory import RouteModelFactory
from connectpt.routes_generator.objectives import CostFactory
from connectpt.routes_generator.search.edit_bee import build_edit_bee_model
from connectpt.routes_generator.search.compat import plan_from_flat_cfg
from connectpt.routes_generator.search.runs import BeeColonySearchRun
from connectpt.routes_generator.search.seeded_search import run_seeded_bee_colony
from connectpt.routes_generator.torch_utils import get_batch_tensor_from_routes

EDIT_CKPT = EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_rttconn_adj_w10_t02_finetune100.pt"
N_ITERS = 3


def _as(r):
    return r.detach().cpu() if torch.is_tensor(r) else get_batch_tensor_from_routes(r).detach().cpu()


def _init_routes(spec, tensors):
    rng = random.Random(0)
    n = tensors["node_locs"].shape[0]
    adj = tensors["street_adj"]
    nbrs = {u: [v for v in range(n) if v != u and torch.isfinite(adj[u, v])] for u in range(n)}
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


def test_seeded_beecolonysearchrun_matches_flat_our_nbco():
    if not (CONSTRUCTION_MODEL_WEIGHTS_PATH.exists() and EDIT_CKPT.exists()):
        pytest.skip("construction/edit weights not present")
    from connectpt.routes_generator.data.loaders import BENCHMARK_SPECS, load_benchmark_tensors
    from connectpt.routes_generator.paper_experiments.cfg_compose import load_experiment_cfg
    from connectpt.routes_generator.paper_experiments.cfg_compose import bco_cfg_set, set_cfg_value
    from connectpt.routes_generator.paper_experiments.macsa import UNIFIED_ADJ

    spec = next(s for s in BENCHMARK_SPECS if s["city"] == "Mumford0")
    tensors = load_benchmark_tensors("Mumford0")
    R = _init_routes(spec, tensors)
    eval_dims = {"n_routes": spec["n_routes"], "min_route_len": spec["min_route_len"],
                 "max_route_len": spec["max_route_len"]}

    # B -- config-first BeeColonySearchRun, seeded on the provided network
    with initialize_config_dir(config_dir=str(LIB_CFG), version_base=None):
        cfg = compose(config_name="experiments/seeded/our_nbco_mumford0",
                      overrides=[f"search.n_iterations={N_ITERS}"])
    art = BeeColonySearchRun(cfg).run(init_routes=R, tensors=tensors, eval_dims=eval_dims)
    routes_b = _as(art.result["routes"])

    # A -- flat/golden our_nbco config, shared-style models, reseed before run
    device = torch.device("cpu")
    cost = CostFactory.build_unified("rtt_wmc_no_demand", for_training=False); cost.to(device)
    construction = RouteModelFactory.build_construction_model_by_name("construction")
    CheckpointStore.load_model_weights(construction, CONSTRUCTION_MODEL_WEIGHTS_PATH,
                                       strict=True, map_location=device)
    construction.to(device).eval()
    edit = build_edit_bee_model(device, EDIT_CKPT, n_adjustment_cond_feats=0)
    dl = DataLoader(get_dataset_from_config(OmegaConf.create({"type": "tensor"}), tensors=tensors),
                    batch_size=1)
    flat = load_experiment_cfg("nbco_variants/our_nbco_mumford0")
    for k, v in eval_dims.items():
        set_cfg_value(flat, f"eval.{k}", int(v))
    bco_cfg_set(flat, n_iterations=N_ITERS, **UNIFIED_ADJ)
    set_cfg_value(flat, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
    seed_everything(0)
    plan = plan_from_flat_cfg(flat, bee_model=construction, edit_model=edit)
    out = run_seeded_bee_colony(dl, OmegaConf.create(eval_dims), cost, R, search_cfg=flat,
                                plan=plan, device=device, silent=True)
    routes_a = _as(out[4])

    assert routes_b.shape == routes_a.shape
    assert torch.equal(routes_b, routes_a), "seeded BeeColonySearchRun diverged from flat our_nbco"


def test_run_sweep_varies_alpha_per_point():
    """run_sweep reconfigures the RTT/WMC trade-off per point and runs one seeded
    search each -- distinct alphas give distinct route sets, and the cost is left
    reconfigured to the last swept point."""
    if not (CONSTRUCTION_MODEL_WEIGHTS_PATH.exists() and EDIT_CKPT.exists()):
        pytest.skip("construction/edit weights not present")
    from connectpt.routes_generator.data.loaders import BENCHMARK_SPECS, load_benchmark_tensors

    spec = next(s for s in BENCHMARK_SPECS if s["city"] == "Mumford0")
    tensors = load_benchmark_tensors("Mumford0")
    R = _init_routes(spec, tensors)
    eval_dims = {"n_routes": spec["n_routes"], "min_route_len": spec["min_route_len"],
                 "max_route_len": spec["max_route_len"]}

    with initialize_config_dir(config_dir=str(LIB_CFG), version_base=None):
        cfg = compose(config_name="experiments/seeded/our_nbco_mumford0",
                      overrides=[f"search.n_iterations={N_ITERS}"])
    run = BeeColonySearchRun(cfg)
    run.setup()
    rows = run.runner.run_sweep(R, tensors, eval_dims=eval_dims,
                                alpha_grid=[0.0, 1.0], adj_targets=[0.2])

    assert [r["alpha"] for r in rows] == [0.0, 1.0]
    r0, r1 = _as(rows[0]["routes"]), _as(rows[1]["routes"])
    assert not torch.equal(r0, r1), "alpha=0 and alpha=1 gave identical routes"
    # cost left at the last point (alpha=1.0 -> route_time_weight=1, conn=0)
    assert run.cost_obj.route_time_weight == 1.0
    assert run.cost_obj.median_connectivity_weight == 0.0


def test_seeded_classic_bco_matches_flat():
    """Heuristic classic BCO (no neural models) via BeeColonySearchRun == flat
    nbco_variants/classic_bco_mumford0, seeded. Validates the type1-heuristic +
    type2 path end-to-end (no weights needed)."""
    from connectpt.routes_generator.data.loaders import BENCHMARK_SPECS, load_benchmark_tensors
    from connectpt.routes_generator.paper_experiments.cfg_compose import load_experiment_cfg
    from connectpt.routes_generator.paper_experiments.cfg_compose import bco_cfg_set, set_cfg_value
    from connectpt.routes_generator.paper_experiments.macsa import UNIFIED_ADJ

    spec = next(s for s in BENCHMARK_SPECS if s["city"] == "Mumford0")
    tensors = load_benchmark_tensors("Mumford0")
    R = _init_routes(spec, tensors)
    eval_dims = {"n_routes": spec["n_routes"], "min_route_len": spec["min_route_len"],
                 "max_route_len": spec["max_route_len"]}

    with initialize_config_dir(config_dir=str(LIB_CFG), version_base=None):
        cfg = compose(config_name="experiments/seeded/classic_bco_mumford0",
                      overrides=[f"search.n_iterations={N_ITERS}"])
    art = BeeColonySearchRun(cfg).run(init_routes=R, tensors=tensors, eval_dims=eval_dims)
    routes_b = _as(art.result["routes"])

    cost = CostFactory.build_unified("rtt_wmc_no_demand", for_training=False)
    cost.to(torch.device("cpu"))
    dl = DataLoader(get_dataset_from_config(OmegaConf.create({"type": "tensor"}), tensors=tensors),
                    batch_size=1)
    flat = load_experiment_cfg("nbco_variants/classic_bco_mumford0")
    for k, v in eval_dims.items():
        set_cfg_value(flat, f"eval.{k}", int(v))
    bco_cfg_set(flat, n_iterations=N_ITERS, **UNIFIED_ADJ)
    set_cfg_value(flat, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
    seed_everything(0)
    plan = plan_from_flat_cfg(flat)
    out = run_seeded_bee_colony(dl, OmegaConf.create(eval_dims), cost, R, search_cfg=flat,
                                plan=plan, device=torch.device("cpu"), silent=True)
    routes_a = _as(out[4])

    assert routes_b.shape == routes_a.shape
    assert torch.equal(routes_b, routes_a), "seeded classic BCO diverged from flat"


def test_seeded_neural_bco_matches_flat():
    """Neural BCO (type-1 neural rebuild + type-2 random edit, construction model
    only) via BeeColonySearchRun.run(seeded) == flat nbco_variants/neural_bco_mumford0
    bit-for-bit. Completes the E1 method parity gate (neural_bco + our_nbco)."""
    if not CONSTRUCTION_MODEL_WEIGHTS_PATH.exists():
        pytest.skip("construction weights not present")
    from connectpt.routes_generator.data.loaders import BENCHMARK_SPECS, load_benchmark_tensors
    from connectpt.routes_generator.paper_experiments.cfg_compose import load_experiment_cfg
    from connectpt.routes_generator.paper_experiments.cfg_compose import bco_cfg_set, set_cfg_value
    from connectpt.routes_generator.paper_experiments.macsa import UNIFIED_ADJ

    spec = next(s for s in BENCHMARK_SPECS if s["city"] == "Mumford0")
    tensors = load_benchmark_tensors("Mumford0")
    R = _init_routes(spec, tensors)
    eval_dims = {"n_routes": spec["n_routes"], "min_route_len": spec["min_route_len"],
                 "max_route_len": spec["max_route_len"]}

    with initialize_config_dir(config_dir=str(LIB_CFG), version_base=None):
        cfg = compose(config_name="experiments/seeded/neural_bco_mumford0",
                      overrides=[f"search.n_iterations={N_ITERS}"])
    art = BeeColonySearchRun(cfg).run(init_routes=R, tensors=tensors, eval_dims=eval_dims)
    routes_b = _as(art.result["routes"])

    device = torch.device("cpu")
    cost = CostFactory.build_unified("rtt_wmc_no_demand", for_training=False); cost.to(device)
    construction = RouteModelFactory.build_construction_model_by_name("construction")
    CheckpointStore.load_model_weights(construction, CONSTRUCTION_MODEL_WEIGHTS_PATH,
                                       strict=True, map_location=device)
    construction.to(device).eval()
    dl = DataLoader(get_dataset_from_config(OmegaConf.create({"type": "tensor"}), tensors=tensors),
                    batch_size=1)
    flat = load_experiment_cfg("nbco_variants/neural_bco_mumford0")
    for k, v in eval_dims.items():
        set_cfg_value(flat, f"eval.{k}", int(v))
    bco_cfg_set(flat, n_iterations=N_ITERS, **UNIFIED_ADJ)
    set_cfg_value(flat, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
    seed_everything(0)
    plan = plan_from_flat_cfg(flat, bee_model=construction)
    out = run_seeded_bee_colony(dl, OmegaConf.create(eval_dims), cost, R, search_cfg=flat,
                                plan=plan, device=device, silent=True)
    routes_a = _as(out[4])

    assert routes_b.shape == routes_a.shape
    assert torch.equal(routes_b, routes_a), "seeded neural BCO diverged from flat"
