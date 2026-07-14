"""BeeColonySearchRun runs SEEDED from a declarative config (C-native).

Improves an EXISTING network passed to ``run(init_routes=, tensors=, eval_dims=)``
-- the paper contract -- for the three shipped seeded configs (our_nbco / classic
/ neural on Mumford0). Asserts the run produces a valid route tensor of the right
shape. (The historical flat-vs-C bit-for-bit golden-parity gate was retired with
the compat path in M010; the C path is now canonical and covered here + by
test_search_execution / test_declarative_experiments.)
"""
import random
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]
LIB_CFG = REPO / "connectpt" / "routes_generator" / "cfg"
ROUTE_EX = REPO / "examples" / "route_generator"
if str(ROUTE_EX) not in sys.path:
    sys.path.insert(0, str(ROUTE_EX))

from connectpt.routes_generator.core.paths import (
    CONSTRUCTION_MODEL_WEIGHTS_PATH, EDIT_MODEL_WEIGHTS_DIR)
from connectpt.routes_generator.data.loaders import (
    benchmark_spec, load_benchmark_tensors)
from connectpt.routes_generator.search.runs import BeeColonySearchRun
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


def _run_seeded(config_name):
    spec = benchmark_spec("Mumford0")
    tensors = load_benchmark_tensors("Mumford0")
    R = _init_routes(spec, tensors)
    eval_dims = {"n_routes": spec["n_routes"], "min_route_len": spec["min_route_len"],
                 "max_route_len": spec["max_route_len"]}
    with initialize_config_dir(config_dir=str(LIB_CFG), version_base=None):
        cfg = compose(config_name=config_name, overrides=[f"search.n_iterations={N_ITERS}"])
    art = BeeColonySearchRun(cfg).run(init_routes=R, tensors=tensors, eval_dims=eval_dims)
    routes = _as(art.result["routes"])
    assert routes.shape[-2] == spec["n_routes"]
    assert routes.dtype == torch.long
    assert (routes >= -1).all()
    return routes


@pytest.mark.skipif(not (CONSTRUCTION_MODEL_WEIGHTS_PATH.exists() and EDIT_CKPT.exists()),
                    reason="construction/edit weights not present")
def test_seeded_our_nbco_runs():
    _run_seeded("experiments/seeded/our_nbco_mumford0")


def test_seeded_classic_bco_runs():
    # heuristic classic BCO -- no neural weights needed.
    _run_seeded("experiments/seeded/classic_bco_mumford0")


@pytest.mark.skipif(not CONSTRUCTION_MODEL_WEIGHTS_PATH.exists(),
                    reason="construction weights not present")
def test_seeded_neural_bco_runs():
    _run_seeded("experiments/seeded/neural_bco_mumford0")
