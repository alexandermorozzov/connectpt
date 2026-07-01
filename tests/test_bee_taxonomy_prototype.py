"""U2 prototype: the declarative bee schema can express the paper's Our-NBCO mix.

The rich schema (search/bee_sets/*) is an adapter back onto bee_colony's numbered
n_type* counts. Before this prototype it could NOT express the paper's neural
full-route rebuild (type-1 + construction model): a neural construction bee was
always classified as type-4 (one extend step). Adding the ``neural_rebuild``
operator closes that gap. This test proves the ``our_nbco`` bee_set produces the
SAME n_type* plan counts as the flat nbco_variants/our_nbco config
(n_type1=5, n_type5=5), which is what drives bee_colony's behaviour.
"""
from pathlib import Path

from omegaconf import OmegaConf

from connectpt.routes_generator.search.bee_plan import BeeColonyPlan
from connectpt.routes_generator.search.bee_specs import parse_bee_specs
from connectpt.routes_generator.search.search_policies import (
    ConstructionSearchPolicy, EditSearchPolicy)

CFG = Path(__file__).resolve().parents[1] / "connectpt" / "routes_generator" / "cfg"
BEE_SET = CFG / "search" / "bee_sets" / "our_nbco.yaml"
FLAT = CFG / "experiments" / "nbco_variants" / "our_nbco_mumford0.yaml"


def _policies():
    # role-only adapters (validation uses ROLE_CAPABILITIES, not the model)
    return {
        "construction": ConstructionSearchPolicy(model=None, name="construction"),
        "edit": EditSearchPolicy(model=None, name="edit"),
    }


def test_our_nbco_bee_set_matches_flat_counts():
    specs = parse_bee_specs(OmegaConf.load(BEE_SET).bees)
    plan = BeeColonyPlan.from_specs(specs, _policies())

    # The flat config's per-type counts are the ground truth.
    flat = OmegaConf.load(FLAT)
    expected = {f"n_type{i}": int(flat.get(f"n_type{i}_bees", 0)) for i in range(1, 8)}
    assert expected["n_type1"] == 5 and expected["n_type5"] == 5  # sanity on the fixture

    assert plan.counts == expected, (plan.counts, expected)
    assert plan.needs_construction is True  # type-1 rebuild drives the construction model
    assert plan.needs_edit is True          # type-5 edit bees drive the edit model


def test_rpc_path_mix_is_type3():
    """A random path-combiner rebuild bee maps to type-3 (path_mix_rebuild).
    bee_colony derives the type-3 count as the remainder, so it is not emitted
    explicitly -- but the plan classifies it correctly and needs no models."""
    rpc = CFG / "search" / "bee_sets" / "rpc_trim_extend.yaml"
    plan = BeeColonyPlan.from_specs(parse_bee_specs(OmegaConf.load(rpc).bees), _policies())
    assert plan.counts["n_type3"] == 5   # rpc rebuild
    assert plan.counts["n_type5"] == 5   # edit trim/extend
    assert plan.counts["n_type1"] == 0 and plan.counts["n_type4"] == 0
    assert plan.needs_construction is False  # type-3 is heuristic, no model
    assert plan.needs_edit is True


def test_neural_rebuild_is_type1_not_type4():
    """A neural construction rebuild bee must map to type-1 (full rebuild), not
    type-4 (single construction-extend step)."""
    specs = parse_bee_specs([
        {"name": "r", "count": 3, "operator": "neural_rebuild", "policy": "construction"},
    ])
    plan = BeeColonyPlan.from_specs(specs, _policies())
    assert plan.counts["n_type1"] == 3
    assert plan.counts["n_type4"] == 0
    assert plan.needs_construction is True


# --- U1: seeded end-to-end parity (declarative our_nbco == flat our_nbco) -----
import random as _random

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from connectpt.routes_generator.core.checkpoints import CheckpointStore
from connectpt.routes_generator.core.paths import (
    CONSTRUCTION_MODEL_WEIGHTS_PATH, EDIT_MODEL_WEIGHTS_DIR)
from connectpt.routes_generator.core.runtime import seed_everything
from connectpt.routes_generator.model_factory import RouteModelFactory
from connectpt.routes_generator.objectives import CostFactory
from connectpt.routes_generator.citygraph_dataset import get_dataset_from_config
from connectpt.routes_generator.search.edit_bee import build_edit_bee_model
from connectpt.routes_generator.search.plan_kwargs import plan_to_search_cfg
from connectpt.routes_generator.search.seeded_search import run_seeded_bee_colony
from connectpt.routes_generator.torch_utils import get_batch_tensor_from_routes

_EDIT_CKPT = EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_rttconn_adj_w10_t02_finetune100.pt"
_CITY, _N_ITERS = "Mumford0", 3


def _as_tensor(routes):
    if isinstance(routes, torch.Tensor):
        return routes.detach().cpu()
    return get_batch_tensor_from_routes(routes).detach().cpu()


def _init_routes(spec, tensors):
    rng = _random.Random(0)
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


def test_declarative_our_nbco_seeded_matches_flat_run():
    if not (CONSTRUCTION_MODEL_WEIGHTS_PATH.exists() and _EDIT_CKPT.exists()):
        pytest.skip("construction/edit weights not present")

    import sys
    route_examples = str(Path(__file__).resolve().parents[1] / "examples" / "route_generator")
    if route_examples not in sys.path:
        sys.path.insert(0, route_examples)
    from eval_lib.baselines import load_benchmark_tensors, BENCHMARK_SPECS
    from eval_lib.experiments import load_experiment_cfg
    from eval_lib.paper import bco_cfg_set, UNIFIED_ADJ, set_cfg_value

    device = torch.device("cpu")
    spec = next(s for s in BENCHMARK_SPECS if s["city"] == _CITY)
    tensors = load_benchmark_tensors(_CITY)
    R = _init_routes(spec, tensors)

    # shared cost / models / data -- so the ONLY difference is the search cfg
    cost = CostFactory.build_unified("rtt_wmc_no_demand", for_training=False); cost.to(device)
    construction = RouteModelFactory.build_construction_model_by_name("construction")
    CheckpointStore.load_model_weights(construction, CONSTRUCTION_MODEL_WEIGHTS_PATH,
                                       strict=True, map_location=device)
    construction.to(device).eval()
    edit = build_edit_bee_model(device, _EDIT_CKPT, n_adjustment_cond_feats=0)
    dataloader = DataLoader(get_dataset_from_config(OmegaConf.create({"type": "tensor"}),
                                                    tensors=tensors), batch_size=1)
    eval_cfg = OmegaConf.create({"n_routes": spec["n_routes"],
                                 "min_route_len": spec["min_route_len"],
                                 "max_route_len": spec["max_route_len"]})

    def run(search_cfg):
        seed_everything(0)
        out = run_seeded_bee_colony(dataloader, eval_cfg, cost, R, search_cfg=search_cfg,
                                    bee_model=construction, edit_model=edit,
                                    device=device, silent=True)
        return _as_tensor(out[4])

    # flat reference == golden our_Mumford0 config
    flat = load_experiment_cfg("nbco_variants/our_nbco_mumford0")
    for k in ("n_routes", "min_route_len", "max_route_len"):
        set_cfg_value(flat, f"eval.{k}", int(spec[k]))
    bco_cfg_set(flat, n_iterations=_N_ITERS, **UNIFIED_ADJ)
    set_cfg_value(flat, "experiment.cost_function.kwargs.use_weighted_connectivity", True)

    # declarative path: our_nbco bee_set -> plan -> flat search cfg
    plan = BeeColonyPlan.from_specs(parse_bee_specs(OmegaConf.load(BEE_SET).bees), _policies())
    declarative = plan_to_search_cfg(plan, n_bees=10, n_iterations=_N_ITERS)

    routes_flat = run(flat)
    routes_decl = run(declarative)
    assert routes_flat.shape == routes_decl.shape
    assert torch.equal(routes_flat, routes_decl), "declarative our_nbco diverged from flat"
