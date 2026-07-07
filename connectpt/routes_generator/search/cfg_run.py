"""Run a seeded bee-colony search from a composed BCO config -- library-owned.

``run_bco_from_cfg`` is the library home of the notebook-support ``run_bco``
orchestration: from a composed BCO cfg + explicit graph tensors + an initial
route set it builds the tensor dataloader, the cost + construction (rebuild) bee
model (via ``process_standard_experiment_cfg``) and, when edit bees are present,
the trim/extend edit model, then runs the seeded bee-colony search and annotates
the metrics with the per-component cost breakdown.

The edit checkpoint path and its adjustment-conditioning feature count are
explicit arguments (not module globals), so callers stay in control of which
edit model the edit bees use. Behaviour is identical to the old eval_lib
``run_bco`` body -- the seeded golden outputs are reproduced bit-for-bit.
"""
from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader

from ..citygraph_dataset import get_dataset_from_config
from ..evaluation.cost_breakdown import add_cost_breakdown_to_metrics
from ..torch_utils import get_batch_tensor_from_routes
from ..utils import process_standard_experiment_cfg
from .compat import plan_from_flat_cfg
from .edit_bee import build_edit_bee_model
from .seeded_search import run_seeded_bee_colony


def _as_route_tensor(routes):
    if isinstance(routes, torch.Tensor):
        return routes.detach().cpu()
    return get_batch_tensor_from_routes(routes).detach().cpu()


def run_bco_from_cfg(cfg, init_routes, tensors, *, mutation_counts_out=None,
                     run_name_scope="", cost_history_out=None,
                     iteration_callback=None, edit_weights_path=None,
                     edit_n_adjustment_cond_feats: int = 0):
    """Build cost/models/data from ``cfg`` + ``tensors`` and run seeded BCO.

    Returns ``(run_name, metrics, unserved_demand, routes_tensor,
    mutation_counts_out)``. When ``cost_history_out`` is a dict, the first
    sample's cost history is stored under its ``"history"`` key.
    """
    dataloader = DataLoader(
        get_dataset_from_config(cfg.eval.dataset, tensors=tensors), batch_size=1)

    use_neural_bees = cfg.get("neural_bees", False)
    prefix = f"{run_name_scope}" + ("neural_bco_" if use_neural_bees else "bco_")
    device, run_name, _, cost_obj, bee_model = process_standard_experiment_cfg(
        cfg, run_name_prefix=prefix, weights_required=use_neural_bees)

    force_linking_unlinked = cfg.get("force_linking_unlinked", False)
    if not use_neural_bees:
        bee_model = None
    elif bee_model is not None:
        bee_model.force_linking_unlinked = force_linking_unlinked
        bee_model.eval()

    # The plan owns the bee taxonomy: build it from the cfg counts and let it
    # decide whether an edit checkpoint is needed (edit/trim/compound bees),
    # instead of peeking at legacy edit-bee counts. If needed, load the edit
    # model and
    # rebuild the plan with it attached.
    plan = plan_from_flat_cfg(cfg, bee_model=bee_model)
    if plan.needs_edit:
        edit_model = build_edit_bee_model(
            device, edit_weights_path,
            n_adjustment_cond_feats=edit_n_adjustment_cond_feats)
        plan = plan_from_flat_cfg(
            cfg, bee_model=bee_model, edit_model=edit_model)
    mutation_counts_out = {} if mutation_counts_out is None else mutation_counts_out

    output = run_seeded_bee_colony(
        dataloader, cfg.eval, cost_obj, init_routes,
        search_cfg=cfg, plan=plan,
        mutation_counts_out=mutation_counts_out, device=device, silent=False,
        return_histories=cost_history_out is not None,
        iteration_callback=iteration_callback)

    if cost_history_out is not None:
        _, _, unserved_demand, metrics, routes, cost_histories = output
        if cost_histories:
            _h = cost_histories[0]
            cost_history_out["history"] = (
                _h.detach().cpu().clone() if hasattr(_h, "detach") else _h)
    else:
        _, _, unserved_demand, metrics, routes = output

    routes_tensor = _as_route_tensor(routes)
    metrics = add_cost_breakdown_to_metrics(
        metrics, dataloader, cfg.eval, cost_obj, routes_tensor, device)
    return run_name, metrics, unserved_demand, routes_tensor, mutation_counts_out
