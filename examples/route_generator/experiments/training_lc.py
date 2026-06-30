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
