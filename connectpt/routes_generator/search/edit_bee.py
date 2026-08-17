"""Load the trim-capable edit model used by the trim/extend BCO edit bees.

The edit bee (type-5/6/7 in ``bee_colony``) is driven by the trained route-edit
policy -- a ``TrimPathCombiningRouteGenerator`` (extend / trim / halt). This
builder composes that model config and strict-loads a checkpoint, so old weights
keep loading. It is the library home of what used to be
``eval_lib.helpers.build_edit_model``; behaviour (config composition + load) is
identical, but the checkpoint path and adjustment-conditioning feature count are
explicit arguments rather than module globals.
"""
from __future__ import annotations

import torch
from hydra import compose, initialize_config_dir

from ..core.paths import CFG_DIR, EDIT_MODEL_WEIGHTS_PATH
from ..utils import build_model_from_cfg


def build_edit_bee_model(device, weights_path=None, *, load_weights=True,
                         n_adjustment_cond_feats: int = 0):
    """Build the trim-capable edit model for the edit/trim BCO mutations.

    ``weights_path`` defaults to the preserved edit checkpoint
    (:data:`core.paths.EDIT_MODEL_WEIGHTS_PATH`) but may point at any compatible
    trim-model checkpoint. ``load_weights=False`` returns a randomly-initialised
    model of the same architecture (the untrained-policy RL ablation baseline).

    ``n_adjustment_cond_feats`` must match the checkpoint: adjustment-conditioned
    checkpoints carry 1-2 extra global features, so the architecture has to be
    built with the same count or ``load_state_dict`` fails.
    """
    weights_path = weights_path or EDIT_MODEL_WEIGHTS_PATH
    overrides = [
        "model=bestsofar_feb2023_trim",
        "model.route_generator.kwargs.serial_halting=True",
        "++run_name=eval_seeded_edit_model",
        "++experiment.logdir=null",
    ]
    if n_adjustment_cond_feats:
        overrides.append(
            "++model.route_generator.kwargs.n_adjustment_cond_feats="
            f"{int(n_adjustment_cond_feats)}")
    if load_weights:
        overrides.append(f"+model.weights='{weights_path}'")
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        edit_cfg = compose(config_name="training/ppo_50nodes.yaml", overrides=overrides)
    edit_model = build_model_from_cfg(edit_cfg.model, edit_cfg.experiment)
    if load_weights:
        edit_model.load_state_dict(torch.load(weights_path, map_location=device))
    edit_model.to(device).eval()
    return edit_model
