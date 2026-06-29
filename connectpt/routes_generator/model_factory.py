"""Construct route-generation models from config, with role checks.

Thin facade over the existing :func:`build_model_from_cfg` so callers depend on
``RouteModelFactory`` rather than reaching into ``utils``. The factory never
wraps the model in a new ``nn.Module`` -- the returned object is exactly the
class a checkpoint was trained on, so ``CheckpointStore.load_model_weights(...,
strict=True)`` keeps working.

The ``*_model`` builders additionally assert the produced class matches the
expected role (construction vs edit), turning a silent wrong-config into a loud
error.
"""
from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from .utils import build_model_from_cfg

CFG_DIR = Path(__file__).resolve().parent / "cfg"

_ROLE_CLASS = {
    "construction": "PathCombiningRouteGenerator",
    "edit": "TrimPathCombiningRouteGenerator",
}


class RouteModelFactory:
    """Build route models from config nodes or by config name."""

    @staticmethod
    def build(model_cfg, exp_cfg):
        """Delegate to the existing builder (no behaviour change)."""
        return build_model_from_cfg(model_cfg, exp_cfg)

    @staticmethod
    def build_from_cfg(cfg):
        """Build from a composed cfg that has ``.model`` and ``.experiment``."""
        return build_model_from_cfg(cfg.model, cfg.experiment)

    @staticmethod
    def _require_role(model, role: str):
        expected = _ROLE_CLASS[role]
        actual = type(model).__name__
        if actual != expected:
            raise TypeError(
                f"expected a {role} model ({expected}), got {actual}; check the model config"
            )
        return model

    @staticmethod
    def build_construction_model(model_cfg, exp_cfg):
        return RouteModelFactory._require_role(
            build_model_from_cfg(model_cfg, exp_cfg), "construction"
        )

    @staticmethod
    def build_edit_model(model_cfg, exp_cfg):
        return RouteModelFactory._require_role(
            build_model_from_cfg(model_cfg, exp_cfg), "edit"
        )

    # -- compose-by-name helpers (used by the search/eval runs) --------------

    @staticmethod
    def compose_and_build(model_name: str, *, cfg_dir: Path | str = CFG_DIR):
        """Build a model from its standalone ``model_build/<model_name>`` config.

        Config-first: no ``model=...`` Hydra override. ``model_build/<name>.yaml``
        pulls in the experiment group + the model group, so the composed cfg has
        both ``.model`` and ``.experiment``.
        """
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cfg = compose(config_name=f"model_build/{model_name}")
        return build_model_from_cfg(cfg.model, cfg.experiment)

    @staticmethod
    def build_construction_model_by_name(model_name: str, **kw):
        return RouteModelFactory._require_role(
            RouteModelFactory.compose_and_build(model_name, **kw), "construction"
        )

    @staticmethod
    def build_edit_model_by_name(model_name: str, **kw):
        return RouteModelFactory._require_role(
            RouteModelFactory.compose_and_build(model_name, **kw), "edit"
        )
