"""Compose an experiment config by name -- the entry point of every run.

``load_experiment("validation")`` composes the declarative run config
under ``cfg/runs/`` --
single runs and batch configs alike. The caller names an experiment and runs it:
no hydra plumbing, no config building in cells.

``build_experiment`` is the procedural layer on top: it composes the parent
config, then injects sweep/data/run parameters (alpha, adj_target, instance,
seed, ...) into the composed cfg in code. Every parameter defaults to ``None``
-> keep the YAML value; a passed value overrides it. This replaces string
``overrides=["sweep.alpha=[0,1]"]`` with typed keyword arguments.
"""
from __future__ import annotations

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from .const import CFG_DIR
from ..objectives.cost import CostParams
from ..objectives.preferences import PreferenceSamplerParams
from ..policy.build import ModelParams, PolicyProfile
from ..common.parameters import CostConstants


def load_experiment(name: str, *, overrides=None):
    """Compose a declarative run config (``cfg/runs/<name>.yaml``)."""
    config_name = name if name.startswith("runs/") else f"runs/{name}"
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides or []))


def cost_params(cost_cfg) -> CostParams:
    """Decode the cost group into library parameters at the run boundary."""

    return CostParams(**OmegaConf.to_container(cost_cfg.kwargs, resolve=True))


def sampler_params(preferences_cfg):
    """Decode the sampler's own preset at the run boundary."""
    values = OmegaConf.to_container(preferences_cfg, resolve=True)
    values['objectives'] = tuple(values['objectives'])
    return PreferenceSamplerParams(**values)


def policy_profile(model_cfg, cost_cfg, common, device, *, checkpoint_path=None):
    """Resolve config groups at the Run.setup boundary."""

    values = OmegaConf.to_container(cost_cfg.kwargs, resolve=True)
    return PolicyProfile(
        ModelParams.from_values(OmegaConf.to_container(model_cfg, resolve=True)),
        CostConstants(common.symmetric_routes, values["mean_stop_time_s"], values["avg_transfer_wait_time_s"]),
        tuple(values["objectives"]),
        checkpoint_path,
        device,
    )


def _runs_root(cfg) -> str:
    """The scratch root for per-run outputs (``paths.runs_dir``)."""
    return str(cfg.paths.runs_dir)


def _set_run_dir(cfg) -> None:
    """Re-point ``paths.output_dir`` at ``<runs_dir>/<run.name>``."""
    cfg.paths.output_dir = f"{_runs_root(cfg)}/{cfg.run.name}"


def _retarget_instance(cfg, name: str) -> None:
    """Scope the run name/output directory to the selected instance."""
    base = str(cfg.run.name)
    seg = name.lower()
    if not base.endswith(f"/{seg}") and not base.endswith(seg):
        cfg.run.name = f"{base}/{seg}"
    _set_run_dir(cfg)


def build_experiment(
    name: str,
    *,
    instance: str | None = None,
    method: str | None = None,
    alpha=None,
    adj_target=None,
    adj_weight=None,
    n_iterations: int | None = None,
    seed: int | None = None,
    cpu: bool | None = None,
    weights_dir: str | None = None,
    runs_dir: str | None = None,
):
    """Compose a declarative experiment and inject parameters procedurally.

    Every keyword defaults to ``None`` -> the value stays as written in the YAML
    (parent config + groups). A passed value is merged into the composed cfg so
    the sweep/data/run block reflects it, without touching any file on disk.

    A **method** selects the ``search/method=<x>`` Hydra group, which carries the
    operators AND the models they need.

    ``alpha``/``adj_target`` set the one search point (the script loops over the grid).

    ``weights_dir`` is the root relative checkpoint paths resolve against and
    ``runs_dir`` the root of the per-run scratch dir -- both default to the
    config's ``paths`` block (``artifacts/model_weights`` / ``artifacts/runs``).
    """
    # Group selections must be compose-time overrides (Hydra picks the defaults
    # group before merge); everything else is injected into the composed cfg.
    group_overrides = []
    if method is not None:
        group_overrides.append(f"search/method={method}")
    if instance is not None:
        group_overrides.append(f"city/instance={instance}")
    cfg = load_experiment(name, overrides=group_overrides)
    with open_dict(cfg):
        # -- output roots (weights read from, scratch written to) --
        if "paths" not in cfg:
            cfg.paths = OmegaConf.create({})
        if weights_dir is not None:
            cfg.paths.weights_dir = str(weights_dir)
        if runs_dir is not None:
            cfg.paths.runs_dir = str(runs_dir)
            _set_run_dir(cfg)
        # -- data / instance --
        if instance is not None:
            _retarget_instance(cfg, instance)
        # -- search point (create the block if the leaf has none) --
        if "sweep" not in cfg:
            cfg.sweep = OmegaConf.create({})
        if alpha is not None:
            cfg.sweep.alpha = alpha
        if adj_target is not None:
            cfg.sweep.adj_target = adj_target
        if adj_weight is not None:
            cfg.sweep.adj_weight = adj_weight
        if n_iterations is not None:
            cfg.sweep.n_iterations = int(n_iterations)

        # -- run --
        if seed is not None:
            cfg.run.seed = int(seed)
        if cpu is not None:
            cfg.run.cpu = bool(cpu)
    return cfg
