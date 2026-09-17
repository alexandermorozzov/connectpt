"""Compose an experiment config by name -- the entry point of every run.

``load_experiment("validation")`` composes the declarative run config
under ``cfg/runs/`` --
single runs and batch configs alike. The caller names an experiment and runs it:
no hydra plumbing, no config building in cells.

``build_experiment`` is the procedural layer on top: it composes the parent
config, then injects sweep/data/run parameters (alpha, adj_target, instance, route
bounds, ...) into the composed cfg in code. Every parameter defaults to ``None``
-> keep the YAML value; a passed value overrides it. This replaces string
``overrides=["sweep.alpha=[0,1]"]`` with typed keyword arguments.
"""
from __future__ import annotations

import re
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from .paths import CFG_DIR


def _slug(text: str) -> str:
    """Filesystem-safe slug for a method label (disambiguates run outputs)."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _compose_experiment(name: str, overrides):
    config_name = name if name.startswith("runs/") else f"runs/{name}"
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides or []))


def load_experiment(name: str, *, overrides=None):
    """Compose a declarative run config (``cfg/runs/<name>.yaml``)."""
    return _compose_experiment(name, overrides)


def _as_list(value):
    """A scalar sweep axis is normalised to a one-element grid."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _runs_root(cfg) -> str:
    """The scratch root for per-run outputs (``paths.runs_dir``)."""
    return str((cfg.get("paths") or {}).get("runs_dir") or "artifacts/runs")


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
    n_bees: int | None = None,
    label: str | None = None,
    alpha=None,
    adj_target=None,
    adj_weight=None,
    n_iterations: int | None = None,
    route_len: tuple[int, int] | Sequence[int] | None = None,
    n_routes: int | None = None,
    seed: int | None = None,
    cpu: bool | None = None,
    weights_dir: str | None = None,
    runs_dir: str | None = None,
    overrides=None,
):
    """Compose a declarative experiment and inject parameters procedurally.

    Every keyword defaults to ``None`` -> the value stays as written in the YAML
    (parent config + groups). A passed value is merged into the composed cfg so
    the sweep/data/run block reflects it, without touching any file on disk.

    A **method** selects the ``method=<x>`` Hydra group, which carries the
    operators AND the models they need; ``label`` names it in the results table.

    ``alpha``/``adj_target`` accept a scalar or a list (the sweep grid axis);
    ``route_len`` is ``(min_route_len, max_route_len)``.

    ``weights_dir`` is the root relative checkpoint paths resolve against and
    ``runs_dir`` the root of the per-run scratch dir -- both default to the
    config's ``paths`` block (``artifacts/model_weights`` / ``artifacts/runs``).
    """
    # Group selections must be compose-time overrides (Hydra picks the defaults
    # group before merge); everything else is injected into the composed cfg.
    group_overrides = list(overrides or [])
    if method is not None:
        group_overrides.append(f"method={method}")
    if instance is not None:
        group_overrides.append(f"instance={instance}")
    cfg = _compose_experiment(name, group_overrides)
    with open_dict(cfg):
        # -- output roots (weights read from, scratch written to) --
        if cfg.get("paths") is None:
            cfg.paths = OmegaConf.create({})
        if weights_dir is not None:
            cfg.paths.weights_dir = str(weights_dir)
        if runs_dir is not None:
            cfg.paths.runs_dir = str(runs_dir)
            _set_run_dir(cfg)
        # -- method (group choice already applied at compose; size + name here) --
        if n_bees is not None:
            cfg.search.n_bees = int(n_bees)
        if label is not None:
            cfg.run.label = label
        # -- data / instance --
        if instance is not None:
            _retarget_instance(cfg, instance)
        if label is not None:
            seg = _slug(label)
            if seg and not str(cfg.run.name).endswith(seg):
                cfg.run.name = f"{cfg.run.name}/{seg}"
                _set_run_dir(cfg)
        if n_routes is not None:
            cfg.instance.n_routes = int(n_routes)
        if route_len is not None:
            cfg.instance.min_route_len = int(route_len[0])
            cfg.instance.max_route_len = int(route_len[1])

        # -- sweep grid (create the block if the leaf has none) --
        if cfg.get("sweep") is None:
            cfg.sweep = OmegaConf.create({})
        if alpha is not None:
            cfg.sweep.alpha = _as_list(alpha)
        if adj_target is not None:
            cfg.sweep.adj_target = (_as_list(adj_target)
                                    if isinstance(adj_target, (list, tuple))
                                    else adj_target)
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
