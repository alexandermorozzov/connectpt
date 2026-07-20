"""Compose an experiment / suite config by name -- the notebook's entry point.

``load_experiment("e1/mumford0/our_nbco")`` composes the declarative run config
under ``cfg/experiments/`` (the ``experiments/`` prefix is added if absent);
``load_suite`` does the same for a batch config. The notebook names an
experiment and runs it -- no hydra plumbing, no config building in cells.

``build_experiment`` is the procedural layer on top: it composes the parent
config, then injects sweep/data/run parameters (alpha, adj_target, city, route
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

# Smoke = a fast throwaway dry-run: the SAME experiment at a tiny iteration
# budget (the only thing every ``*_smoke.yaml`` twin used to change). Driven by
# ``build_experiment(smoke=True)`` now -- no per-experiment smoke files.
SMOKE_ITERS = 2


def _compose_experiment(name: str, overrides):
    config_name = name if name.startswith("experiments/") else f"experiments/{name}"
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides or []))


def load_experiment(name: str, *, overrides=None):
    """Compose a declarative run config (``cfg/experiments/<name>.yaml``)."""
    return _compose_experiment(name, overrides)


def load_suite(name: str, *, overrides=None):
    """Compose a batch/suite config (``cfg/experiments/<name>.yaml``)."""
    return _compose_experiment(name, overrides)


def _as_list(value):
    """A scalar sweep axis is normalised to a one-element grid."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _retarget_city(cfg, city: str) -> None:
    """Point ``run.name``/``paths.output_dir`` at the chosen city.

    A collapsed leaf (one per method) carries a city-less ``run.name`` template;
    substituting a city nests the outputs under ``<name>/<city>`` so per-city
    runs never clobber each other. Idempotent when the name already ends in the
    city segment.

    Benchmark configs inherit the Mandl data group from ``bee_colony_base``.
    When retargeting to another benchmark, refresh the route-count / route-len
    bounds from the eval registry so the inherited Mandl bounds do not leak.
    """
    cfg.data.city = city
    if cfg.data.get("source") == "benchmark":
        from ..data.loaders import benchmark_spec
        spec = benchmark_spec(city)
        cfg.data.n_routes = int(spec["n_routes"])
        cfg.data.min_route_len = int(spec["min_route_len"])
        cfg.data.max_route_len = int(spec["max_route_len"])
    base = str(cfg.run.name)
    seg = city.lower()
    if not base.endswith(f"/{seg}") and not base.endswith(seg):
        cfg.run.name = f"{base}/{seg}"
    cfg.paths.output_dir = f"artifacts/runs/{cfg.run.name}"


def build_experiment(
    name: str,
    *,
    city: str | None = None,
    bee_sets: str | None = None,
    models: str | None = None,
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
    smoke: bool = False,
    overrides=None,
):
    """Compose a declarative experiment and inject parameters procedurally.

    Every keyword defaults to ``None`` -> the value stays as written in the YAML
    (parent config + groups). A passed value is merged into the composed cfg so
    the sweep/data/run block reflects it, without touching any file on disk.

    A **method** is picked in code exactly like a sweep axis: ``bee_sets`` and
    ``models`` select the Hydra groups (``search/bee_sets=<x>`` /
    ``search/models=<x>``) that used to be a hand-written per-method leaf file;
    ``n_bees`` sizes the colony to the chosen set and ``label`` names the method
    (it becomes the ``method`` column of the sweep table + disambiguates the run
    output dir). This lets one artifact YAML carry a ``methods:`` list instead of
    N leaf files -- see :class:`ExperimentBatch`.

    ``alpha``/``adj_target`` accept a scalar or a list (the sweep grid axis);
    ``route_len`` is ``(min_route_len, max_route_len)``. ``smoke=True`` caps the
    iteration budget to :data:`SMOKE_ITERS` for a fast dry-run (an explicit
    ``n_iterations`` still wins).
    """
    # Group selections must be compose-time overrides (Hydra picks the defaults
    # group before merge); everything else is injected into the composed cfg.
    group_overrides = list(overrides or [])
    if bee_sets is not None:
        group_overrides.append(f"search/bee_sets={bee_sets}")
    if models is not None:
        group_overrides.append(f"search/models={models}")
    cfg = _compose_experiment(name, group_overrides)
    with open_dict(cfg):
        # -- method (group choice already applied at compose; size + name here) --
        if n_bees is not None:
            cfg.search.n_bees = int(n_bees)
        if label is not None:
            cfg.run.label = label
        # -- data / instance --
        if city is not None:
            _retarget_city(cfg, city)
        if label is not None:
            seg = _slug(label)
            if seg and not str(cfg.run.name).endswith(seg):
                cfg.run.name = f"{cfg.run.name}/{seg}"
                cfg.paths.output_dir = f"artifacts/runs/{cfg.run.name}"
        if n_routes is not None:
            cfg.data.n_routes = int(n_routes)
        if route_len is not None:
            cfg.data.min_route_len = int(route_len[0])
            cfg.data.max_route_len = int(route_len[1])

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
        elif smoke:
            cfg.sweep.n_iterations = SMOKE_ITERS

        # -- run --
        if seed is not None:
            cfg.run.seed = int(seed)
        if cpu is not None:
            cfg.run.cpu = bool(cpu)
    return cfg
