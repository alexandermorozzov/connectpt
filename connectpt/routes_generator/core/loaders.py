"""Compose an experiment / suite config by name -- the notebook's entry point.

``load_experiment("e1/mumford0/our_nbco")`` composes the declarative run config
under ``cfg/experiments/`` (the ``experiments/`` prefix is added if absent);
``load_suite`` does the same for a batch config. The notebook names an
experiment and runs it -- no hydra plumbing, no config building in cells.
"""
from __future__ import annotations

from hydra import compose, initialize_config_dir

from .paths import CFG_DIR


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
