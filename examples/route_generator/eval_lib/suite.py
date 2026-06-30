"""Experiment-suite config loader.

``load_suite_config(name)`` composes ``cfg/experiments/<name>.yaml`` into a plain
OmegaConf config that selects which experiments run, on which cities, with which
shared parameters. The notebook loads one of these (``suite`` for the full paper
run, ``suite_smoke`` for an all-on 1-iteration TEMP dry-run) and reads everything
from it -- mirroring load_train_config so PART 1 and PART 2 share one pattern.
"""
from __future__ import annotations


def load_suite_config(name="suite_smoke", *, overrides=None, cfg_dir=None):
    """Compose and return the experiment-suite config ``cfg/experiments/<name>``."""
    from hydra import compose, initialize_config_dir

    from .context import CFG_DIR

    cfg_dir = cfg_dir or CFG_DIR
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        return compose(config_name=f"experiments/{name}", overrides=list(overrides or []))
