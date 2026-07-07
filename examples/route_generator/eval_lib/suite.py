"""Experiment-suite config loader.

Thin wrapper over the library ``core.loaders.load_suite`` (one config-compose
implementation) -- composes ``cfg/experiments/<name>.yaml``. The notebook loads
one of these (``suite`` for the full paper run, ``suite_smoke`` for an all-on
1-iteration TEMP dry-run) and reads everything from it.
"""
from __future__ import annotations

from connectpt.routes_generator.core.loaders import load_suite as _load_suite


def load_suite_config(name="suite_smoke", *, overrides=None, cfg_dir=None):
    """Compose and return the experiment-suite config ``cfg/experiments/<name>``.

    ``cfg_dir`` is accepted for backward compatibility; the library loader uses
    the canonical ``core.paths.CFG_DIR`` (the same directory).
    """
    return _load_suite(name, overrides=overrides)
