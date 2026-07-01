"""Config-first experiment loader for paper_combined.ipynb PART 2.

The mechanical "build a baseline / NBCO config in Python" helpers
(``ExperimentContext`` + ``sa_cfg`` / ``variant_bco_cfg`` / ``our_model_cfg`` ...)
have been retired: every experiment now loads a captured config-first YAML from
``cfg/experiments`` and runs it through the library seeded-BCO path. Only the
loader remains here.
"""
from __future__ import annotations


def load_experiment_cfg(name):
    """Load a captured config-first experiment YAML (no Python builder).

    ``name`` is relative to ``cfg/experiments`` without the .yaml suffix, e.g.
    "nbco_variants/our_nbco_mumford0" or "ekb/our_nbco_ekb". Per-run sweep
    overrides (alpha weights, adj target, n_iterations) are applied on top by the
    caller.
    """
    from omegaconf import OmegaConf

    from .context import CFG_DIR
    return OmegaConf.load(CFG_DIR / "experiments" / f"{name}.yaml")
