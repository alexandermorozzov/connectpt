"""Config-first experiment loader + THE single composition point.

Every experiment loads a captured YAML from ``cfg/experiments``; the ONLY
values applied on top in Python are (a) data-derived route bounds, (b) run
identity (run_name / seed / cpu / csv), (c) the sweep point (alpha,
adjustment target, iteration budget) whose grid lives in the experiment spec
YAML, and (d) the unified adjustment kwargs sourced from the objective YAML.
All of them go through :func:`compose_experiment_cfg` -- there is no ad-hoc
``set_cfg_value`` scattering at call sites (experiment design stays in YAML:
one experiment = one YAML).
"""
from __future__ import annotations

from omegaconf import OmegaConf


def load_experiment_cfg(name):
    """Load a captured config-first experiment YAML (no Python builder).

    ``name`` is relative to ``cfg/experiments`` without the .yaml suffix, e.g.
    "nbco_variants/our_nbco_mumford0" or "ekb/our_nbco_ekb".
    """
    from ..core.paths import CFG_DIR
    return OmegaConf.load(CFG_DIR / "experiments" / f"{name}.yaml")


def set_cfg_value(cfg, dotted_key, value):
    """Set a nested OmegaConf value even when the composed cfg is structured.

    Composition-internal helper (and test plumbing); experiment code calls
    :func:`compose_experiment_cfg` instead of scattering this.
    """
    target = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        OmegaConf.set_struct(target, False)
        target = target[part]
    OmegaConf.set_struct(target, False)
    target[parts[-1]] = value
    return cfg


def bco_cfg_set(cfg, **kv):
    """Set top-level fields BCO reads from cfg (n_iterations, adjustment_degree_*)."""
    OmegaConf.set_struct(cfg, False)
    for k, v in kv.items():
        cfg[k] = v
    return cfg


def unify_weights(cfg):
    """Set a cfg's cost weights + connectivity mode to the unified objective."""
    from connectpt.routes_generator.objectives import load_unified_objective
    obj = load_unified_objective()
    for key, value in obj.weights.items():
        set_cfg_value(cfg, f"experiment.cost_function.kwargs.{key}", value)
    set_cfg_value(cfg, "experiment.cost_function.kwargs.connectivity_mode",
                  obj.connectivity_mode)
    return cfg


def compose_experiment_cfg(base, *, bounds=None, run_name=None, seed=None,
                           cpu=None, csv=None, seq_bees=None, alpha=None,
                           n_iterations=None, adj=None,
                           weighted_connectivity=None):
    """Apply the allowed runtime values onto a loaded experiment preset.

    ``base`` is a preset name (loaded via :func:`load_experiment_cfg`) or an
    already-loaded cfg (mutated in place). Parameters:

    * ``bounds`` -- data-derived route bounds dict (``n_routes`` /
      ``min_route_len`` / ``max_route_len``; extra keys ignored);
    * ``run_name`` / ``seed`` / ``cpu`` / ``csv`` -- run identity;
    * ``seq_bees`` -- ``process_neural_bees_sequentially`` (memory knob);
    * ``alpha`` -- the RTT weight of the sweep point
      (``route_time_weight=alpha``, ``median_connectivity_weight=1-alpha``);
    * ``n_iterations`` -- the search budget of the sweep point;
    * ``adj`` -- adjustment kwargs dict (``adjustment_degree_*``), normally
      ``dict(eval_lib.paper.UNIFIED_ADJ, adjustment_degree_target=...)`` so
      the values originate in the objective YAML;
    * ``weighted_connectivity`` -- ``use_weighted_connectivity`` toggle.
    """
    cfg = load_experiment_cfg(base) if isinstance(base, str) else base
    if bounds is not None:
        for key in ("n_routes", "min_route_len", "max_route_len"):
            if key in bounds:
                set_cfg_value(cfg, f"eval.{key}", int(bounds[key]))
    if run_name is not None:
        set_cfg_value(cfg, "run_name", str(run_name))
    if seed is not None:
        set_cfg_value(cfg, "experiment.seed", int(seed))
    if cpu is not None:
        set_cfg_value(cfg, "experiment.cpu", bool(cpu))
    if csv is not None:
        set_cfg_value(cfg, "eval.csv", bool(csv))
    if seq_bees is not None:
        set_cfg_value(cfg, "process_neural_bees_sequentially", bool(seq_bees))
    if alpha is not None:
        set_cfg_value(cfg, "experiment.cost_function.kwargs.route_time_weight",
                      float(alpha))
        set_cfg_value(cfg,
                      "experiment.cost_function.kwargs.median_connectivity_weight",
                      float(1.0 - float(alpha)))
    if n_iterations is not None:
        bco_cfg_set(cfg, n_iterations=int(n_iterations))
    if adj is not None:
        bco_cfg_set(cfg, **dict(adj))
    if weighted_connectivity is not None:
        set_cfg_value(cfg,
                      "experiment.cost_function.kwargs.use_weighted_connectivity",
                      bool(weighted_connectivity))
    return cfg


def scoring_cfg(city, spec, *, run_name=None, cpu=None, csv=None, alpha=None,
                adj=None):
    """1-iteration eval cfg scoring a FIXED route set under the unified objective.

    Shared by the EKB / MACSA scoring paths (was ``paper.eval_routes_cfg`` +
    per-caller ``set_cfg_value`` clusters). ``use_weighted_connectivity`` is
    always on -- WMC is the paper's connectivity term.
    """
    from ..baselines import build_sa_cfg
    from ..objectives import load_unified_objective
    obj = load_unified_objective()
    cfg = unify_weights(build_sa_cfg(
        f"{city}_eval", spec["n_routes"], spec["min_route_len"],
        spec["max_route_len"], n_iterations=1,
        connectivity_mode=obj.connectivity_mode))
    return compose_experiment_cfg(cfg, run_name=run_name, cpu=cpu, csv=csv,
                                  alpha=alpha, adj=adj,
                                  weighted_connectivity=True)
