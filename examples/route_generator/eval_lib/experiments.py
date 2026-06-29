"""Experiment cfg-builders for paper_combined.ipynb PART 2.

The mechanical "build a baseline / NBCO config for a city+spec+variant" helpers
were previously defined inline in the notebook's shared-helpers cell. They are
extracted here so the notebook keeps only the experiment *design* (per-city
budgets, variant tables, alpha grids) and binds these builders to its runtime
context via :class:`ExperimentContext`.

Behaviour is identical to the old inline helpers; only the dependency on
notebook globals is made explicit through ``ctx``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .baselines import build_ga_cfg, build_hh_cfg, build_sa_cfg
from .helpers import build_bco_cfg
from .paper import bco_cfg_set, set_cfg_value


@dataclass
class ExperimentContext:
    """Runtime knobs the cfg-builders need, supplied once by the notebook."""

    algo_settings: Callable[[str], dict]
    connectivity_mode: str
    unified_weights: dict
    unified_adj: dict
    early_stop_patience: int | None
    hh_max_repair_iters: int
    our_model_path: object
    bco_n_type1_bees: int
    include_edit_rebuild: bool


def load_experiment_cfg(name):
    """Load a captured config-first experiment YAML (config-first, no builder).

    ``name`` is relative to ``cfg/experiments`` without the .yaml suffix, e.g.
    "nbco_variants/our_nbco_mumford0" or "ekb/our_nbco_ekb". The notebook loads
    these instead of calling the Python config builders; per-run sweep overrides
    (alpha weights, adj target, n_iterations) are still applied on top.
    """
    from omegaconf import OmegaConf
    from .context import CFG_DIR
    return OmegaConf.load(CFG_DIR / "experiments" / f"{name}.yaml")


def sa_cfg(ctx, city, run_name, n_routes, min_route_len, max_route_len):
    s = ctx.algo_settings(city)
    cfg = build_sa_cfg(run_name, n_routes, min_route_len, max_route_len,
                       n_iterations=s["sa"], early_stop_patience=ctx.early_stop_patience,
                       connectivity_mode=ctx.connectivity_mode)
    for key, value in s.get("sa_args", {}).items():
        set_cfg_value(cfg, f"alg_args.{key}", value)
    return cfg


def ga_cfg(ctx, city, run_name, n_routes, min_route_len, max_route_len):
    s = ctx.algo_settings(city)
    return build_ga_cfg(run_name, n_routes, min_route_len, max_route_len,
                        n_iterations=s["ga"], population_size=s["ga_pop"],
                        early_stop_patience=ctx.early_stop_patience,
                        connectivity_mode=ctx.connectivity_mode)


def hh_cfg(ctx, city, run_name, n_routes, min_route_len, max_route_len):
    s = ctx.algo_settings(city)
    return build_hh_cfg(run_name, n_routes, min_route_len, max_route_len,
                        n_iterations=s["hh"],
                        max_repair_iters=s.get("hh_max_repair_iters", ctx.hh_max_repair_iters),
                        early_stop_patience=ctx.early_stop_patience,
                        connectivity_mode=ctx.connectivity_mode)


def variant_type1_bees(ctx, variant, n_bees):
    if variant.get("n_type2_bees") is None and variant.get("n_type1_bees") == ctx.bco_n_type1_bees:
        return max(1, n_bees // 2)
    return variant["n_type1_bees"]


def variant_bco_cfg(ctx, city, spec, variant, *, weights=None, run_name_suffix=""):
    weights = ctx.unified_weights if weights is None else weights
    s = ctx.algo_settings(city)
    n_bees = int(variant.get("n_bees", s["bco_bees"]))
    cfg = build_bco_cfg(
        run_name=f"{city}_{run_name_suffix}{variant['run_name']}",
        n_routes=spec["n_routes"], min_route_len=spec["min_route_len"],
        max_route_len=spec["max_route_len"], use_neural_bees=variant["use_neural_bees"],
        n_bees=n_bees, n_type1_bees=variant_type1_bees(ctx, variant, n_bees),
        n_type2_bees=variant["n_type2_bees"], n_type4_bees=variant["n_type4_bees"],
        n_type5_bees=variant.get("n_type5_bees", 0),
        n_type6_bees=variant.get("n_type6_bees", 0),
        n_type7_bees=variant.get("n_type7_bees", 0),
        connectivity_mode=ctx.connectivity_mode,
        **s.get("bco_args", {}), **weights)
    bco_cfg_set(cfg, n_iterations=s["bco"])
    if any(int(variant.get(k, 0) or 0) > 0 for k in (
            "n_type4_bees", "n_type5_bees", "n_type6_bees", "n_type7_bees")):
        bco_cfg_set(cfg, type4_allow_halt=False, type5_allow_halt=False,
                    type6_allow_halt=False, type7_allow_halt=False)
    _halt_overrides = {
        key: bool(variant[key])
        for key in ("type4_allow_halt", "type5_allow_halt",
                    "type6_allow_halt", "type7_allow_halt")
        if key in variant
    }
    if _halt_overrides:
        bco_cfg_set(cfg, **_halt_overrides)
    return cfg


def our_nbco_versions(include_edit_rebuild):
    """Our NBCO version list (B = edit-model rebuild) is opt-in (very slow)."""
    versions = [(True, "GNN rebuild + trim/extend")]
    if include_edit_rebuild:
        versions.append((False, "edit-model rebuild + trim/extend"))
    return versions


def our_model_cfg(ctx, city, spec, *, adj_target=None, run_name_suffix="",
                  seed=None, use_gnn=True, force_cpu=False):
    s = ctx.algo_settings(city)
    n_bees = int(s["bco_bees"])
    rebuild_bees = max(1, n_bees // 2)         # rebuild bees (type-1)
    trim_extend_bees = n_bees - rebuild_bees   # one-step edit bees (type-5)
    if use_gnn:
        bee_arch, bee_w, tag = None, None, "our_nbco_gnn_rebuild_trimext"
    else:
        bee_arch = "bestsofar_feb2023_trim"    # rebuild driven by OUR edit model
        bee_w = str(ctx.our_model_path)
        tag = "our_nbco_editmodel_rebuild_trimext"
    cfg = build_bco_cfg(
        run_name=f"{city}_{run_name_suffix}{tag}",
        n_routes=spec["n_routes"], min_route_len=spec["min_route_len"],
        max_route_len=spec["max_route_len"], use_neural_bees=True,
        n_bees=n_bees, n_type1_bees=rebuild_bees, n_type2_bees=0,
        n_type4_bees=0, n_type5_bees=trim_extend_bees,
        n_type6_bees=0, n_type7_bees=0,
        bee_model_arch=bee_arch, bee_model_weights=bee_w,
        force_cpu=force_cpu, connectivity_mode=ctx.connectivity_mode,
        **s.get("bco_args", {}), **ctx.unified_weights)
    bco_cfg_set(cfg, n_iterations=s["bco"])
    # Edit/trim/extend bees must propose a real mutation when one is valid;
    # only the main type-1 neural constructor may use halt.
    bco_cfg_set(cfg, type4_allow_halt=False, type5_allow_halt=False,
                type6_allow_halt=False, type7_allow_halt=False)
    if adj_target is not None:
        bco_cfg_set(cfg, **dict(ctx.unified_adj,
                                adjustment_degree_target=float(adj_target)))
    if seed is not None:
        set_cfg_value(cfg, "experiment.seed", int(seed))
    return cfg


def abl_variant(construct, edit, n_bees):
    """2x2 ablation BCO variant (GNN/RPC construct x trim-extend/type2 edit)."""
    if edit in {"trim5+extend5", "trim 5 + extend 5",
                "trim12+extend12", "trim 12 + extend 12"}:
        per_side = 12 if "12" in str(edit) else 5
        extend_n = int(per_side)
        trim_n = int(per_side)
        return {"run_name": f"abl_trim{trim_n}_extend{extend_n}",
                "summary_label": f"trim {trim_n} + extend {extend_n}",
                "use_neural_bees": True,
                "n_bees": trim_n + extend_n,
                "n_type1_bees": 0,
                "n_type2_bees": 0,
                "n_type4_bees": extend_n,
                "n_type5_bees": 0,
                "n_type6_bees": trim_n,
                "n_type7_bees": 0,
                "type4_allow_halt": True,
                "type6_allow_halt": True}
    if construct == "none":   # all bees are the edit type (no rebuild bee)
        return {"run_name": f"abl_only_{edit}".replace("/", ""),
                "summary_label": f"{edit} only",
                "use_neural_bees": False,
                "n_type1_bees": 0,
                "n_type2_bees": n_bees if edit == "type2" else 0,
                "n_type4_bees": 0,
                "n_type5_bees": n_bees if edit == "trim/extend" else 0,
                "n_type6_bees": 0, "n_type7_bees": 0}
    half = max(1, n_bees // 2)
    edit_n = n_bees - half
    return {"run_name": f"abl_{construct}_{edit}".replace("/", ""),
            "summary_label": f"{construct} + {edit}",
            "use_neural_bees": construct == "GNN",
            "n_type1_bees": half if construct == "GNN" else 0,
            "n_type2_bees": edit_n if edit == "type2" else 0,
            "n_type4_bees": 0,
            "n_type5_bees": edit_n if edit == "trim/extend" else 0,
            "n_type6_bees": 0, "n_type7_bees": 0}
