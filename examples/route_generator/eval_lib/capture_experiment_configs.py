"""Capture the BCO/NBCO experiment configs the notebook builds in Python and
serialize them to config-first YAML files.

This implements the "mock the config-building code, read the values it sets,
write the equivalent YAML" approach: it calls the REAL eval_lib.experiments
builders (which compose bco_mumford.yaml + apply the per-experiment kwargs --
no model/data needed) for every bee mix used by the EKB / E1 / E2 / M0 / MACSA
experiments, then dumps each resulting config to
``cfg/experiments/<experiment>/<name>.yaml``.

Run:  python -m eval_lib.capture_experiment_configs
The notebook then loads these YAMLs instead of building configs in Python; a
round-trip check proves the YAML reloads to exactly the captured config.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from .context import CFG_DIR
from .params import (bco_config, CONNECTIVITY_MODE, UNIFIED_COST_WEIGHTS,
                     ADJ_WEIGHT, ADJ_TARGET, ADJ_GAP, ADJ_MODE)
from .paper import UNIFIED_ADJ
from . import experiments as ex
from .helpers import build_bco_cfg
from .baselines import BENCHMARK_SPECS

OUT_DIR = CFG_DIR / "experiments"

# Per-city benchmark route specs (Mandl / Mumford0-3).
SPECS = {s["city"]: s for s in BENCHMARK_SPECS}


# --- minimal algo_settings (the bco-relevant fields the builders read) -------
# The notebook's full algo_settings tunes SA/HH/GA schedules too; for the BCO
# configs only bco_bees + bco_args (the worse-accept schedule, Mandl-tuned)
# matter, and n_iterations is overridden per experiment below.
_MANDL_BCO_ARGS = dict(
    worse_accept_temperature=0.02, worse_accept_decay=0.985,
    worse_accept_min_temperature=0.001, worse_selection_temperature=0.02,
    worse_selection_decay=0.985, worse_selection_uniform_mix=0.10,
    worse_selection_elite_count=2)


def algo_settings(city):
    s = dict(bco=200, bco_bees=10, bco_args={})
    if city == "Mandl":
        s["bco_args"] = dict(_MANDL_BCO_ARGS)
    return s


CTX = ex.ExperimentContext(
    algo_settings=algo_settings, connectivity_mode=CONNECTIVITY_MODE,
    unified_weights=UNIFIED_COST_WEIGHTS, unified_adj=UNIFIED_ADJ,
    early_stop_patience=None, hh_max_repair_iters=5000,
    our_model_path="artifacts/model_weights/improvement/"
                   "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt",
    bco_n_type1_bees=int(bco_config().n_type1_bees), include_edit_rebuild=False)


# --- bee-mix variant dicts (extracted verbatim from the notebook cells) ------
NEURAL_BCO_VARIANT = dict(run_name="neural_bco", use_neural_bees=True,
                          n_type1_bees=int(bco_config().n_type1_bees), n_type2_bees=None,
                          n_type4_bees=0, n_type5_bees=0, n_type6_bees=0,
                          n_type7_bees=0)
CLASSIC_BCO_VARIANT = dict(run_name="classic_bco", use_neural_bees=False,
                           n_type1_bees=int(bco_config().n_type1_bees), n_type2_bees=None,
                           n_type4_bees=0, n_type5_bees=0, n_type6_bees=0,
                           n_type7_bees=0)
TRIM12_EXTEND12_VARIANT = dict(run_name="trim12_extend12", use_neural_bees=True,
                               n_bees=24, n_type1_bees=0, n_type2_bees=0,
                               n_type4_bees=12, n_type5_bees=0, n_type6_bees=12,
                               n_type7_bees=0, type4_allow_halt=True,
                               type6_allow_halt=True)


def _save(cfg, experiment, name):
    path = OUT_DIR / experiment / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    # round-trip: reloading must reproduce the captured config exactly
    reloaded = OmegaConf.load(path)
    assert OmegaConf.to_container(reloaded) == OmegaConf.to_container(cfg), name
    return path


def _apply_iters_target(cfg, n_iterations, adj_target):
    ex.bco_cfg_set(cfg, n_iterations=int(n_iterations),
                   **dict(UNIFIED_ADJ, adjustment_degree_target=float(adj_target)))
    return cfg


def capture():
    n = 0
    # our NBCO (type1 GNN rebuild + type5 edit) for each city used by the sweeps
    for city in ("Mandl", "Mumford0", "Mumford1", "Mumford2", "Mumford3"):
        spec = SPECS[city]
        cfg = ex.our_model_cfg(CTX, city, spec, adj_target=ADJ_TARGET, use_gnn=True)
        _save(cfg, "nbco_variants", f"our_nbco_{city.lower()}")
        n += 1
        cfg = ex.variant_bco_cfg(CTX, city, spec, NEURAL_BCO_VARIANT)
        _save(cfg, "nbco_variants", f"neural_bco_{city.lower()}")
        n += 1
        cfg = ex.variant_bco_cfg(CTX, city, spec, CLASSIC_BCO_VARIANT)
        _save(cfg, "nbco_variants", f"classic_bco_{city.lower()}")
        n += 1
        cfg = ex.variant_bco_cfg(CTX, city, spec, TRIM12_EXTEND12_VARIANT)
        _save(cfg, "nbco_variants", f"trim12_extend12_{city.lower()}")
        n += 1

    # E2 5-model ablation (GNN/RPC x trim-extend/type2) on Mumford1.
    for construct, edit, name in [("RPC", "trim/extend", "rpc_trim_extend"),
                                  ("GNN", "type2", "gnn_type2"),
                                  ("RPC", "type2", "rpc_type2")]:
        variant = ex.abl_variant(construct, edit, 10)
        cfg = ex.variant_bco_cfg(CTX, "Mumford1", SPECS["Mumford1"], variant)
        _save(cfg, "e2_ablation", f"{name}_mumford1")
        n += 1

    # EKB (Ekaterinburg) -- the active case study; spec is data-derived.
    try:
        from .ekb import ekb_spec, load_ekb_routes
        ekb = ekb_spec(load_ekb_routes())
        cfg = ex.our_model_cfg(CTX, "EKB", ekb, adj_target=ADJ_TARGET,
                               use_gnn=True, force_cpu=True)
        _save(cfg, "ekb", "our_nbco_ekb")
        n += 1
        print(f"[EKB] captured our_nbco (spec={ekb})")
    except Exception as exc:  # pragma: no cover - data may be absent
        print(f"[EKB] skipped (data unavailable): {exc!r}")

    print(f"captured {n} config-first experiment YAMLs -> {OUT_DIR}")
    return n


if __name__ == "__main__":
    capture()
