"""Per-city algorithm budgets (config-first).

The notebook's algo_settings + the DEFAULT/MANDL_ALGO_SETTINGS and *_ITERS_BY_CITY
magic tables move to cfg/search/budgets.yaml; make_algo_settings(smoke, quick)
returns the same algo_settings(city) closure the notebook had, reading the
tables from the YAML. The SA temperature schedule (computation, not data) stays
in code, reproduced verbatim.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from .core.paths import CFG_DIR


def load_budgets():
    return OmegaConf.to_container(
        OmegaConf.load(CFG_DIR / "search" / "budgets.yaml"), resolve=True)


def make_algo_settings(smoke: bool = False, quick: bool = False, budgets=None):
    B = budgets or load_budgets()

    def algo_settings(city):
        settings = dict(B["smoke" if smoke else "full"]["default"])
        settings["sa_args"] = {}
        settings["bco_args"] = dict(B["mandl_bco_args"]) if city == "Mandl" else {}
        if not (smoke or quick):
            for key in ("sa", "hh", "ga", "ga_pop"):
                settings[key] = B["by_city"][key].get(city, settings[key])
            settings["hh_max_repair_iters"] = B["by_city"]["hh_max_repair_iters"].get(
                city, B["hh_max_repair_iters_default"])
        if quick:
            settings.update(B["quick"])
        # SA temperature schedule: linear anneal over the full budget + reheating.
        _sa_n = int(settings["sa"])
        if city in {"Mumford2", "Mumford3"} and not (smoke or quick):
            _t0, _t1, _rt, _rf = 0.22, 0.004, max(1000, _sa_n // 25), 1.8
        else:
            _t0, _t1, _rt, _rf = 0.15, 0.005, max(500, _sa_n // 20), 1.5
        settings["sa_args"] = dict(schedule="linear", initial_temp=_t0, final_temp=_t1,
                                   cooling_rate=(_t0 - _t1) / max(_sa_n, 1),
                                   reheating_threshold=_rt, reheating_factor=_rf)
        return settings

    return algo_settings
