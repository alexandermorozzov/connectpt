"""Declarative experiment configs compose + dispatch (M010 stage 3).

The flat ``n_type`` method configs are being replaced by declarative
``bee_colony_search`` runs that compose the reusable groups (bee_colony_base +
models group + bee_sets group) and attach ``data`` / ``sweep`` / ``metrics``. A
run config must: compose standalone, carry ``run.type: bee_colony_search``,
declare ``n_bees`` equal to its bee-set total, and dispatch to
``BeeColonySearchRun`` via the factory. A batch config must list run configs
that each satisfy the above. This scans the converted sections so every new
section is covered automatically.
"""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.core import ExperimentRunFactory
from connectpt.routes_generator.search import BeeColonySearchRun

REPO = Path(__file__).resolve().parents[1]
LIB_CFG = REPO / "connectpt" / "routes_generator" / "cfg"
EXP_DIR = LIB_CFG / "experiments"

# Sections converted to the declarative bee_colony_search shape (stage 3). Each
# entry is a dir under experiments/ that holds ONLY declarative run/batch configs
# (the legacy flat n_type configs live elsewhere and are deleted at stage 4/7).
SECTIONS = ["e1", "m0", "macsa/mandl8", "ekb/case_study", "e2"]


def _config_names():
    names = []
    for section in SECTIONS:
        for p in sorted((EXP_DIR / section).rglob("*.yaml")):
            names.append(p.relative_to(LIB_CFG).with_suffix("").as_posix())
    return names


def _compose(name):
    with initialize_config_dir(config_dir=str(LIB_CFG), version_base=None):
        return compose(config_name=name)


@pytest.mark.parametrize("name", _config_names())
def test_declarative_config_composes_and_dispatches(name):
    cfg = _compose(name)

    if cfg.get("batch") is not None:
        assert cfg.batch.runs, f"{name}: empty batch"
        for run_name in cfg.batch.runs:
            run_cfg = _compose(str(run_name))
            _check_run(str(run_name), run_cfg)
        return

    _check_run(name, cfg)


def _check_run(name, cfg):
    assert cfg.run.type == "bee_colony_search", f"{name}: wrong run.type"
    # n_bees must match the declarative bee-set total (the plan the runner builds).
    total = sum(int(b["count"]) for b in cfg.bees)
    assert int(cfg.search.n_bees) == total, f"{name}: n_bees != bee-set total"
    # a sweep run carries data + sweep + a non-empty metric selection.
    assert cfg.data.get("source") is not None, f"{name}: no data.source"
    assert cfg.get("sweep") is not None, f"{name}: no sweep block"
    assert list(cfg.get("metrics", [])), f"{name}: no metrics"
    assert isinstance(ExperimentRunFactory.from_cfg(cfg), BeeColonySearchRun)
