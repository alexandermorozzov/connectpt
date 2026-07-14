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

from connectpt.routes_generator.core import ExperimentRunFactory, build_experiment
from connectpt.routes_generator.search import BeeColonySearchRun

REPO = Path(__file__).resolve().parents[1]
LIB_CFG = REPO / "connectpt" / "routes_generator" / "cfg"
EXP_DIR = LIB_CFG / "experiments"

# Paper artifacts as flat, self-contained files (M018): one YAML per table/figure.
# A multi-method artifact carries a ``methods:`` list (each method = a group
# choice picked in code) instead of per-method leaf files.
FLAT_ARTIFACTS = [
    "experiments/table3_nbco_vs_our",
    "experiments/table4_fig4_our_pareto",
    "experiments/table5_fig5_5model",
    "experiments/ekb_case_study",
    "experiments/macsa_alpha_sweep",
    "experiments/macsa_alpha_sweep_iter1",
]
# No nested declarative dirs left (paper artifacts are all flat; bee_type_comparison
# is an internal ablation, seeded/ + construction_only are test fixtures).
SECTIONS: list[str] = []


def _config_names():
    names = list(FLAT_ARTIFACTS)
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

    # methods-based artifact: each method re-composes THIS config with its group
    # choice (bee_sets + models) picked in code -- exactly what ExperimentBatch does.
    methods = cfg.get("methods")
    if methods:
        for m in methods:
            run_cfg = build_experiment(
                name, bee_sets=m["bee_sets"], models=m["models"],
                n_bees=m.get("n_bees"), label=m["label"])
            _check_run(f"{name}[{m['label']}]", run_cfg)
        return

    # heterogeneous batch: a list of separate run configs (e.g. GA/SA baselines).
    if cfg.get("batch") is not None and cfg.batch.get("runs"):
        for run_name in cfg.batch.runs:
            _check_run(str(run_name), _compose(str(run_name)))
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


def test_construction_only_from_scratch_config():
    """The construction-only, NON-SEEDED config (new-logic analog of the deleted
    frozen evaluation.ipynb learned-construction runs): composes, has no sweep /
    no data.source (so run() takes the from-scratch run_suite path), and its
    n_bees matches the construction bee-set total."""
    cfg = _compose("experiments/construction_only/mandl")
    assert cfg.run.type == "bee_colony_search"
    assert cfg.get("sweep") is None, "from-scratch run must NOT declare a sweep"
    assert cfg.data.get("source") is None, "from-scratch run uses the benchmark city shape"
    total = sum(int(b["count"]) for b in cfg.bees)
    assert int(cfg.search.n_bees) == total
    assert all(b["policy"] == "construction" for b in cfg.bees), "construction-only"
    assert isinstance(ExperimentRunFactory.from_cfg(cfg), BeeColonySearchRun)
