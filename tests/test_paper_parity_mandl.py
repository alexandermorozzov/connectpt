"""Approximate-parity: the C-layer reproduces the paper's Mandl numbers.

Targets are the curated CSVs in ``artifacts/paper_final`` (the same numbers as
the manuscript tables in template.tex). Comparisons are tolerance-based --
RTT/WMC/cost within a relative tolerance, adjustment degree within an absolute
one -- NOT bit-for-bit (the golden-parity discipline was retired in M010).

Two tiers:

* fast (default ``pytest -q``):
  - scoring parity of the pinned Table-1 initial network (no search at all);
  - the MACSA iter=1 alpha sweep -- the paper's lightest experiment
    (Table "macsa-alpha-sweep", 11 seeded runs x 1 BCO iteration);
  - a reduced-budget E1 Mandl run with loose bounds.
* full paper budget (opt-in ``pytest -m paper_full``, deselected by default):
  - Table 1 method rows (classic BCO / neural BCO / Our NBCO) at alpha=0.5,
    adj target 0.2;
  - Table 2 alpha sweeps (neural BCO iter=200, Our NBCO iter=500) at
    adj target 0.3.

The full-budget tier takes minutes to hours -- it is for the user to run
manually, never part of the fast suite.
"""
import math
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.core.paths import (
    CONSTRUCTION_MODEL_WEIGHTS_PATH, EDIT_MODEL_WEIGHTS_DIR)
from connectpt.routes_generator.search import BeeColonySearchRun

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"
PAPER_FINAL = REPO_ROOT / "artifacts" / "paper_final"

TABLE1_DIR = PAPER_FINAL / "table1_e1u_unified_alpha05_target02"
TABLE1_CSV = TABLE1_DIR / "final_main_unified_Mandl.csv"
TABLE1_ROUTES = TABLE1_DIR / "final_main_unified_Mandl_routes.pt"
TABLE2_DIR = PAPER_FINAL / "table2_nbco_vs_improved_target03"
TABLE6_CSV = (PAPER_FINAL / "table6_macsa_alpha_sweep_iter1"
              / "final_macsa_mandl8_alpha_sweep_iter1.csv")

EDIT_CKPT = EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_rttconn_adj_w10_t02_finetune100.pt"

needs_construction = pytest.mark.skipif(
    not CONSTRUCTION_MODEL_WEIGHTS_PATH.exists(),
    reason="construction weights not present")
needs_all_weights = pytest.mark.skipif(
    not (CONSTRUCTION_MODEL_WEIGHTS_PATH.exists() and EDIT_CKPT.exists()),
    reason="construction/edit weights not present")

# Exact-config runs: same seed/config as the paper, tolerance only absorbs
# hardware/library nondeterminism. Reduced-budget runs get looser bounds.
REL_TOL = 0.05
ADJ_ABS_TOL = 0.05
GATED_METRICS = ("RTT", "WMC", "cost")


def _targets(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        pytest.skip(f"paper targets not present: {csv_path}")
    return pd.read_csv(csv_path)


def _target_row(df: pd.DataFrame, method: str) -> pd.Series:
    hit = df[df["method"] == method]
    assert len(hit) == 1, f"expected 1 target row for {method!r}, got {len(hit)}"
    return hit.iloc[0]


def _compare(actual, target, *, label, rel=REL_TOL, adj_abs=ADJ_ABS_TOL,
             metrics=GATED_METRICS):
    """Collect (not raise) tolerance violations for one result row."""
    problems = []
    for key in metrics:
        a, t = float(actual[key]), float(target[key])
        if not math.isfinite(a) or abs(a - t) > rel * abs(t):
            problems.append(
                f"{label}: {key} actual={a:.4f} target={t:.4f} "
                f"(off by {abs(a - t) / abs(t):.1%}, allowed {rel:.0%})")
    a, t = float(actual["adj_vs_seed"]), float(target["adj_vs_seed"])
    if abs(a - t) > adj_abs:
        problems.append(
            f"{label}: adj_vs_seed actual={a:.4f} target={t:.4f} "
            f"(allowed abs {adj_abs})")
    return problems


def _compose(config_name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _run_sweep(config_name: str, tmp_path: Path, extra_overrides: list[str]):
    # run.silent=false turns the BCO per-iteration tqdm on; pytest captures it
    # unless run with -s, so use ``pytest -s`` to watch search progress live.
    cfg = _compose(config_name,
                   [f"paths.output_dir={tmp_path.as_posix()}",
                    "+run.silent=false", *extra_overrides])
    art = BeeColonySearchRun(cfg).run()
    assert art.table is not None and len(art.table) > 0
    return art.table


# --- tier A: fast ---------------------------------------------------------


def test_initial_network_scoring_matches_table1():
    """Metric pipeline parity, no search: score the pinned Table-1 initial
    network under the unified objective (alpha=0.5, target=0.2) and compare
    with the 'Initial (LC+realistic)' row. Deterministic."""
    from connectpt.routes_generator.data.sources import BenchmarkDataSource
    from connectpt.routes_generator.evaluation import (full_metric_row,
                                                       score_fixed_routes)

    targets = _targets(TABLE1_CSV)
    if not TABLE1_ROUTES.exists():
        pytest.skip(f"paper routes dump not present: {TABLE1_ROUTES}")
    inst = BenchmarkDataSource(
        city="Mandl", init_dump=str(TABLE1_ROUTES)).load()

    # The paper's Initial rows are scored WITHOUT the adjustment penalty (vs
    # its own seed it is a degenerate constant 10*|0-target|), so pass no
    # adj_target -- the atom keeps the penalty off.
    metrics, scored = score_fixed_routes(
        inst.init_routes, inst.tensors, inst.spec, alpha=0.5)
    row = full_metric_row(metrics, scored, inst.init_routes)

    problems = _compare(row, _target_row(targets, "Initial (LC+realistic)"),
                        label="Initial")
    assert not problems, "\n".join(problems)


@needs_all_weights
def test_macsa_alpha_sweep_iter1_parity(tmp_path):
    """The paper's lightest experiment, exact config: Our NBCO on the Mandl-8
    MACSA network, 1 BCO iteration per alpha point (Table macsa-alpha-sweep).

    A single BCO iteration does not average out hardware nondeterminism (the
    paper values were produced on GPU, CI/dev boxes may run CPU), so instead of
    a flat 5% gate we check what the objective optimizes at each point:

    * cost -- no worse than the paper by 10% (improvements are fine), plus a
      sanity floor catching metric-definition breaks;
    * WMC within 15% where it carries weight (alpha <= 0.5), RTT within 15%
      where it does (alpha >= 0.5);
    * realized adjustment degree within 0.05 of the paper's.
    """
    targets = _targets(TABLE6_CSV)
    # the smoke variant IS the paper's iter=1 experiment (Table 6).
    table = _run_sweep("experiments/macsa/mandl8/our_nbco_alpha_sweep_iter1",
                       tmp_path, [])

    problems = []
    for _, target in targets.iterrows():
        alpha = float(target["alpha"])
        hit = table[abs(table["alpha"].astype(float) - alpha) < 1e-9]
        assert len(hit) == 1, f"no sweep row for alpha={alpha}"
        row = hit.iloc[0]
        label = f"alpha={alpha}"

        cost, cost_t = float(row["cost"]), float(target["cost"])
        if not math.isfinite(cost) or cost > 1.10 * cost_t:
            problems.append(f"{label}: cost actual={cost:.4f} worse than "
                            f"target={cost_t:.4f} by more than 10%")
        if cost < 0.5 * cost_t:
            problems.append(f"{label}: cost actual={cost:.4f} implausibly far "
                            f"below target={cost_t:.4f} (metric break?)")
        gated = [k for k, on in (("WMC", alpha <= 0.5), ("RTT", alpha >= 0.5))
                 if on]
        problems += _compare(row, target, label=label, rel=0.15,
                             metrics=gated)
    assert not problems, "\n".join(problems)


@needs_all_weights
def test_e1_mandl_reduced_budget(tmp_path):
    """Reduced-budget surrogate of Table 1 (Our NBCO, alpha=0.5, target=0.2):
    10 of the paper's 200 iterations, so only loose bounds -- the search must
    clearly improve on the initial network and land in the target's
    neighbourhood, not reproduce it."""
    targets = _targets(TABLE1_CSV)
    if not TABLE1_ROUTES.exists():
        pytest.skip(f"paper routes dump not present: {TABLE1_ROUTES}")
    initial = _target_row(targets, "Initial (LC+realistic)")
    target = _target_row(targets, "Our NBCO (GNN rebuild + trim/extend)")

    table = _run_sweep(
        "experiments/e1/our_nbco", tmp_path,
        ["sweep.alpha=[0.5]", "sweep.adj_target=0.2", "sweep.n_iterations=10",
         f"+data.init_dump={TABLE1_ROUTES.as_posix()}"])
    row = table.iloc[0]

    problems = []
    if not float(row["cost"]) < float(initial["cost"]):
        problems.append(f"cost {row['cost']:.3f} did not improve on the "
                        f"initial {initial['cost']:.3f}")
    if float(row["cost"]) > 2.0 * float(target["cost"]):
        problems.append(f"cost {row['cost']:.3f} > 2x paper target "
                        f"{target['cost']:.3f}")
    for key in ("RTT", "WMC"):
        a, t = float(row[key]), float(target[key])
        if abs(a - t) > 0.30 * t:
            problems.append(f"{key} actual={a:.3f} target={t:.3f} (off by "
                            f"{abs(a - t) / t:.1%}, allowed 30%)")
    if float(row["adj_vs_seed"]) > 0.35:
        problems.append(f"adj_vs_seed {row['adj_vs_seed']:.3f} > 0.35 "
                        "(target 0.2)")
    assert not problems, "\n".join(problems)


# --- tier B: full paper budget (opt-in) -----------------------------------

FULL_BUDGET_CASES = [
    pytest.param(
        "experiments/e1/neural_bco",
        ["search/models=no_neural_models", "search/bee_sets=classic_bco_paper",
         "search.n_bees=10", "sweep.n_iterations=500"],
        "BCO", id="classic_bco"),
    pytest.param(
        "experiments/e1/neural_bco", ["sweep.n_iterations=200"],
        "neural BCO", id="neural_bco",
        marks=needs_construction),
    pytest.param(
        "experiments/e1/our_nbco", ["sweep.n_iterations=200"],
        "Our NBCO (GNN rebuild + trim/extend)", id="our_nbco",
        marks=needs_all_weights),
]


@pytest.mark.paper_full
@pytest.mark.parametrize("config_name, overrides, method", FULL_BUDGET_CASES)
def test_e1_mandl_full_budget_parity(tmp_path, config_name, overrides, method):
    """Table 1 Mandl method rows at the paper's budget (alpha=0.5, target=0.2,
    same pinned initial network). Minutes per method -- run manually via
    ``pytest -m paper_full``."""
    targets = _targets(TABLE1_CSV)
    if not TABLE1_ROUTES.exists():
        pytest.skip(f"paper routes dump not present: {TABLE1_ROUTES}")
    table = _run_sweep(
        config_name, tmp_path,
        ["sweep.alpha=[0.5]", "sweep.adj_target=0.2",
         f"+data.init_dump={TABLE1_ROUTES.as_posix()}", *overrides])

    problems = _compare(table.iloc[0], _target_row(targets, method),
                        label=method)
    assert not problems, "\n".join(problems)


TABLE2_CASES = [
    pytest.param(
        "experiments/e1/neural_bco",
        "final_main_unified_Mandl_nbco_only_target03_iter200",
        "neural BCO", id="neural_bco", marks=needs_construction),
    pytest.param(
        "experiments/e1/our_nbco",
        "final_main_unified_Mandl_our_only_target03_iter500",
        "Our NBCO", id="our_nbco", marks=needs_all_weights),
]


@pytest.mark.paper_full
@pytest.mark.parametrize("config_name, stem, method_prefix", TABLE2_CASES)
def test_table2_mandl_parity(tmp_path, config_name, stem, method_prefix):
    """Table 2 Mandl (alpha in {0, 0.5, 1}, adj target 0.3) at the paper's
    budget, seeded from the table's own pinned initial network."""
    targets = _targets(TABLE2_DIR / f"{stem}.csv")
    routes_pt = TABLE2_DIR / f"{stem}_routes.pt"
    if not routes_pt.exists():
        pytest.skip(f"paper routes dump not present: {routes_pt}")

    method_rows = targets[targets["method"].str.startswith(method_prefix)]
    assert len(method_rows) == 3, f"expected 3 target rows, got {len(method_rows)}"
    n_iterations = int(method_rows["n_iterations"].iloc[0])

    table = _run_sweep(
        config_name, tmp_path,
        [f"sweep.n_iterations={n_iterations}",
         f"+data.init_dump={routes_pt.as_posix()}"])

    problems = []
    for _, target in method_rows.iterrows():
        alpha = float(target["alpha"])
        hit = table[abs(table["alpha"].astype(float) - alpha) < 1e-9]
        assert len(hit) == 1, f"no sweep row for alpha={alpha}"
        problems += _compare(hit.iloc[0], target, label=f"alpha={alpha}")
    assert not problems, "\n".join(problems)
