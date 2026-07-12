"""MACSA Table-B case study as ONE config-first call for the notebook.

``run_macsa_table_b(suite)`` runs the whole Mandl-8 Table-B workflow:

1. the Our-NBCO alpha sweep goes through the SAME declarative path as every
   other experiment (:func:`paper_runs.run_experiment` on
   ``macsa/mandl8/our_nbco_alpha_sweep``; the ``_smoke`` variant is one BCO
   iteration per point) -- the grid, budget, bee set and model checkpoints all
   live in the experiment YAML, nothing here;
2. what stays in this module is only the case-study specificity: reading the
   fixed Table-B route sets from the scenario txt files, scoring them (and the
   best sweep solution) as FIXED networks via
   :func:`evaluation.score_fixed_routes`, selecting the best sweep solution
   that beats MACSA on both RTT and WMC, and drawing the route grids through
   the shared :func:`reports.plot_routes_grid`.

Every run recomputes and overwrites its outputs under the suite's paper folder;
the returned :class:`MacsaResult` is what the notebook cell displays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from ..core.paths import DATASETS_DIR
from ..data.routes import as_route_tensor
from ..evaluation import adj_vs_init, score_fixed_routes
from ..objectives import load_unified_objective
from ..reports import paper_row, plot_routes_grid, save_paper_fig
from ._common import pad_routes_to
from .paper_runs import paper_dir, run_experiment, save_paper

# --- static scenario constants --------------------------------------------------
SCENARIO_NAME = "mandl_8"
SCENARIO_DIR = DATASETS_DIR / "MACSA_data" / SCENARIO_NAME
METHOD_ORDER = ["original", "rga", "as", "ga", "ras", "mmas", "ma", "macsa"]
METHOD_TITLE = {
    "original": "Original network [4]",
    "rga": "RGA",
    "as": "AS",
    "ga": "GA",
    "ras": "RAS",
    "mmas": "MMAS",
    "ma": "MA",
    "macsa": "MACSA",
}
REF_METHOD = METHOD_TITLE["original"]
ARTICLE_STEM = "final_macsa_mandl8_tableb_article_only"
COMPARISON_STEM = "final_macsa_mandl8_tableb"
SWEEP_EXPERIMENT = "macsa/mandl8/our_nbco_alpha_sweep"
# The smoke path is a distinct paper config (Table 6, iter=1) -- NOT a generic
# 2-iter dry-run -- so it is selected by name and run with its own budget.
SWEEP_EXPERIMENT_SMOKE = "macsa/mandl8/our_nbco_alpha_sweep_iter1"
NODE_SIZE = 70.0

# Fixed-network scoring point: the paper's eval alpha + the unified objective's
# two-sided adjustment penalty (Table-B rows and the best-solution row are all
# scored at this SAME point; the sweep rows keep the metrics of their own run).
_OBJ = load_unified_objective()
EVAL_ALPHA = 0.5
EVAL_ADJ_TARGET = float(_OBJ.adj_target)
EVAL_ADJ_OBJECTIVE = str(_OBJ.adj_objective)


@dataclass
class MacsaResult:
    """Everything the MACSA cell needs to display: tables, routes, figures."""

    tableb: pd.DataFrame
    sweep: pd.DataFrame
    comparison: pd.DataFrame
    best_label: str
    figures: dict = field(default_factory=dict)

    def display(self) -> None:
        from IPython.display import display as show
        for name, table in (("Table-B", self.tableb), ("Our-NBCO sweep", self.sweep),
                            ("comparison", self.comparison)):
            if table is not None and len(table):
                show(table)
        for fig in self.figures.values():
            show(fig)


# --- Table-B specific helpers ----------------------------------------------------


def read_routes_0indexed(path):
    """Read one ``routes_<method>.txt`` into a padded (-1) route tensor."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            rows.append([int(tok) for tok in line.split()])
    if not rows:
        raise ValueError(f"No routes in {path}")
    width = max(len(row) for row in rows)
    out = torch.full((len(rows), width), -1, dtype=torch.long)
    for idx, row in enumerate(rows):
        out[idx, :len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def _build_spec(raw_routes, n_nodes):
    """Route bounds accommodating the LONGEST route across all Table-B sets."""
    n_routes = int(raw_routes[REF_METHOD].shape[0])
    longest = max(int((rt > -1).sum(-1).max().item()) for rt in raw_routes.values())
    return {"n_routes": n_routes,
            "min_route_len": 2,
            "max_route_len": min(int(n_nodes), max(12, longest))}


def _alpha_label(alpha, n_iterations):
    return f"Our NBCO alpha={float(alpha):.1f} (iter={int(n_iterations)})"


def _metric_subtitle(row):
    if row is None:
        return None
    return (f"ATT={float(row['ATT']):.2f}  WMC={float(row['WMC']):.2f}\n"
            f"RTT={float(row['RTT']):.0f}  cost={float(row['cost']):.3f}  "
            f"adj={float(row['adj_vs_seed']):.3f}")


def _score(method, source, routes, *, seed_routes, tensors, spec):
    """Score a FIXED route set at the paper's eval point -> (paper row, routes)."""
    t0 = perf_counter()
    metrics, scored = score_fixed_routes(
        routes, tensors, spec, alpha=EVAL_ALPHA, adj_target=EVAL_ADJ_TARGET,
        adj_objective=EVAL_ADJ_OBJECTIVE, seed_routes=seed_routes)
    row = paper_row(SCENARIO_NAME, method, source, metrics, scored, seed_routes,
                    duration_s=perf_counter() - t0)
    row.update(eval_alpha=EVAL_ALPHA,
               eval_adj_target=EVAL_ADJ_TARGET,
               eval_adj_objective=EVAL_ADJ_OBJECTIVE,
               objective=f"alpha*RTT + (1-alpha)*WMC + adj({EVAL_ADJ_OBJECTIVE})")
    return row, scored


def _select_best(sweep_df, macsa_row):
    """The sweep row beating MACSA on both RTT and WMC with the best joint score
    (falls back to the best joint score overall when none beats it)."""
    work = sweep_df.copy()
    work["beats_macsa_rtt_wmc"] = ((work["RTT"].astype(float) < float(macsa_row["RTT"])) &
                                    (work["WMC"].astype(float) < float(macsa_row["WMC"])))
    work["joint_norm_rtt_wmc"] = (work["RTT"].astype(float) / float(macsa_row["RTT"]) +
                                  work["WMC"].astype(float) / float(macsa_row["WMC"]))
    pool = work[work["beats_macsa_rtt_wmc"]].copy()
    if pool.empty:
        pool = work.copy()
    pool = pool.sort_values(["joint_norm_rtt_wmc", "RTT", "WMC", "alpha"], ascending=True)
    return pool.iloc[0].to_dict()


# --- the one call ---------------------------------------------------------------


def run_macsa_table_b(suite) -> MacsaResult:
    """Score Table-B, sweep Our NBCO, compare, draw grids -- one call, one object."""
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite)

    # 1) the alpha sweep -- the same declarative path as every other experiment
    #    (persisted + Pareto-rendered by run_experiment, stem from the YAML). The
    #    smoke profile runs the iter-1 paper table (its own budget), so smoke is
    #    off here -- it must not be capped to the generic dry-run budget.
    sweep_name = SWEEP_EXPERIMENT_SMOKE if suite.smoke else SWEEP_EXPERIMENT
    sweep = run_experiment(sweep_name, suite, kind="pareto", smoke=False,
                           title="Mandl-8 MACSA: Our NBCO alpha sweep")
    art = sweep.artifact
    sweep_df = art.table
    n_iterations = int(sweep_df["n_iterations"].iloc[0])
    # per-point routes/subtitles, in the sweep grid's construction order
    point_keys = [k for k in art.routes if k != "Initial"]
    sweep_routes, sweep_subs = {}, {}
    for key, (_, row) in zip(point_keys, sweep_df.iterrows()):
        label = _alpha_label(float(row["alpha"]), n_iterations)
        sweep_routes[label] = as_route_tensor(art.routes[key])
        sweep_subs[label] = _metric_subtitle(row)

    inst = art.instance
    tensors, coords, street_adj = inst.tensors, inst.coords, inst.street_adj
    demand = tensors["demand"]

    # 2) Table-B fixed route sets from the scenario txt files, scored as-is.
    raw = {METHOD_TITLE[key]: read_routes_0indexed(SCENARIO_DIR / f"routes_{key}.txt")
           for key in METHOD_ORDER}
    spec = _build_spec(raw, int(coords.shape[0]))
    padded = {method: pad_routes_to(routes, spec["n_routes"], spec["max_route_len"],
                                    strict=True)
              for method, routes in raw.items()}
    seed_routes = padded[REF_METHOD]
    adj_realized = float(adj_vs_init(padded["MACSA"], seed_routes))

    rows, tableb_routes = [], {}
    for method, routes in padded.items():
        row, scored = _score(method, "macsa_table_b", routes,
                             seed_routes=seed_routes, tensors=tensors, spec=spec)
        row.update(alpha=np.nan, run_alpha=np.nan, n_iterations=np.nan,
                   macsa_adj_target=adj_realized)
        rows.append(row)
        tableb_routes[method] = scored
    tableb_df = pd.DataFrame(rows).round(6)
    save_paper(suite, ARTICLE_STEM, table=tableb_df, routes=tableb_routes,
               coords=coords, street_adj=street_adj,
               meta={"scenario": SCENARIO_NAME})

    # 3) best sweep solution vs MACSA + the combined comparison table.
    macsa_row = tableb_df[tableb_df["method"] == "MACSA"].iloc[0]
    best = _select_best(sweep_df, macsa_row)
    best_alpha = float(best["alpha"])
    best_key = _alpha_label(best_alpha, n_iterations)
    compare_label = f"Our NBCO best (alpha={best_alpha:.1f}, iter={n_iterations})"
    row, best_scored = _score(compare_label, "our_nbco_alpha_sweep_best",
                              sweep_routes[best_key], seed_routes=seed_routes,
                              tensors=tensors, spec=spec)
    row.update(alpha=best_alpha, run_alpha=best_alpha, n_iterations=n_iterations,
               macsa_adj_target=adj_realized,
               beats_macsa_rtt_wmc=bool(best["beats_macsa_rtt_wmc"]),
               selected_from=best_key)
    comparison_routes = dict(tableb_routes)
    comparison_routes[compare_label] = best_scored
    comparison_df = pd.concat([tableb_df, pd.DataFrame([row])],
                              ignore_index=True, sort=False).round(6)
    save_paper(suite, COMPARISON_STEM, table=comparison_df, routes=comparison_routes,
               coords=coords, street_adj=street_adj,
               meta={"scenario": SCENARIO_NAME, "best": compare_label})

    # 4) the four route grids -- the shared grid plotter, MACSA styling on top.
    cmp_subs = {str(r["method"]): _metric_subtitle(r)
                for _, r in comparison_df.iterrows()}
    style = dict(demand=demand, node_size=NODE_SIZE, node_label_offset=1,
                 palette="tab20", with_overlap_curves=True, show_node_labels=True)
    figures = dict(sweep.figures)
    grids = [
        ("sweep_plain", f"{SCENARIO_NAME}_sweep_plain",
         dict(route_sets=sweep_routes, subtitles=sweep_subs, ncols=4,
              title="Mandl-8 MACSA: Our NBCO alpha sweep")),
        ("sweep_diff", f"{SCENARIO_NAME}_sweep_diff",
         dict(route_sets=sweep_routes, subtitles=sweep_subs, ncols=4,
              diff_ref_routes=seed_routes,
              title="Mandl-8 MACSA: Our NBCO alpha sweep, diff vs original")),
        ("comparison_plain", f"{COMPARISON_STEM}_plain",
         dict(route_sets=comparison_routes, subtitles=cmp_subs, ncols=5,
              title="Mandl-8 MACSA Table B: paper methods + best Our NBCO")),
        ("comparison_diff", f"{COMPARISON_STEM}_diff",
         dict(route_sets=comparison_routes, subtitles=cmp_subs, ncols=5,
              diff_against=REF_METHOD,
              title="Mandl-8 MACSA Table B: diff vs original")),
    ]
    for key, fig_name, kw in grids:
        fig = plot_routes_grid(kw.pop("route_sets"), coords, street_adj,
                               **{**style, **kw})
        save_paper_fig(fig, fig_name, prefix=prefix, out_dir=out_dir)
        figures[key] = fig

    return MacsaResult(tableb=tableb_df, sweep=sweep_df, comparison=comparison_df,
                       best_label=compare_label, figures=figures)
