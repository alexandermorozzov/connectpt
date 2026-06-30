"""MACSA Table-B case study: scoring, the Our-NBCO alpha sweep, and route grids.

This is a one-off paper experiment, not part of the reusable library. The thick
helpers used to live inline in ``paper_combined.ipynb``; they are collected here
so the notebook keeps only the data loading, orchestration and computed results.

Runtime-dependent values (smoke vs full iteration count, the Our-NBCO checkpoint
path, the seed) are injected via :func:`configure` -- call it once before using
any helper. The static scenario constants and all helpers are exported via
``__all__`` so the notebook can ``from experiments.macsa import *``.
"""
from __future__ import annotations

import contextlib
import io
import math
import time as _t
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from IPython.display import Image, display

from eval_lib import plots as route_plots
from eval_lib.baselines import _run_baseline
from eval_lib.context import DATASETS_DIR
from eval_lib.helpers import as_route_tensor, build_bco_cfg, run_bco
import eval_lib.helpers as _eh
from eval_lib.paper import (PAPER_DIR, UNIFIED_ADJ, bco_cfg_set,
                            eval_routes_cfg as _eval_routes_cfg,
                            paper_row as _row,
                            set_cfg_value as _set_cfg_value,
                            unify_weights as _unify_weights)
from eval_lib.params import ADJ_OBJECTIVE, ADJ_TARGET, CONNECTIVITY_MODE

# --- static scenario constants -------------------------------------------------
MACSA_SCENARIO_NAME = "mandl_8"
MACSA_SCENARIO_DIR = DATASETS_DIR / "MACSA_data" / MACSA_SCENARIO_NAME
MACSA_METHOD_ORDER = ["original", "rga", "as", "ga", "ras", "mmas", "ma", "macsa"]
MACSA_METHOD_TITLE = {
    "original": "Original network [4]",
    "rga": "RGA",
    "as": "AS",
    "ga": "GA",
    "ras": "RAS",
    "mmas": "MMAS",
    "ma": "MA",
    "macsa": "MACSA",
}
MACSA_REF_METHOD = MACSA_METHOD_TITLE["original"]
MACSA_ARTICLE_STEM = "final_macsa_mandl8_tableb_article_only"
MACSA_SWEEP_BEES = 10
MACSA_SWEEP_FORCE_RERUN = False
MACSA_FORCE_CPU = False
MACSA_ALPHA_GRID = [round(i / 10.0, 1) for i in range(11)]
MACSA_COMPARISON_STEM = "final_macsa_mandl8_tableb"
MACSA_TABLEB_EVAL_ALPHA = 0.5
MACSA_TABLEB_EVAL_ADJ_TARGET = float(ADJ_TARGET)
MACSA_TABLEB_EVAL_ADJ_OBJECTIVE = str(ADJ_OBJECTIVE)
MACSA_NODE_SIZE = 70.0
MACSA_DPI = 220

# --- runtime-configured values (set by configure()) ----------------------------
MACSA_SWEEP_BCO_ITERATIONS = 100
MACSA_SWEEP_SEED = 0
MACSA_SWEEP_STEM = f"final_macsa_mandl8_alpha_sweep_iter{MACSA_SWEEP_BCO_ITERATIONS}"
OUR_MODEL_PATH = None


def configure(*, smoke=False, our_model_path=None, seeds=None):
    """Inject the runtime-dependent values and wire the Our-NBCO checkpoint.

    ``smoke`` collapses the sweep to a single BCO iteration; ``our_model_path``
    selects the edit-model checkpoint (defaulting to the finetuned adj model);
    ``seeds`` provides the sweep seed. Mirrors the side effects the notebook cell
    used to perform inline (setting ``eval_lib.helpers`` module globals).
    """
    global MACSA_SWEEP_BCO_ITERATIONS, MACSA_SWEEP_SEED, MACSA_SWEEP_STEM, OUR_MODEL_PATH
    MACSA_SWEEP_BCO_ITERATIONS = 1 if smoke else 100
    if our_model_path is None:
        from eval_lib.context import EDIT_MODEL_WEIGHTS_DIR
        our_model_path = (EDIT_MODEL_WEIGHTS_DIR /
                          "improvement_lc_rttconn_adj_w10_t02_finetune100.pt")
    OUR_MODEL_PATH = our_model_path
    seeds = seeds or [0]
    MACSA_SWEEP_SEED = int(seeds[0])
    MACSA_SWEEP_STEM = f"final_macsa_mandl8_alpha_sweep_iter{MACSA_SWEEP_BCO_ITERATIONS}"
    _eh.EDIT_MODEL_WEIGHTS_PATH = OUR_MODEL_PATH
    _eh.EDIT_MODEL_N_ADJ_COND_FEATS = 0
    return config_summary()


def config_summary():
    return {
        "scenario_dir": str(MACSA_SCENARIO_DIR),
        "alpha_grid": MACSA_ALPHA_GRID,
        "bco_iterations": MACSA_SWEEP_BCO_ITERATIONS,
        "bco_bees": MACSA_SWEEP_BEES,
        "seed": MACSA_SWEEP_SEED,
        "force_cpu": MACSA_FORCE_CPU,
        "our_model": Path(OUR_MODEL_PATH).name if OUR_MODEL_PATH else None,
    }


def _macsa_alpha_tag(alpha):
    return f"{float(alpha):.1f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p") or "0"


def _macsa_alpha_label(alpha):
    return f"Our NBCO alpha={float(alpha):.1f} (iter={MACSA_SWEEP_BCO_ITERATIONS})"


def _macsa_alpha_weights(alpha):
    alpha = float(alpha)
    return {"demand_time_weight": 0.0,
            "route_time_weight": alpha,
            "median_connectivity_weight": 1.0 - alpha}


def _macsa_set_alpha_weights(cfg, alpha):
    for key, value in _macsa_alpha_weights(alpha).items():
        _set_cfg_value(cfg, f"experiment.cost_function.kwargs.{key}", float(value))
    return cfg


def _macsa_read_routes_0indexed(path):
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


def _macsa_pad_routes(routes, n_routes, max_route_len):
    routes = as_route_tensor(routes).long()
    if routes.ndim == 3:
        routes = routes[0]
    if routes.shape[0] != int(n_routes):
        raise ValueError(f"Expected {n_routes} routes, got {tuple(routes.shape)}")
    if routes.shape[-1] > int(max_route_len):
        raise ValueError(f"Route width {routes.shape[-1]} exceeds {max_route_len}")
    if routes.shape[-1] < int(max_route_len):
        routes = torch.nn.functional.pad(routes, (0, int(max_route_len) - routes.shape[-1]), value=-1)
    return routes


def _macsa_2d(routes):
    routes = as_route_tensor(routes)
    return routes[0] if routes.ndim == 3 else routes


def _macsa_build_spec(raw_routes, n_nodes):
    n_routes = int(raw_routes[MACSA_REF_METHOD].shape[0])
    longest = max(int((rt > -1).sum(-1).max().item()) for rt in raw_routes.values())
    return {"city": MACSA_SCENARIO_NAME,
            "n_routes": n_routes,
            "min_route_len": 2,
            "max_route_len": min(int(n_nodes), max(12, longest))}


def _macsa_score_routes(method, source, routes, *, seed_routes, tensors, spec,
                        alpha, adj_target, adj_objective):
    cfg = _unify_weights(_eval_routes_cfg(MACSA_SCENARIO_NAME, spec))
    _macsa_set_alpha_weights(cfg, alpha)
    _set_cfg_value(cfg, "experiment.cpu", True)
    _set_cfg_value(cfg, "eval.csv", False)
    _set_cfg_value(cfg, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
    adj_kwargs = dict(UNIFIED_ADJ,
                      adjustment_degree_target=float(adj_target),
                      adjustment_degree_objective=str(adj_objective))
    t0 = _t.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        _run_name, metrics, _unserved, scored_routes = _run_baseline(
            None, cfg, routes,
            f"{MACSA_SCENARIO_NAME}_{method.lower().replace(' ', '_')}_score_",
            {}, tensors=tensors,
            use_weighted_connectivity=True,
            connectivity_mode=CONNECTIVITY_MODE,
            adjustment_seed_routes=seed_routes,
            **adj_kwargs)
    row = _row(MACSA_SCENARIO_NAME, method, source, metrics, scored_routes,
               seed_routes, duration_s=_t.perf_counter() - t0)
    row.update(eval_alpha=float(alpha),
               eval_adj_target=float(adj_target),
               eval_adj_objective=str(adj_objective),
               objective=f"alpha*RTT + (1-alpha)*WMC + adj({adj_objective})")
    return row, as_route_tensor(scored_routes)


def _macsa_build_our_cfg(spec, *, alpha, adj_target, adj_objective,
                         n_iterations, n_bees, seed, force_cpu):
    n_bees = int(n_bees)
    rebuild_bees = max(1, n_bees // 2)
    trim_extend_bees = n_bees - rebuild_bees
    cfg = build_bco_cfg(
        run_name=f"{MACSA_SCENARIO_NAME}_alpha_sweep_our_nbco_gnn_rebuild_trimext",
        n_routes=spec["n_routes"], min_route_len=spec["min_route_len"],
        max_route_len=spec["max_route_len"], use_neural_bees=True,
        n_bees=n_bees, n_type1_bees=rebuild_bees, n_type2_bees=0,
        n_type4_bees=0, n_type5_bees=trim_extend_bees,
        n_type6_bees=0, n_type7_bees=0,
        force_cpu=bool(force_cpu), connectivity_mode=CONNECTIVITY_MODE,
        worse_accept_temperature=0.02, worse_accept_decay=0.985,
        worse_accept_min_temperature=0.001,
        worse_selection_temperature=0.02, worse_selection_decay=0.985,
        worse_selection_uniform_mix=0.10, worse_selection_elite_count=2,
        **_macsa_alpha_weights(alpha))
    bco_cfg_set(cfg, n_iterations=int(n_iterations),
                type4_allow_halt=False, type5_allow_halt=False,
                type6_allow_halt=False, type7_allow_halt=False,
                **dict(UNIFIED_ADJ,
                       adjustment_degree_target=float(adj_target),
                       adjustment_degree_objective=str(adj_objective)))
    _macsa_set_alpha_weights(cfg, alpha)
    _set_cfg_value(cfg, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
    _set_cfg_value(cfg, "eval.csv", False)
    _set_cfg_value(cfg, "experiment.seed", int(seed))
    return cfg


def _macsa_run_our_nbco(seed_routes, *, tensors, spec, alpha, adj_target,
                        adj_objective, n_iterations, n_bees, seed, force_cpu):
    cfg = _macsa_build_our_cfg(spec, alpha=alpha, adj_target=adj_target,
                               adj_objective=adj_objective,
                               n_iterations=n_iterations, n_bees=n_bees,
                               seed=seed, force_cpu=force_cpu)
    t0 = _t.perf_counter()
    _run_name, _metrics, _unserved, routes, _mutation_counts = run_bco(
        cfg, seed_routes, tensors=tensors, run_name_scope=f"{MACSA_SCENARIO_NAME}_")
    return as_route_tensor(routes), _t.perf_counter() - t0


def _macsa_metric_subtitle(row):
    if row is None:
        return ""
    return (f"ATT={float(row['ATT']):.2f}  WMC={float(row['WMC']):.2f}\n"
            f"RTT={float(row['RTT']):.0f}  cost={float(row['cost']):.3f}  "
            f"adj={float(row['adj_vs_seed']):.3f}")


def _macsa_relabel_nodes_1indexed(ax):
    for txt in ax.texts:
        value = txt.get_text()
        if value.isdigit():
            txt.set_text(str(int(value) + 1))


def _macsa_draw_grid(*, routes, rows_by_method, coords, street_adj, demand,
                     diff=False, ref_key=None, ref_routes=None,
                     include_demand=True, include_ref=True, ncol=5,
                     title="MACSA routes", node_size=MACSA_NODE_SIZE):
    panels = []
    if include_demand:
        panels.append("__demand__")
    if include_ref and ref_key is not None and ref_key in routes:
        panels.append(ref_key)
    panels += [name for name in routes if not (include_ref and name == ref_key)]
    ncol = max(1, int(ncol))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 5.8 * nrow),
                             squeeze=False, constrained_layout=True)
    if ref_routes is None and ref_key is not None and ref_key in routes:
        ref_routes = routes[ref_key]
    ref_routes_2d = _macsa_2d(ref_routes) if ref_routes is not None else None
    for ax, panel in zip(axes.flat, panels):
        if panel == "__demand__":
            route_plots.plot_demand_graph(ax, demand, coords, street_adj,
                                          title="OD demand",
                                          subtitle="edge color/width = demand")
            _macsa_relabel_nodes_1indexed(ax)
            continue
        panel_routes = _macsa_2d(routes[panel])
        subtitle = _macsa_metric_subtitle(rows_by_method.get(panel))
        if diff and ref_routes_2d is not None and panel != ref_key:
            route_plots.plot_route_diff(ax, panel_routes, ref_routes_2d,
                                        coords, street_adj,
                                        title=f"{panel} vs original",
                                        subtitle=subtitle, palette="tab20",
                                        with_overlap_curves=True,
                                        show_node_labels=True,
                                        node_size=node_size)
        else:
            route_plots.plot_plain_route_set(ax, panel_routes, coords, street_adj,
                                             title=panel, subtitle=subtitle,
                                             palette="tab20",
                                             with_overlap_curves=True,
                                             show_node_labels=True,
                                             node_size=node_size)
        _macsa_relabel_nodes_1indexed(ax)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return fig


def _macsa_save_fig(fig, stem, suffix):
    path = PAPER_DIR / f"{stem}_{suffix}.png"
    fig.savefig(path, dpi=MACSA_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper] figure -> {path}")
    return path


def _macsa_display_image(path):
    try:
        display(Image(filename=str(path)))
    except Exception:
        print(path)


def _macsa_upsert_row(rows, row):
    return [r for r in rows if str(r.get("method")) != str(row.get("method"))] + [row]


def _macsa_select_best_sweep_row(sweep_df, macsa_row):
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


# Export the static scenario constants + every helper so the notebook can do
# ``from experiments.macsa import *`` and keep calling them by their bare names.
# OUR_MODEL_PATH is deliberately NOT exported so importing here never clobbers
# the notebook's own training-time OUR_MODEL_PATH.
__all__ = [n for n in dict(globals()) if n.startswith("MACSA_") or n.startswith("_macsa_")]
__all__ += ["configure", "config_summary"]
