"""MACSA Table-B case study: scoring, the Our-NBCO alpha sweep, and route grids.

This is a one-off paper experiment, not part of the reusable library. The thick
helpers used to live inline in ``paper_combined.ipynb``; they are collected here
so the notebook keeps only the data loading, orchestration and computed results.

Runtime-dependent values (smoke vs full iteration count, the Our-NBCO checkpoint
path, the seed) live in an immutable :class:`MacsaRunConfig` built by
:func:`configure` from the notebook's ``RunContext`` -- no module globals are
mutated. The static scenario constants and all helpers are exported via
``__all__`` so the notebook can ``from experiments.macsa import *``.
"""
from __future__ import annotations

import contextlib
import io
import math
import time as _t
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from IPython.display import Image, display

from eval_lib import plots as route_plots
from eval_lib.baselines import _run_baseline
from eval_lib.context import DATASETS_DIR
from eval_lib.experiments import compose_experiment_cfg, scoring_cfg
from eval_lib.helpers import as_route_tensor
from connectpt.routes_generator.search.cfg_run import run_bco_from_cfg
from eval_lib.paper import PAPER_DIR, UNIFIED_ADJ, paper_row as _row
from connectpt.routes_generator.objectives import load_unified_objective as _load_objective

_OBJ = _load_objective()
ADJ_OBJECTIVE = _OBJ.adj_objective
ADJ_TARGET = _OBJ.adj_target
CONNECTIVITY_MODE = _OBJ.connectivity_mode

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

# --- runtime run config (immutable; built by configure()) ----------------------

@dataclass(frozen=True)
class MacsaRunConfig:
    """Runtime knobs of the MACSA sweep -- explicit, no module globals."""
    bco_iterations: int
    seed: int
    edit_weights_path: Path
    edit_adj_cond_feats: int = 0

    @property
    def sweep_stem(self) -> str:
        return f"final_macsa_mandl8_alpha_sweep_iter{self.bco_iterations}"

    def summary(self) -> dict:
        return {
            "scenario_dir": str(MACSA_SCENARIO_DIR),
            "alpha_grid": MACSA_ALPHA_GRID,
            "bco_iterations": self.bco_iterations,
            "bco_bees": MACSA_SWEEP_BEES,
            "seed": self.seed,
            "force_cpu": MACSA_FORCE_CPU,
            "our_model": Path(self.edit_weights_path).name,
        }


def configure(ctx, *, smoke=False, seeds=None) -> MacsaRunConfig:
    """Build the immutable MACSA run config from the notebook's RunContext.

    ``smoke`` collapses the sweep to a single BCO iteration; the edit-model
    checkpoint comes from ``ctx.edit_weights_path`` (the suite profile);
    ``seeds`` provides the sweep seed.
    """
    seeds = seeds or [0]
    return MacsaRunConfig(
        bco_iterations=1 if smoke else 100,
        seed=int(seeds[0]),
        edit_weights_path=Path(ctx.edit_weights_path),
        edit_adj_cond_feats=int(ctx.edit_adj_cond_feats),
    )


def macsa_alpha_tag(alpha):
    return f"{float(alpha):.1f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p") or "0"


def macsa_alpha_label(alpha, n_iterations):
    return f"Our NBCO alpha={float(alpha):.1f} (iter={int(n_iterations)})"


def macsa_read_routes_0indexed(path):
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


def macsa_pad_routes(routes, n_routes, max_route_len):
    from ._common import pad_routes_to
    return pad_routes_to(routes, n_routes, max_route_len, strict=True)


def macsa_2d(routes):
    from ._common import route_2d
    return route_2d(routes)


def macsa_build_spec(raw_routes, n_nodes):
    n_routes = int(raw_routes[MACSA_REF_METHOD].shape[0])
    longest = max(int((rt > -1).sum(-1).max().item()) for rt in raw_routes.values())
    return {"city": MACSA_SCENARIO_NAME,
            "n_routes": n_routes,
            "min_route_len": 2,
            "max_route_len": min(int(n_nodes), max(12, longest))}


def macsa_score_routes(method, source, routes, *, seed_routes, tensors, spec,
                        alpha, adj_target, adj_objective):
    cfg = scoring_cfg(MACSA_SCENARIO_NAME, spec, cpu=True, csv=False, alpha=alpha)
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


def macsa_build_our_cfg(spec, *, alpha, adj_target, adj_objective,
                         n_iterations, seed, force_cpu):
    """Our-NBCO cfg for the MACSA case: captured preset + the sweep point.

    The bee composition (5 GNN-rebuild + 5 trim/extend, forced-mutation edit
    bees) and the worse-accept schedule live in
    ``cfg/experiments/macsa/our_nbco_mandl8.yaml``; only the sweep point and
    run identity are applied here.
    """
    return compose_experiment_cfg(
        "macsa/our_nbco_mandl8", bounds=spec, seed=int(seed),
        cpu=bool(force_cpu), alpha=alpha, n_iterations=int(n_iterations),
        adj=dict(UNIFIED_ADJ,
                 adjustment_degree_target=float(adj_target),
                 adjustment_degree_objective=str(adj_objective)))


def macsa_run_our_nbco(seed_routes, *, tensors, spec, alpha, adj_target,
                        adj_objective, n_iterations, seed, force_cpu,
                        edit_weights_path, edit_adj_cond_feats=0):
    cfg = macsa_build_our_cfg(spec, alpha=alpha, adj_target=adj_target,
                               adj_objective=adj_objective,
                               n_iterations=n_iterations,
                               seed=seed, force_cpu=force_cpu)
    t0 = _t.perf_counter()
    _run_name, _metrics, _unserved, routes, _mutation_counts = run_bco_from_cfg(
        cfg, seed_routes, tensors, run_name_scope=f"{MACSA_SCENARIO_NAME}_",
        edit_weights_path=edit_weights_path,
        edit_n_adjustment_cond_feats=int(edit_adj_cond_feats))
    return as_route_tensor(routes), _t.perf_counter() - t0


def macsa_metric_subtitle(row):
    if row is None:
        return ""
    return (f"ATT={float(row['ATT']):.2f}  WMC={float(row['WMC']):.2f}\n"
            f"RTT={float(row['RTT']):.0f}  cost={float(row['cost']):.3f}  "
            f"adj={float(row['adj_vs_seed']):.3f}")


def macsa_relabel_nodes_1indexed(ax):
    for txt in ax.texts:
        value = txt.get_text()
        if value.isdigit():
            txt.set_text(str(int(value) + 1))


def macsa_draw_grid(*, routes, rows_by_method, coords, street_adj, demand,
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
    ref_routes_2d = macsa_2d(ref_routes) if ref_routes is not None else None
    for ax, panel in zip(axes.flat, panels):
        if panel == "__demand__":
            route_plots.plot_demand_graph(ax, demand, coords, street_adj,
                                          title="OD demand",
                                          subtitle="edge color/width = demand")
            macsa_relabel_nodes_1indexed(ax)
            continue
        panel_routes = macsa_2d(routes[panel])
        subtitle = macsa_metric_subtitle(rows_by_method.get(panel))
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
        macsa_relabel_nodes_1indexed(ax)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return fig


def macsa_save_fig(fig, stem, suffix, *, prefix):
    path = PAPER_DIR / f"{prefix}{stem}_{suffix}.png"
    fig.savefig(path, dpi=MACSA_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper] figure -> {path}")
    return path


def macsa_display_image(path):
    try:
        display(Image(filename=str(path)))
    except Exception:
        print(path)


def macsa_upsert_row(rows, row):
    return [r for r in rows if str(r.get("method")) != str(row.get("method"))] + [row]


def macsa_select_best_sweep_row(sweep_df, macsa_row):
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
__all__ = [n for n in dict(globals()) if n.startswith("MACSA_") or n.startswith("macsa_")]
__all__ += ["configure", "MacsaRunConfig"]
