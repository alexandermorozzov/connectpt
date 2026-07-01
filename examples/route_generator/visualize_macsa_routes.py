# -*- coding: utf-8 -*-
"""Score and visualize the Mandl-8 MACSA Table-B route sets.

The route files live in ``datasets/MACSA_data/mandl_8``:
``routes_original.txt`` plus one ``routes_<method>.txt`` per method from the
MACSA supplementary table.  This script evaluates them with the same unified
paper/E1u metric plumbing used by ``paper_combined.ipynb``, saves the table and
route dump, and renders plain and diff-vs-original figures.

Optionally, ``--run-our-target`` runs Our NBCO from the original network with
``adjustment_degree_target`` set to the actual adjustment reached by the MACSA
route set.  ``--our-alpha`` controls the RTT/WMC trade-off:
``alpha*RTT + (1-alpha)*WMC + adj``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from connectpt.routes_generator.citygraph_dataset import load_macsa_tensors  # noqa: E402

from eval_lib import plots as route_plots  # noqa: E402
from eval_lib.baselines import _run_baseline  # noqa: E402
from eval_lib.context import ARTIFACTS_DIR, DATASETS_DIR  # noqa: E402
from eval_lib.helpers import as_route_tensor, run_bco  # noqa: E402
from connectpt.routes_generator.search.bco_config import compose_bco_cfg as build_bco_cfg  # noqa: E402
from eval_lib.paper import (  # noqa: E402
    UNIFIED_ADJ,
    adj_vs_init,
    bco_cfg_set,
    eval_routes_cfg,
    paper_row,
    save_paper_routes,
    save_paper_table,
    set_cfg_value,
    unify_weights,
)
from connectpt.routes_generator.objectives import load_unified_objective  # noqa: E402
CONNECTIVITY_MODE = load_unified_objective().connectivity_mode


SCENARIO_NAME = "mandl_8"
SCENARIO_DIR = DATASETS_DIR / "MACSA_data" / SCENARIO_NAME
PAPER_DIR = ARTIFACTS_DIR / "paper_results"
DEFAULT_STEM = "final_macsa_mandl8_tableb"

# Original first, then the methods in Table B reading order.
METHODS = ["original", "rga", "as", "ga", "ras", "mmas", "ma", "macsa"]
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
OUR_METHOD_LEGACY = "Our NBCO (target=MACSA adj)"


def alpha_tag(alpha: float) -> str:
    text = f"{float(alpha):g}".replace("-", "m").replace(".", "p")
    return text or "0"


def our_method_label(alpha: float, adj_objective: str, adj_target_label: str) -> str:
    return (
        f"Our NBCO (alpha={float(alpha):g}, "
        f"adj={adj_objective}, target={adj_target_label})"
    )


def alpha_weights(alpha: float) -> dict:
    alpha = float(alpha)
    return {
        "demand_time_weight": 0.0,
        "route_time_weight": alpha,
        "median_connectivity_weight": 1.0 - alpha,
    }


def set_cfg_alpha_weights(cfg, alpha: float):
    for key, value in alpha_weights(alpha).items():
        set_cfg_value(cfg, f"experiment.cost_function.kwargs.{key}", float(value))
    return cfg


def read_routes_0indexed(path: Path) -> torch.Tensor:
    """Read a ``routes_<method>.txt`` file with 0-indexed node ids."""
    rows: list[list[int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            rows.append([int(t) for t in line.split()])
    if not rows:
        raise ValueError(f"No routes found in {path}")
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), -1, dtype=torch.long)
    for i, route in enumerate(rows):
        out[i, : len(route)] = torch.tensor(route, dtype=torch.long)
    return out


def pad_routes(routes: torch.Tensor, n_routes: int, max_route_len: int) -> torch.Tensor:
    routes = as_route_tensor(routes).long()
    if routes.ndim == 2:
        routes = routes.unsqueeze(0)
    if routes.shape[1] != n_routes:
        raise ValueError(f"Expected {n_routes} routes, got shape {tuple(routes.shape)}")
    if routes.shape[-1] > max_route_len:
        raise ValueError(
            f"Route width {routes.shape[-1]} exceeds max_route_len={max_route_len}"
        )
    if routes.shape[-1] < max_route_len:
        routes = torch.nn.functional.pad(
            routes, (0, max_route_len - routes.shape[-1]), value=-1
        )
    return routes


def route_stats(routes: torch.Tensor) -> str:
    routes = as_route_tensor(routes)
    if routes.ndim == 3:
        routes = routes[0]
    lens = (routes > -1).sum(-1)
    nonempty = lens[lens > 0]
    stops = {int(n) for row in routes for n in row.tolist() if n >= 0}
    return (
        f"{int((lens > 0).sum())} routes | {len(stops)} stops covered | "
        f"len {int(nonempty.min())}-{int(nonempty.max())} "
        f"(mean {float(nonempty.float().mean()):.1f})"
    )


def build_spec(method_routes: dict[str, torch.Tensor], n_nodes: int) -> dict:
    n_routes = int(method_routes[REF_METHOD].shape[0])
    longest = max(int((rt > -1).sum(-1).max().item()) for rt in method_routes.values())
    return {
        "city": SCENARIO_NAME,
        "n_routes": n_routes,
        "min_route_len": 2,
        "max_route_len": min(int(n_nodes), max(12, longest)),
    }


def score_fixed_routes(
    *,
    method: str,
    source: str,
    routes: torch.Tensor,
    seed_routes: torch.Tensor,
    tensors: dict,
    spec: dict,
    adj_target: float,
    adj_objective: str = "target",
    alpha: float = 0.5,
) -> tuple[dict, torch.Tensor]:
    cfg = unify_weights(eval_routes_cfg(SCENARIO_NAME, spec))
    set_cfg_alpha_weights(cfg, alpha)
    set_cfg_value(cfg, "experiment.cpu", True)
    set_cfg_value(cfg, "eval.csv", False)
    set_cfg_value(cfg, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
    adj_kwargs = dict(
        UNIFIED_ADJ,
        adjustment_degree_target=float(adj_target),
        adjustment_degree_objective=str(adj_objective),
    )
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        _run_name, metrics, _unserved, scored_routes = _run_baseline(
            None,
            cfg,
            routes,
            f"{SCENARIO_NAME}_{method.lower().replace(' ', '_')}_score_",
            {},
            tensors=tensors,
            use_weighted_connectivity=True,
            connectivity_mode=CONNECTIVITY_MODE,
            adjustment_seed_routes=seed_routes,
            **adj_kwargs,
        )
    row = paper_row(
        SCENARIO_NAME,
        method,
        source,
        metrics,
        scored_routes,
        seed_routes,
        duration_s=time.perf_counter() - t0,
    )
    row["eval_adj_target"] = float(adj_target)
    row["eval_adj_objective"] = str(adj_objective)
    row["alpha"] = float(alpha)
    row["objective"] = f"alpha*RTT + (1-alpha)*WMC + adj({adj_objective})"
    return row, as_route_tensor(scored_routes)


def build_our_nbco_cfg(
    *,
    spec: dict,
    adj_target: float,
    n_iterations: int,
    n_bees: int,
    seed: int,
    force_cpu: bool,
    alpha: float,
    adj_objective: str,
):
    """Mirror paper_combined's Our NBCO: GNN rebuild bees + trim/extend bees."""
    n_bees = int(n_bees)
    rebuild_bees = max(1, n_bees // 2)
    trim_extend_bees = n_bees - rebuild_bees
    cfg = build_bco_cfg(
        run_name=f"{SCENARIO_NAME}_macsa_target_our_nbco_gnn_rebuild_trimext",
        n_routes=spec["n_routes"],
        min_route_len=spec["min_route_len"],
        max_route_len=spec["max_route_len"],
        use_neural_bees=True,
        n_bees=n_bees,
        n_type1_bees=rebuild_bees,
        n_type2_bees=0,
        n_type4_bees=0,
        n_type5_bees=trim_extend_bees,
        n_type6_bees=0,
        n_type7_bees=0,
        force_cpu=bool(force_cpu),
        connectivity_mode=CONNECTIVITY_MODE,
        worse_accept_temperature=0.02,
        worse_accept_decay=0.985,
        worse_accept_min_temperature=0.001,
        worse_selection_temperature=0.02,
        worse_selection_decay=0.985,
        worse_selection_uniform_mix=0.10,
        worse_selection_elite_count=2,
        **alpha_weights(alpha),
    )
    bco_cfg_set(
        cfg,
        n_iterations=int(n_iterations),
        type4_allow_halt=False,
        type5_allow_halt=False,
        type6_allow_halt=False,
        type7_allow_halt=False,
        **dict(
            UNIFIED_ADJ,
            adjustment_degree_target=float(adj_target),
            adjustment_degree_objective=str(adj_objective),
        ),
    )
    set_cfg_alpha_weights(cfg, alpha)
    set_cfg_value(cfg, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
    set_cfg_value(cfg, "eval.csv", False)
    set_cfg_value(cfg, "experiment.seed", int(seed))
    return cfg


def load_cached_routes(path: Path, method: str) -> torch.Tensor | None:
    if not path.exists():
        return None
    try:
        dump = torch.load(path, map_location="cpu", weights_only=False)
        routes = dump.get("routes", {})
        if method in routes:
            print(f"[macsa] loaded cached {method!r} from {path}")
            return as_route_tensor(routes[method])
    except Exception as exc:
        print(f"[macsa] cache load failed ({exc}); rerunning if requested")
    return None


def load_cached_routes_for_alpha(path: Path, method: str, alpha: float) -> torch.Tensor | None:
    cached = load_cached_routes(path, method)
    if cached is not None:
        return cached
    if abs(float(alpha) - 0.5) < 1e-9:
        cached = load_cached_routes(path, OUR_METHOD_LEGACY)
        if cached is not None:
            print(f"[macsa] treating legacy Our NBCO cache as alpha=0.5")
            return cached
    return None


def load_cached_row(path: Path, method: str) -> dict | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "method" not in df:
            return None
        matches = df[df["method"].astype(str).eq(method)]
        if not matches.empty:
            return dict(matches.iloc[0])
    except Exception as exc:
        print(f"[macsa] cached row load failed ({exc}); row will be rescored")
    return None


def load_cached_row_for_alpha(path: Path, method: str, alpha: float) -> dict | None:
    cached = load_cached_row(path, method)
    if cached is not None:
        return cached
    if abs(float(alpha) - 0.5) < 1e-9:
        return load_cached_row(path, OUR_METHOD_LEGACY)
    return None


def run_our_nbco(
    *,
    seed_routes: torch.Tensor,
    tensors: dict,
    spec: dict,
    adj_target: float,
    n_iterations: int,
    n_bees: int,
    seed: int,
    force_cpu: bool,
    alpha: float,
    method: str,
    adj_objective: str,
) -> tuple[dict, torch.Tensor]:
    cfg = build_our_nbco_cfg(
        spec=spec,
        adj_target=adj_target,
        n_iterations=n_iterations,
        n_bees=n_bees,
        seed=seed,
        force_cpu=force_cpu,
        alpha=alpha,
        adj_objective=adj_objective,
    )
    t0 = time.perf_counter()
    _run_name, metrics, _unserved, routes, _mutation_counts = run_bco(
        cfg,
        seed_routes,
        tensors=tensors,
        run_name_scope=f"{SCENARIO_NAME}_",
    )
    row = paper_row(
        SCENARIO_NAME,
        method,
        "our_nbco_macsa_target",
        metrics,
        routes,
        seed_routes,
        duration_s=time.perf_counter() - t0,
    )
    row["eval_adj_target"] = float(adj_target)
    row["eval_adj_objective"] = str(adj_objective)
    row["alpha"] = float(alpha)
    row["objective"] = f"alpha*RTT + (1-alpha)*WMC + adj({adj_objective})"
    return row, as_route_tensor(routes)


def metric_subtitle(row: dict | None) -> str:
    if not row:
        return ""
    return (
        f"ATT={float(row['ATT']):.2f}  WMC={float(row['WMC']):.2f}\n"
        f"RTT={float(row['RTT']):.0f}  cost={float(row['cost']):.3f}  "
        f"adj={float(row['adj_vs_seed']):.3f}"
    )


def relabel_nodes_1indexed(ax) -> None:
    for txt in ax.texts:
        value = txt.get_text()
        if value.isdigit():
            txt.set_text(str(int(value) + 1))


def draw_grid(
    *,
    diff: bool,
    routes: dict[str, torch.Tensor],
    rows_by_method: dict[str, dict],
    coords,
    street_adj,
    demand,
    ref_key: str,
    ncol: int,
    demand_top_frac: float | None,
    node_size: float,
    show_node_labels: bool,
):
    panels = ["__demand__", ref_key] + [m for m in routes if m != ref_key]
    ncol = max(1, int(ncol))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(5.8 * ncol, 5.8 * nrow),
        squeeze=False,
        constrained_layout=True,
    )
    ref_routes = as_route_tensor(routes[ref_key])
    if ref_routes.ndim == 3:
        ref_routes = ref_routes[0]

    for ax, panel in zip(axes.flat, panels):
        if panel == "__demand__":
            route_plots.plot_demand_graph(
                ax,
                demand,
                coords,
                street_adj,
                title="OD demand",
                subtitle="edge color/width = demand",
                top_frac=demand_top_frac,
            )
            relabel_nodes_1indexed(ax)
            continue

        panel_routes = as_route_tensor(routes[panel])
        if panel_routes.ndim == 3:
            panel_routes = panel_routes[0]
        subtitle = metric_subtitle(rows_by_method.get(panel))
        if diff and panel != ref_key:
            route_plots.plot_route_diff(
                ax,
                panel_routes,
                ref_routes,
                coords,
                street_adj,
                title=f"{panel} vs original",
                subtitle=subtitle,
                palette="tab20",
                with_overlap_curves=True,
                show_node_labels=show_node_labels,
                node_size=node_size,
            )
        else:
            route_plots.plot_plain_route_set(
                ax,
                panel_routes,
                coords,
                street_adj,
                title=panel,
                subtitle=subtitle,
                palette="tab20",
                with_overlap_curves=True,
                show_node_labels=show_node_labels,
                node_size=node_size,
            )
        relabel_nodes_1indexed(ax)

    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    kind = "diff vs original" if diff else "plain route sets"
    fig.suptitle(f"Mandl-8 MACSA Table B ({kind})", fontsize=15, fontweight="bold")
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--out-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--run-our-target", action="store_true")
    parser.add_argument("--force-our-target", action="store_true")
    parser.add_argument(
        "--our-alpha",
        dest="our_alphas",
        action="append",
        type=float,
        default=None,
        help="RTT weight for Our NBCO target run; repeatable. Default: 0.5.",
    )
    parser.add_argument(
        "--our-adj-objective",
        choices=("raw", "target", "cap", "cap_sq"),
        default="target",
        help="Adjustment penalty objective for Our NBCO target run.",
    )
    parser.add_argument(
        "--our-adj-target",
        type=float,
        default=None,
        help="Adjustment target for Our NBCO. Default: actual MACSA adj.",
    )
    parser.add_argument("--bco-iterations", type=int, default=500)
    parser.add_argument("--bco-bees", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ncol", type=int, default=5)
    parser.add_argument("--demand-top-frac", type=float, default=None)
    parser.add_argument("--hide-node-labels", action="store_true")
    parser.add_argument("--node-size", type=float, default=70.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    our_alphas = args.our_alphas if args.our_alphas is not None else [0.5]
    seen_alphas = set()
    our_alphas = [
        float(alpha)
        for alpha in our_alphas
        if not (float(alpha) in seen_alphas or seen_alphas.add(float(alpha)))
    ]
    for alpha in our_alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"--our-alpha must be in [0, 1], got {alpha}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / f"{args.stem}.csv"
    routes_dump_path = args.out_dir / f"{args.stem}_routes.pt"

    tensors = load_macsa_tensors(args.scenario_dir)
    coords = tensors["node_locs"]
    street_adj = tensors["street_adj"]
    demand = tensors["demand"]

    raw_routes = {
        METHOD_TITLE[method]: read_routes_0indexed(
            args.scenario_dir / f"routes_{method}.txt"
        )
        for method in METHODS
    }
    spec = build_spec(raw_routes, int(coords.shape[0]))
    routes = {
        method: pad_routes(rt, spec["n_routes"], spec["max_route_len"])
        for method, rt in raw_routes.items()
    }
    seed_routes = routes[REF_METHOD]
    macsa_adj = adj_vs_init(routes[METHOD_TITLE["macsa"]], seed_routes)
    our_adj_target = macsa_adj if args.our_adj_target is None else float(args.our_adj_target)
    our_adj_target_label = (
        "MACSA adj" if args.our_adj_target is None else f"{our_adj_target:g}"
    )
    our_adj_objective = str(args.our_adj_objective)
    print(f"[macsa] MACSA adj vs original = {macsa_adj:.6f}")
    if args.run_our_target:
        print(
            f"[macsa] Our NBCO adj objective={our_adj_objective} "
            f"target={our_adj_target:.6f}"
        )
    print(
        "[macsa] spec="
        f"n_routes={spec['n_routes']} min_len={spec['min_route_len']} "
        f"max_len={spec['max_route_len']}"
    )

    rows: list[dict] = []
    rows_by_method: dict[str, dict] = {}
    default_adj_target = float(UNIFIED_ADJ["adjustment_degree_target"])
    fixed_adj_target = (
        our_adj_target
        if args.run_our_target and args.our_adj_target is not None
        else default_adj_target
    )
    fixed_adj_objective = (
        our_adj_objective
        if args.run_our_target and args.our_adj_target is not None
        else str(UNIFIED_ADJ["adjustment_degree_objective"])
    )
    for method, rt in routes.items():
        row, scored_routes = score_fixed_routes(
            method=method,
            source="macsa_table_b",
            routes=rt,
            seed_routes=seed_routes,
            tensors=tensors,
            spec=spec,
            adj_target=fixed_adj_target,
            adj_objective=fixed_adj_objective,
            alpha=0.5,
        )
        routes[method] = scored_routes
        rows.append(row)
        rows_by_method[method] = row
        print(
            f"  {method:20} ATT={row['ATT']:.2f} RTT={row['RTT']:.0f} "
            f"WMC={row['WMC']:.2f} adj={row['adj_vs_seed']:.3f} "
            f"cost={row['cost']:.3f}"
        )

    if args.run_our_target:
        for alpha in our_alphas:
            method = our_method_label(alpha, our_adj_objective, our_adj_target_label)
            cached = None if args.force_our_target else load_cached_routes_for_alpha(
                routes_dump_path, method, alpha
            )
            if cached is not None:
                cached_row = load_cached_row_for_alpha(table_path, method, alpha)
                cached = pad_routes(cached, spec["n_routes"], spec["max_route_len"])
                row, scored_routes = score_fixed_routes(
                    method=method,
                    source="our_nbco_macsa_target_cached",
                    routes=cached,
                    seed_routes=seed_routes,
                    tensors=tensors,
                    spec=spec,
                    adj_target=our_adj_target,
                    adj_objective=our_adj_objective,
                    alpha=alpha,
                )
                if cached_row is not None:
                    row["duration_s"] = cached_row.get("duration_s", row["duration_s"])
                    row["source"] = cached_row.get("source", row["source"])
            else:
                run_row, scored_routes = run_our_nbco(
                    seed_routes=seed_routes,
                    tensors=tensors,
                    spec=spec,
                    adj_target=our_adj_target,
                    n_iterations=int(args.bco_iterations),
                    n_bees=int(args.bco_bees),
                    seed=int(args.seed),
                    force_cpu=bool(args.cpu),
                    alpha=alpha,
                    method=method,
                    adj_objective=our_adj_objective,
                )
                row, scored_routes = score_fixed_routes(
                    method=method,
                    source=run_row.get("source", "our_nbco_macsa_target"),
                    routes=scored_routes,
                    seed_routes=seed_routes,
                    tensors=tensors,
                    spec=spec,
                    adj_target=our_adj_target,
                    adj_objective=our_adj_objective,
                    alpha=alpha,
                )
                row["duration_s"] = run_row.get("duration_s", row["duration_s"])
            routes[method] = scored_routes
            rows.append(row)
            rows_by_method[method] = row
            print(
                f"  {method:20} ATT={row['ATT']:.2f} RTT={row['RTT']:.0f} "
                f"WMC={row['WMC']:.2f} adj={row['adj_vs_seed']:.3f} "
                f"cost={row['cost']:.3f} target={our_adj_target:.3f}"
            )
    else:
        print("[macsa] Our NBCO target run skipped; pass --run-our-target to add it.")

    df = pd.DataFrame(rows).round(6)
    df["macsa_adj_target"] = float(macsa_adj)
    save_paper_table(df, args.stem)
    save_paper_routes(
        args.stem,
        routes,
        coords,
        street_adj,
        meta={
            "scenario": SCENARIO_NAME,
            "scenario_dir": str(args.scenario_dir),
            "ref_method": REF_METHOD,
            "macsa_adj_target": float(macsa_adj),
            "default_eval_adj_target": default_adj_target,
            "fixed_eval_adj_target": fixed_adj_target,
            "fixed_eval_adj_objective": fixed_adj_objective,
            "our_alphas": our_alphas if args.run_our_target else [],
            "our_adj_target": float(our_adj_target) if args.run_our_target else None,
            "our_adj_objective": our_adj_objective if args.run_our_target else None,
        },
    )

    common_plot_kwargs = dict(
        routes=routes,
        rows_by_method=rows_by_method,
        coords=coords,
        street_adj=street_adj,
        demand=demand,
        ref_key=REF_METHOD,
        ncol=int(args.ncol),
        demand_top_frac=args.demand_top_frac,
        node_size=float(args.node_size),
        show_node_labels=not args.hide_node_labels,
    )
    plain_fig = draw_grid(diff=False, **common_plot_kwargs)
    plain_path = args.out_dir / f"{args.stem}_viz_plain.png"
    plain_fig.savefig(plain_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(plain_fig)

    diff_fig = draw_grid(diff=True, **common_plot_kwargs)
    diff_path = args.out_dir / f"{args.stem}_viz_diff.png"
    diff_fig.savefig(diff_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(diff_fig)

    alpha_paths = []
    if args.run_our_target:
        fixed_methods = [METHOD_TITLE[method] for method in METHODS]
        for alpha in our_alphas:
            method = our_method_label(alpha, our_adj_objective, our_adj_target_label)
            if method not in routes:
                continue
            alpha_rows_by_method = {}
            alpha_routes = {name: routes[name] for name in fixed_methods}
            for fixed_method in fixed_methods:
                if (
                    abs(alpha - 0.5) < 1e-9
                    and abs(float(fixed_adj_target) - float(rows_by_method[fixed_method]["eval_adj_target"])) < 1e-9
                    and str(fixed_adj_objective) == str(rows_by_method[fixed_method]["eval_adj_objective"])
                ):
                    alpha_rows_by_method[fixed_method] = rows_by_method[fixed_method]
                else:
                    fixed_row, _ = score_fixed_routes(
                        method=fixed_method,
                        source="macsa_table_b",
                        routes=routes[fixed_method],
                        seed_routes=seed_routes,
                        tensors=tensors,
                        spec=spec,
                        adj_target=fixed_adj_target,
                        adj_objective=fixed_adj_objective,
                        alpha=alpha,
                    )
                    alpha_rows_by_method[fixed_method] = fixed_row
            alpha_routes[method] = routes[method]
            alpha_rows_by_method[method] = rows_by_method[method]
            tag = alpha_tag(alpha)
            alpha_kwargs = dict(
                routes=alpha_routes,
                rows_by_method=alpha_rows_by_method,
                coords=coords,
                street_adj=street_adj,
                demand=demand,
                ref_key=REF_METHOD,
                ncol=int(args.ncol),
                demand_top_frac=args.demand_top_frac,
                node_size=float(args.node_size),
                show_node_labels=not args.hide_node_labels,
            )
            alpha_plain = draw_grid(diff=False, **alpha_kwargs)
            alpha_plain_path = args.out_dir / f"{args.stem}_alpha{tag}_viz_plain.png"
            alpha_plain.savefig(alpha_plain_path, dpi=int(args.dpi), bbox_inches="tight")
            plt.close(alpha_plain)
            alpha_diff = draw_grid(diff=True, **alpha_kwargs)
            alpha_diff_path = args.out_dir / f"{args.stem}_alpha{tag}_viz_diff.png"
            alpha_diff.savefig(alpha_diff_path, dpi=int(args.dpi), bbox_inches="tight")
            plt.close(alpha_diff)
            alpha_paths.extend([alpha_plain_path, alpha_diff_path])

    print(f"[macsa] table={table_path}")
    print(f"[macsa] routes={routes_dump_path}")
    print(f"[macsa] plain={plain_path}")
    print(f"[macsa] diff={diff_path}")
    for path in alpha_paths:
        print(f"[macsa] alpha_viz={path}")


if __name__ == "__main__":
    main()
