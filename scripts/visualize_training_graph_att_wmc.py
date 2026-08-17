"""Visualize per-node ATT and WMC on a saved training graph.

The script uses an existing training dataset example: a raw graph from
``raw_graphs_subset.pkl`` and its saved LC/curriculum route tensor from the
matching ``graph_XXXX`` folder. It produces two PNG files:

* ``training_graph_XXXX_att.png`` -- per-origin served-demand ATT.
* ``training_graph_XXXX_wmc.png`` -- per-origin demand-weighted connectivity.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from connectpt.routes_generator.citygraph_dataset import STOP_KEY
from connectpt.routes_generator.improvement_learning import (
    load_raw_graphs_and_lc_routes,
)
from connectpt.routes_generator.transit_time_estimator import (
    MyCostModule,
    RouteGenBatchState,
    _finite_time_diameter,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    REPO_ROOT / "datasets" / "lc_realistic_curriculum_n700_n50_r12_len8_15_v5"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "training_graph_metric_viz"


def _node_values(graph, routes):
    cost_obj = MyCostModule(
        mean_stop_time_s=0,
        avg_transfer_wait_time_s=300,
        symmetric_routes=True,
        use_weighted_connectivity=True,
        connectivity_mode="median_weighted",
    )
    state = RouteGenBatchState(
        graph,
        cost_obj,
        n_routes_to_plan=int(routes.shape[0]),
        min_route_len=2,
        max_route_len=int(routes.shape[1]),
    )
    state.add_new_routes(routes)
    result = cost_obj(state)
    metrics = result.get_metrics()

    demand = state.demand[0]
    transit = state.transit_times[0]
    has_path = state.has_path[0] & torch.isfinite(transit)

    served_weights = torch.where(has_path, demand, torch.zeros_like(demand))
    served_times = torch.where(has_path, transit, torch.zeros_like(transit))
    served_demand = served_weights.sum(dim=1)
    att_num = (served_weights * served_times).sum(dim=1)
    att_node = torch.where(
        served_demand > 1e-9,
        att_num / served_demand.clamp_min(1e-9),
        torch.full_like(att_num, float("nan")),
    )

    all_pairs = state.transit_times.clone()
    n_nodes = all_pairs.shape[-1]
    eye = torch.eye(n_nodes, device=all_pairs.device, dtype=torch.bool)[None]
    max_t = _finite_time_diameter(state.drive_times)
    penalty = (2.0 * max_t).view(-1, 1, 1)
    unreachable = (~state.has_path) | (~all_pairs.isfinite())
    conn_vals = torch.where(unreachable, penalty.expand_as(all_pairs), all_pairs)
    conn_vals = conn_vals.masked_fill(eye, float("nan"))
    wmc_node = cost_obj._demand_weighted_node_times(
        conn_vals, state.demand, dim=2
    )[0]

    return {
        "att_min": (att_node / 60).detach().cpu(),
        "wmc_min": (wmc_node / 60).detach().cpu(),
        "served_demand": served_demand.detach().cpu(),
        "out_demand": demand.sum(dim=1).detach().cpu(),
        "global_att_min": float(metrics["ATT"].detach().cpu().item()),
        "global_wmc_min": float(
            metrics["median_connectivity_weighted"].detach().cpu().item()
        ),
        "global_rtt_min": float(metrics["RTT"].detach().cpu().item()),
        "d_un_pct": float(metrics["$d_{un}$"].detach().cpu().item()),
    }


def _draw_base(ax, pos, street_adj, routes):
    finite_edges = (street_adj > 0) & torch.isfinite(street_adj)
    edge_idx = torch.stack(torch.where(torch.triu(finite_edges, diagonal=1))).T
    for src, dst in edge_idx.tolist():
        xy = pos[[src, dst]]
        ax.plot(xy[:, 0], xy[:, 1], color="#c8ccd2", lw=0.7, zorder=1)

    cmap = plt.get_cmap("tab20")
    for ridx, route in enumerate(routes):
        nodes = route[route >= 0].tolist()
        if len(nodes) < 2:
            continue
        xy = pos[nodes]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=cmap(ridx % cmap.N),
            lw=2.0,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )


def _plot_metric(path, title, values, pos, street_adj, routes, summary):
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    _draw_base(ax, pos, street_adj, routes)

    values_np = values.numpy().astype(float)
    finite = np.isfinite(values_np)
    if finite.any():
        scatter = ax.scatter(
            pos[finite, 0],
            pos[finite, 1],
            c=values_np[finite],
            cmap="viridis",
            s=135,
            edgecolors="#111827",
            linewidths=0.65,
            zorder=4,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
        cbar.set_label("minutes")
    if (~finite).any():
        ax.scatter(
            pos[~finite, 0],
            pos[~finite, 1],
            color="#f8fafc",
            s=135,
            edgecolors="#111827",
            linewidths=0.65,
            zorder=4,
        )

    for idx, (x, y) in enumerate(pos):
        ax.text(
            x,
            y,
            str(idx),
            ha="center",
            va="center",
            fontsize=6.5,
            color="white" if finite[idx] else "#111827",
            fontweight="bold",
            zorder=5,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.text(
        0.01,
        0.01,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#111827",
        bbox=dict(facecolor="white", edgecolor="#d1d5db", alpha=0.88, pad=5),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_csv(path, pos, values):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["node", "x", "y", "ATT_i_min", "WMC_i_min", "served_demand", "out_demand"]
        )
        for idx in range(pos.shape[0]):
            writer.writerow(
                [
                    idx,
                    float(pos[idx, 0]),
                    float(pos[idx, 1]),
                    float(values["att_min"][idx]),
                    float(values["wmc_min"][idx]),
                    float(values["served_demand"][idx]),
                    float(values["out_demand"][idx]),
                ]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--graph-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    raw_graphs_path = args.dataset / "raw_graphs_subset.pkl"
    graphs, seed_routes = load_raw_graphs_and_lc_routes(raw_graphs_path, args.dataset)
    graph = graphs[args.graph_index]
    routes = seed_routes[args.graph_index].long()

    pos = graph[STOP_KEY].pos.detach().cpu().numpy()
    street_adj = graph.street_adj.detach().cpu()
    values = _node_values(graph, routes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"training_graph_{args.graph_index:04d}"
    summary = (
        f"dataset: {args.dataset.name}, graph {args.graph_index}\n"
        f"routes: {routes.shape[0]}, max_len: {routes.shape[1]}\n"
        f"global ATT={values['global_att_min']:.2f} min, "
        f"WMC={values['global_wmc_min']:.2f} min, "
        f"RTT={values['global_rtt_min']:.1f} min, d_un={values['d_un_pct']:.1f}%"
    )

    att_path = args.out_dir / f"{stem}_att.png"
    wmc_path = args.out_dir / f"{stem}_wmc.png"
    csv_path = args.out_dir / f"{stem}_node_metrics.csv"
    routes_path = args.out_dir / f"{stem}_routes.pt"

    _plot_metric(
        att_path,
        f"Training graph {args.graph_index}: per-origin ATT",
        values["att_min"],
        pos,
        street_adj,
        routes,
        summary,
    )
    _plot_metric(
        wmc_path,
        f"Training graph {args.graph_index}: per-origin WMC",
        values["wmc_min"],
        pos,
        street_adj,
        routes,
        summary,
    )
    _write_csv(csv_path, pos, values)
    torch.save(routes.detach().cpu().clone(), routes_path)

    print(att_path)
    print(wmc_path)
    print(csv_path)
    print(routes_path)


if __name__ == "__main__":
    main()
