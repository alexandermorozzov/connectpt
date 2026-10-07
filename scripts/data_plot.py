"""Draw a route network over its street graph.

    python scripts/data_plot.py --instance mandl
    python scripts/data_plot.py artifacts/results/validation/validation_mandl_ea_seed0_routes.pt
    python scripts/data_plot.py dump_ea.pt dump_nea_edit.pt --out compare.png

Writes a PNG of one or more route sets over the street graph. A run dump carries
its own coords / street_adj; with no route file at all the instance's initial
network is drawn.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402

from connectpt.routes_generator.run.context import resolve_under_root  # noqa: E402
from connectpt.routes_generator.data import INSTANCES_DIR, load_instance  # noqa: E402
from connectpt.routes_generator.reports import plot_routes_grid  # noqa: E402


def load_routes(path: Path):
    """-> ({label: routes}, coords | None, street_adj | None)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {path.stem: payload}, None, None
    routes = payload.get("routes", payload)
    if not isinstance(routes, dict):
        routes = {path.stem: routes}
    return routes, payload.get("coords"), payload.get("street_adj")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("routes", nargs="*", default=None,
                        help="route dumps (.pt); several are merged into one "
                             "figure. Omit to draw the instance's own initial "
                             "network")
    parser.add_argument("--instance", default=None,
                        help="instance name: the street data to draw on, and the "
                             "source of the network when no route file is given")
    parser.add_argument("--out", default=None, help="output PNG (default: next to the input)")
    parser.add_argument("--demand", action="store_true", help="prepend an OD-demand panel")
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if not args.routes:
        if args.instance is None:
            raise SystemExit("pass a route file, or --instance to draw its "
                             "initial network")
        inst = load_instance(args.instance)
        path = INSTANCES_DIR / args.instance / "Routes.txt"
        route_sets = {f"{args.instance} (initial)": inst.init_routes}
        coords, street_adj = inst.coords, inst.street_adj
        demand = inst.tensors["demand"] if args.demand else None
    else:
        paths = [resolve_under_root(r) for r in args.routes]
        path = paths[0]
        route_sets, coords, street_adj = {}, None, None
        for p in paths:
            sets, c, adj = load_routes(p)
            # setdefault keeps a single Initial: every dump of one instance
            # carries the same one, and the other labels name their method.
            for label, routes in sets.items():
                route_sets.setdefault(label, routes)
            coords = c if coords is None else coords
            street_adj = adj if street_adj is None else street_adj
        demand = None
        if coords is None or street_adj is None or args.demand:
            if args.instance is None:
                raise SystemExit(
                    f"{path.name} carries no street data -- pass --instance")
            tensors = load_instance(args.instance).tensors
            coords = tensors["node_locs"] if coords is None else coords
            street_adj = tensors["street_adj"] if street_adj is None else street_adj
            demand = tensors["demand"] if args.demand else None

    fig = plot_routes_grid(route_sets, coords, street_adj, ncols=args.ncols,
                           demand=demand, title=args.instance or path.stem)
    out = Path(args.out) if args.out else path.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"{len(route_sets)} route set(s) -> {out}")


if __name__ == "__main__":
    main()
