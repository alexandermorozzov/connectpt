"""Render saved route dumps as case-study map figures -- no search, no Jupyter.

Takes route sets that already exist on disk (whatever produced them), scores each
one against the city's seed network under the unified objective, and renders the
same geo figure the case-study run produces: initial | changed | changed route
slots | route-wise adjustment gradient, over a map basemap.

The dumps are the ONLY input -- this never runs a search, so it is the way to
re-draw (or newly draw) figures for route sets carried over from an earlier run.

    python scripts/render_route_dumps.py --dumps "artifacts/paper_final/ekb final/new routes"
    python scripts/render_route_dumps.py --dumps DIR --alpha 0.5 --adj-target 0.2
    python scripts/render_route_dumps.py --dumps a.pt b.pt --out artifacts/figures
    python scripts/render_route_dumps.py --dumps DIR --zoom 13 --format png pdf

``a_<alpha>_t_<target>`` in a file name (``NEA_Edit_a_0_5_t_0_2.pt``) sets that
dump's alpha / adjustment target; ``--alpha`` / ``--adj-target`` supply them when
the name does not, and the objective YAML defaults apply when neither does.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_cli import setup_logging   # noqa: E402

from connectpt.routes_generator import render_report                    # noqa: E402
from connectpt.routes_generator.reports import figure_key_slug          # noqa: E402
from connectpt.routes_generator.data import as_route_tensor             # noqa: E402
from connectpt.routes_generator.data.sources import REGISTRY            # noqa: E402
from connectpt.routes_generator.evaluation.route_scoring import (       # noqa: E402
    full_metric_row, score_fixed_routes)

# "..._a_0_5_t_0_2" -- alpha and adjustment target encoded in a file name.
NAME_PARAMS = re.compile(r"^(?P<label>.+?)_a_(?P<alpha>[\d_]+?)_t_(?P<target>[\d_]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dumps", nargs="+", required=True, type=Path,
                        help="route dump files, or directories to scan for *.pt")
    parser.add_argument("--city", default="ekb", choices=sorted(REGISTRY),
                        help="data source supplying the seed network + geometry")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: alongside the dumps)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="RTT/WMC trade-off for dumps whose name lacks a_<alpha>")
    parser.add_argument("--adj-target", type=float, default=None,
                        help="adjustment target for dumps whose name lacks t_<target>")
    parser.add_argument("--zoom", type=int, default=None,
                        help="basemap tile zoom (default: the library's)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="output resolution; 200 roughly halves the file size")
    parser.add_argument("--format", nargs="+", default=["png"],
                        choices=["png", "pdf", "svg"],
                        help="output formats; pdf/svg keep the routes vector "
                             "(only the basemap stays raster) -- what a journal wants")
    return parser.parse_args()


def collect_dumps(paths) -> list[Path]:
    """Expand the ``--dumps`` arguments into a sorted list of dump files."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.pt")))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise SystemExit(f"no route dumps found in {[str(p) for p in paths]}")
    return files


def dump_params(path: Path, alpha, adj_target) -> tuple[str, float | None, float | None]:
    """``(label, alpha, adj_target)`` for one dump -- name first, CLI as fallback."""
    match = NAME_PARAMS.match(path.stem)
    if match is None:
        return path.stem.replace("_", " "), alpha, adj_target
    as_float = lambda text: float(text.replace("_", "."))   # noqa: E731
    return (match["label"].replace("_", " "),
            as_float(match["alpha"]) if alpha is None else alpha,
            as_float(match["target"]) if adj_target is None else adj_target)


def main() -> None:
    args = parse_args()
    dumps = collect_dumps(args.dumps)
    out_dir = args.out or dumps[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "render_route_dumps.log")
    log.info("rendering %d dump(s) on the %s network -> %s",
             len(dumps), args.city, out_dir)

    instance = REGISTRY[args.city]().load()
    seed = as_route_tensor(instance.init_routes)

    # The seed network is scored once: RTT / WMC / demand split do not depend on
    # alpha, and its adjustment against itself is zero by construction.
    seed_metrics, seed_scored = score_fixed_routes(seed, instance.tensors, instance.spec)
    rows = [dict(full_metric_row(seed_metrics, seed_scored, seed), method="Initial",
                 alpha=None, adj_target=None, source="seed")]
    routes = {"Initial": seed}
    figure_names = {}

    for path in dumps:
        label, alpha, adj_target = dump_params(path, args.alpha, args.adj_target)
        candidate = as_route_tensor(
            torch.load(path, map_location="cpu", weights_only=False))
        metrics, scored = score_fixed_routes(
            candidate, instance.tensors, instance.spec, alpha=alpha,
            adj_target=adj_target, seed_routes=seed)
        row = full_metric_row(metrics, scored, seed)
        key = f"{label} a={alpha} t={adj_target}"
        rows.append(dict(row, method=label, alpha=alpha, adj_target=adj_target,
                         source=path.name))
        routes[key] = candidate
        figure_names[key] = path.stem
        log.info("%s: RTT=%.1f WMC=%.2f adj=%.3f", path.name,
                 row["RTT"], row["WMC"], row["adj_vs_seed"])

    # render_report matches a panel's metrics row by "<method> a=<alpha> t=<target>",
    # so the table is built with exactly the method/alpha/adj_target used above.
    table = pd.DataFrame(rows)
    table_path = out_dir / "route_dump_metrics.csv"
    table.round(4).to_csv(table_path, index=False)
    log.info("metrics -> %s", table_path)

    result = SimpleNamespace(
        run_name=instance.label, table=table, instance=instance, routes=routes,
        metadata={"report_kind": "gis"})
    report = render_report(result, kind="gis", title=instance.label,
                           max_route_panels=len(dumps), basemap_zoom=args.zoom)

    for key, name in figure_names.items():
        figure = (report.figures.get(f"map_{figure_key_slug(key)}")
                  or report.figures.get("map"))
        if figure is None:
            log.warning("no figure rendered for %s", key)
            continue
        for suffix in args.format:
            path = out_dir / f"{name}.{suffix}"
            figure.savefig(path, dpi=args.dpi, bbox_inches="tight",
                           facecolor="white")
            log.info("figure -> %s", path)


if __name__ == "__main__":
    main()
