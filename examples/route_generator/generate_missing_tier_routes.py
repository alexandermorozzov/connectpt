#!/usr/bin/env python
"""Regenerate the MISSING per-graph LC seed routes for the copy-tiers dataset.

The ``lc_copytiers_n1000_...`` dataset used by the ``NEW_lc_copytiers_curric``
run stores one corrupted-seed-route file per graph in ``graph_XXXX/``. Only the
first tiers (graphs 0000..0399 = ``copy_full`` + ``copy_boundary``) are present
locally; the remaining tiers (``copy_mixed`` / ``covered_dup`` / ``lc_clean``,
graphs 0400..0999) are missing. This script reconstructs ONLY the missing files
by replaying the notebook's generation recipe (paper_combined.ipynb cell 7):

    LC learned construction (per objective combo)  ->  per-tier copy corruption
    (inject_route_copies, seeded per graph)         ->  dump to graph_XXXX/

Faithfulness note: LC construction runs with ``n_samples=1``, which is the
*stochastic* eval path, so regenerated routes are statistically equivalent to
the originals (same tier corruption, same per-graph corruption seed) but not
bit-identical. Existing files on disk are NEVER overwritten -- only missing
graph indices are generated.

Usage:
  python generate_missing_tier_routes.py                 # fill all missing
  python generate_missing_tier_routes.py --only 400-403  # small test slice
  python generate_missing_tier_routes.py --force         # regenerate all 1000
"""
import argparse
import json
import pickle
import random as _random
import sys
from pathlib import Path

import pandas as pd
import torch

from eval_lib.context import ROOT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval_lib import build_lc_cfg, run_lc_batch, as_route_tensor  # noqa: E402
from eval_lib.params import CONNECTIVITY_MODE  # noqa: E402
from eval_lib.route_copies import (  # noqa: E402
    COPY_TIER_CFG as TIER_CFG, inject_route_copies,
    count_changed_routes as _count_changed_routes,
    redundancy_stats as _redundancy_stats,
    uncovered_demand_pct as _uncovered_demand_pct)
from connectpt.routes_generator.torch_utils import dump_routes  # noqa: E402
from connectpt.routes_generator.transit_time_estimator import STOP_KEY  # noqa: E402

# --- dataset knobs, mirrored from paper_combined.ipynb cell 5 --------------
N_GRAPHS = 1000
RAW_N_NODES = 50
TARGET_N_ROUTES = 12
MIN_ROUTE_LEN = 8
MAX_ROUTE_LEN = 15
LC_N_SAMPLES = 1
LC_COMBOS = [(1.0, 0.0, 0.0, "demand"), (0.0, 1.0, 0.0, "route"),
             (0.0, 0.0, 1.0, "conn")]
TIERS = list(TIER_CFG)
PER = N_GRAPHS // len(TIERS)   # 200 graphs per tier, sequential blocks

DEFAULT_DIR = ROOT_DIR / "datasets" / "lc_copytiers_n1000_n50_r12_len8_15_v1"
DEFAULT_RAW = ROOT_DIR / "datasets" / "raw_graphs_1000.pkl"


def _tensors(g):
    return {"node_locs": g[STOP_KEY].pos.detach().cpu().clone(),
            "street_adj": g.street_adj.detach().cpu().clone(),
            "demand": g.demand.detach().cpu().clone()}


def _to_fixed(routes):
    t = as_route_tensor(routes).long()
    if t.ndim == 3:
        t = t[0]
    if t.shape[0] < TARGET_N_ROUTES:
        t = torch.cat([t, torch.full((TARGET_N_ROUTES - t.shape[0], t.shape[1]), -1, dtype=t.dtype)], 0)
    else:
        t = t[:TARGET_N_ROUTES]
    if t.shape[1] < MAX_ROUTE_LEN:
        t = torch.cat([t, torch.full((t.shape[0], MAX_ROUTE_LEN - t.shape[1]), -1, dtype=t.dtype)], 1)
    elif t.shape[1] > MAX_ROUTE_LEN:
        t = t[:, :MAX_ROUTE_LEN]
    return t


def _route_file(dataset_dir, gi):
    """Existing corrupted-routes file for graph gi, or None."""
    gdir = dataset_dir / f"graph_{gi:04d}"
    matches = sorted(gdir.glob("lc_*_routes_routes.pkl")) if gdir.exists() else []
    return matches[0] if matches else None


def tier_of(gi):
    return TIERS[min(gi // PER, len(TIERS) - 1)]


def parse_only(spec):
    """"400-403" or "400,401,999" -> sorted list of ints."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < N_GRAPHS)


def _corrupt_and_dump(gi, lc_route, tn, dataset_dir):
    """Per-tier copy corruption + dump for one graph; returns its meta row."""
    tier = tier_of(gi)
    tier_cfg = TIER_CFG[tier]
    rng = _random.Random(1000 + gi)
    routes = _to_fixed(lc_route)
    before = _redundancy_stats(routes)
    clean_routes = routes.clone()
    routes, event_counts, _ = inject_route_copies(
        routes, tier_cfg, rng, MIN_ROUTE_LEN, MAX_ROUTE_LEN,
        demand=tn["demand"], n_nodes=RAW_N_NODES, street_adj=tn["street_adj"])
    after = _redundancy_stats(routes)
    n_corrupted = _count_changed_routes(clean_routes, routes)
    gdir = dataset_dir / f"graph_{gi:04d}"
    gdir.mkdir(parents=True, exist_ok=True)
    dump_routes(f"lc_copy_cur_graph_{gi:04d}_routes", routes, out_dir=gdir)
    return {
        "graph_index": gi, "tier": tier,
        "tier_kind": ("clean" if not tier_cfg["kinds"] else "copies"),
        "lc_combo": LC_COMBOS[gi % len(LC_COMBOS)][3],
        "applied_events": int(sum(event_counts.values())),
        "n_corrupted_routes": int(n_corrupted),
        "mutation_events": json.dumps(dict(event_counts), sort_keys=True),
        "d_un_after_pct": round(_uncovered_demand_pct(routes, tn["demand"], RAW_N_NODES), 2),
        "redun_before": round(before["redundancy"], 4),
        "redun_after": round(after["redundancy"], 4),
        "max_leg_use_before": before["max_leg_use"],
        "max_leg_use_after": after["max_leg_use"],
    }


def generate(graphs, indices, dataset_dir, batch_size):
    """Construct (LC) + corrupt + dump each missing graph, per objective combo.

    Dumps incrementally right after each batch is constructed, so progress is
    visible live and a crash mid-run keeps everything produced so far.
    """
    tensors = {gi: _tensors(graphs[gi]) for gi in indices}
    meta_rows = []
    done = 0
    for ci, (d, rt, cn, ctag) in enumerate(LC_COMBOS):
        combo = [gi for gi in indices if gi % len(LC_COMBOS) == ci]
        if not combo:
            continue
        cfg = build_lc_cfg(
            run_name=f"copy_cur_{ctag}", n_routes=TARGET_N_ROUTES,
            min_route_len=MIN_ROUTE_LEN, max_route_len=MAX_ROUTE_LEN,
            demand_time_weight=d, route_time_weight=rt,
            median_connectivity_weight=cn, connectivity_mode=CONNECTIVITY_MODE)
        for s in range(0, len(combo), batch_size):
            chunk = combo[s:s + batch_size]
            routes_b = run_lc_batch(
                cfg, [tensors[gi] for gi in chunk],
                run_name_prefix="copy_cur_", n_samples=LC_N_SAMPLES,
                batch_size=len(chunk))
            for j, gi in enumerate(chunk):
                meta_rows.append(_corrupt_and_dump(gi, routes_b[j], tensors[gi], dataset_dir))
            done += len(chunk)
            print(f"[{ctag}] {min(s + len(chunk), len(combo))}/{len(combo)} "
                  f"| total {done}/{len(indices)} dumped", flush=True)
    return meta_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="dataset directory")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="raw graphs pkl (all N_GRAPHS)")
    p.add_argument("--only", type=str, help="restrict to indices, e.g. '400-403' or '400,999'")
    p.add_argument("--batch", type=int, default=8, help="LC construction batch size")
    p.add_argument("--seed", type=int, default=0, help="torch/random seed for reproducibility")
    p.add_argument("--force", action="store_true", help="also regenerate graphs that already have a routes file (overwrites!)")
    p.add_argument("--write-meta", action="store_true", help="(re)write meta.csv + raw_graphs_subset.pkl if missing")
    args = p.parse_args()

    with args.raw.open("rb") as fh:
        graphs = pickle.load(fh)
    if len(graphs) != N_GRAPHS:
        print(f"WARNING: raw graphs has {len(graphs)} != expected {N_GRAPHS}")

    candidates = parse_only(args.only) if args.only else list(range(len(graphs)))
    if args.force:
        todo = candidates
    else:
        todo = [gi for gi in candidates if _route_file(args.dir, gi) is None]
    present = len(candidates) - len(todo)
    print(f"{len(candidates)} candidate graphs: {present} already on disk, "
          f"{len(todo)} to generate")
    if not todo:
        print("nothing to do")
        return

    torch.manual_seed(args.seed)
    _random.seed(args.seed)

    meta_rows = generate(graphs, todo, args.dir, args.batch)
    print(f"dumped {len(meta_rows)} route files -> {args.dir}", flush=True)

    # Optional: complete the dataset so paper_combined.ipynb can load it too.
    # tier is purely index-derived, so a correct tier map can always be written
    # for ALL graphs even where per-graph corruption stats are unknown.
    if args.write_meta:
        subset_pkl = args.dir / "raw_graphs_subset.pkl"
        if not subset_pkl.exists():
            with subset_pkl.open("wb") as fh:
                pickle.dump(list(graphs), fh)
            print(f"wrote {subset_pkl}")
        meta_csv = args.dir / "meta.csv"
        if not meta_csv.exists():
            by_gi = {r["graph_index"]: r for r in meta_rows}
            full = [by_gi.get(gi, {"graph_index": gi, "tier": tier_of(gi),
                                   "lc_combo": LC_COMBOS[gi % len(LC_COMBOS)][3]})
                    for gi in range(len(graphs))]
            pd.DataFrame(full).to_csv(meta_csv, index=False)
            print(f"wrote {meta_csv} (tier map exact; corruption stats only for "
                  f"freshly generated graphs)")


if __name__ == "__main__":
    main()
