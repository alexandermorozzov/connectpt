#!/usr/bin/env python
"""Compute PER-GRAPH validation-seed cost components for the copy-tiers run.

The training loop logs only the *mean* validation seed cost per epoch, which is
piecewise-constant within a curriculum stage (fixed val graphs + fixed eval
weights). To draw the actual per-graph spread -- and to reweight the seed cost
under freshly sampled weights (a faithful "mock" of the training operating
point) -- we need each val graph's unweighted cost components.

This script:
  1. Rebuilds the exact balanced validation monitor (``MONITOR_VAL_INDICES``)
     from the deterministic split in paper_combined.ipynb (randperm seed 0,
     stratified 4 graphs/tier).
  2. Builds the eval cost object (RTT + WMC, demand off, median_weighted).
  3. Computes each val graph's ``route`` and ``conn`` seed-cost components, from
     which any weighting ``w_route*route + w_conn*conn`` is exact (weights lie
     on the simplex, so the shared constraint term cancels).

Output: a NEW CSV (never overwrites); the render script consumes it.

Faithfulness: for the copy_full / copy_boundary tiers the on-disk original seed
routes are used; the other tiers use regenerated routes (see
generate_missing_tier_routes.py), so their absolute values are statistically
equivalent but not identical to the original run.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from eval_lib.context import ROOT_DIR, CFG_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hydra import compose, initialize_config_dir  # noqa: E402
import connectpt.routes_generator.utils as lrnu  # noqa: E402
from eval_lib.params import (  # noqa: E402
    CONNECTIVITY_MODE, DISABLED_COST_COMPONENTS, UNIFIED_COST_WEIGHTS)
from connectpt.routes_generator.improvement_learning import (  # noqa: E402
    load_raw_graphs_and_lc_routes, make_improvement_batch, _clone_cost_weights)
from connectpt.routes_generator.transit_time_estimator import RouteGenBatchState  # noqa: E402

# Mirrored from paper_combined.ipynb cell 5 / cell 9.
N_GRAPHS = 1000
MIN_ROUTE_LEN, MAX_ROUTE_LEN = 8, 15
TARGET_N_ROUTES = 12
SPLIT_SEED = 0
TRAIN_FRACTION = 0.9
TRAIN_EVAL_N_PER_TIER = 4
TIERS = ["copy_full", "copy_boundary", "copy_mixed", "covered_dup", "lc_clean"]
PER = N_GRAPHS // len(TIERS)
# Cumulative curriculum stages (label -> tiers active), matching CURRICULUM.
STAGES = [
    ("full",      TIERS[:1]),
    ("+boundary", TIERS[:2]),
    ("+mixed",    TIERS[:3]),
    ("+covered",  TIERS[:4]),
    ("+clean",    TIERS[:5]),
]

DEFAULT_DIR = ROOT_DIR / "datasets" / "lc_copytiers_n1000_n50_r12_len8_15_v1"
DEFAULT_OUT = (ROOT_DIR / "artifacts" / "paper_final" / "training"
               / "NEW_lc_copytiers_curric_noadj_v1_val_seed_per_graph.csv")


def tier_of(gi):
    return TIERS[min(gi // PER, len(TIERS) - 1)]


def monitor_val_indices():
    """Reproduce MONITOR_VAL_INDICES: 4 validation graphs per tier."""
    perm = torch.randperm(
        N_GRAPHS, generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
    by_tier = {t: [] for t in TIERS}
    for gi in perm:
        by_tier[tier_of(gi)].append(gi)
    val_by_tier = {}
    for t in TIERS:
        idxs = by_tier[t]
        n_val = max(TRAIN_EVAL_N_PER_TIER, round((1 - TRAIN_FRACTION) * len(idxs)))
        n_val = min(n_val, max(0, len(idxs) - 1))
        val_by_tier[t] = idxs[:n_val]
    # Balanced monitor slice: first TRAIN_EVAL_N_PER_TIER val graphs per tier.
    return {t: val_by_tier[t][:TRAIN_EVAL_N_PER_TIER] for t in TIERS}


def build_cost_obj(device):
    overrides = [
        "model=bestsofar_feb2023_trim",
        "++run_name=val_seed_components", "++experiment.logdir=null",
        f"++experiment.cost_function.kwargs.connectivity_mode={CONNECTIVITY_MODE}",
    ]
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="ppo_50nodes.yaml", overrides=overrides)
    _, _, _, cost_obj, _ = lrnu.process_standard_experiment_cfg(
        cfg, run_name_prefix="val_seed_")
    cost_obj.ignore_stops_oob = True
    cost_obj.set_enabled_components(disabled_components=DISABLED_COST_COMPONENTS)
    cost_obj.set_weights(route_time_weight=UNIFIED_COST_WEIGHTS["route_time_weight"])
    cost_obj.median_connectivity_weight = float(
        UNIFIED_COST_WEIGHTS["median_connectivity_weight"])
    return cost_obj


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="dataset directory")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output CSV (must not exist unless --force)")
    p.add_argument("--force", action="store_true", help="overwrite the output CSV if it exists")
    args = p.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.out} (use --force)")

    device = "cpu"
    cost_obj = build_cost_obj(device)
    w = cost_obj.get_weights(device)
    print("eval weights:", {k: float(v) for k, v in w.items()})

    graphs, seed_routes = load_raw_graphs_and_lc_routes(
        args.dir / "raw_graphs_subset.pkl", args.dir)
    monitor = monitor_val_indices()
    # component index order: [demand, route, connectivity]
    rows = []
    for tier in TIERS:
        for gi in monitor[tier]:
            gb, rb = make_improvement_batch(
                graphs, seed_routes, torch.tensor([gi]), device,
                training=False, target_n_routes=TARGET_N_ROUTES)
            st = RouteGenBatchState(gb, cost_obj, rb.shape[1], MIN_ROUTE_LEN,
                                    MAX_ROUTE_LEN, cost_weights=_clone_cost_weights(w))
            st.add_new_routes(rb)
            res = cost_obj(st)
            comps = cost_obj.get_cost_components(st, result=res)[0].tolist()
            rows.append({
                "graph_index": int(gi), "tier": tier,
                "route_comp": round(comps[1], 6),
                "conn_comp": round(comps[2], 6),
                "seed_cost_fixed": round(float(res.cost[0]), 6),  # 0.5/0.5 point
            })
        print(f"[{tier}] {len(monitor[tier])} val graphs done", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")

    # Sanity: per-tier mean fixed seed cost (compare to training_history steps).
    print("\nper-tier mean seed_cost_fixed:")
    print(df.groupby("tier")["seed_cost_fixed"].mean().reindex(TIERS).round(3).to_string())


if __name__ == "__main__":
    main()
