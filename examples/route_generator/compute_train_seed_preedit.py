#!/usr/bin/env python
"""Compute the PRE-EDIT train seed cost, per curriculum stage.

The recorded ``train_seed_cost`` in the training history is NOT the cost of the
untouched input network: the rollout edits route slots sequentially and logs, for
each slot, the network cost at the moment that slot starts -- which already
contains the model's edits to the previous slots. So it drops as the model
learns, unlike ``val_seed_cost`` (the static full-network pre-edit cost).

This script computes the honest apples-to-apples analogue: the FULL seed network
cost of the training pool, before any edits, at the fixed 0.5/0.5 eval weights --
exactly how val seed is measured, but on the train split. It is model-independent
and therefore constant within a curriculum stage (stepped, like val seed).

Output: a NEW CSV (never overwrites) consumed by the render script to overlay a
"train initial, pre-edit (fixed weights)" line on the training panel.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_lib.context import ROOT_DIR
from compute_val_seed_components import (
    TIERS, PER, STAGES, tier_of, N_GRAPHS, SPLIT_SEED, TRAIN_FRACTION,
    TRAIN_EVAL_N_PER_TIER, MIN_ROUTE_LEN, MAX_ROUTE_LEN, TARGET_N_ROUTES,
    build_cost_obj)
from connectpt.routes_generator.improvement_learning import (
    load_raw_graphs_and_lc_routes, make_improvement_batch, _clone_cost_weights)
from connectpt.routes_generator.transit_time_estimator import RouteGenBatchState

DEFAULT_DIR = ROOT_DIR / "datasets" / "lc_copytiers_n1000_n50_r12_len8_15_v1"
DEFAULT_OUT = (ROOT_DIR / "artifacts" / "paper_final" / "training"
               / "NEW_lc_copytiers_curric_noadj_v1_train_seed_preedit.csv")
# Curriculum stage boundaries (until_epoch), mirroring CURRICULUM in cell 5.
N_ITERATIONS = 700
STAGE_UNTIL = {
    "full":      round(0.15 * N_ITERATIONS),
    "+boundary": round(0.30 * N_ITERATIONS),
    "+mixed":    round(0.55 * N_ITERATIONS),
    "+covered":  round(0.75 * N_ITERATIONS),
    "+clean":    N_ITERATIONS,
}


def train_indices_by_tier():
    """Reproduce the training split: per tier, the graphs NOT in validation."""
    perm = torch.randperm(
        N_GRAPHS, generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
    by_tier = {t: [] for t in TIERS}
    for gi in perm:
        by_tier[tier_of(gi)].append(gi)
    train_by_tier = {}
    for t in TIERS:
        idxs = by_tier[t]
        n_val = max(TRAIN_EVAL_N_PER_TIER, round((1 - TRAIN_FRACTION) * len(idxs)))
        n_val = min(n_val, max(0, len(idxs) - 1))
        train_by_tier[t] = idxs[n_val:]
    return train_by_tier


def per_graph_seed_cost(cost_obj, graphs, seed_routes, indices, device, batch=25):
    """Full-network pre-edit seed cost (fixed weights) for each graph index."""
    w = cost_obj.get_weights(device)
    out = {}
    idx = list(indices)
    for s in range(0, len(idx), batch):
        chunk = torch.tensor(idx[s:s + batch], dtype=torch.long)
        gb, rb = make_improvement_batch(
            graphs, seed_routes, chunk, device, training=False,
            target_n_routes=TARGET_N_ROUTES)
        st = RouteGenBatchState(gb, cost_obj, rb.shape[1], MIN_ROUTE_LEN,
                                MAX_ROUTE_LEN, cost_weights=_clone_cost_weights(w))
        st.add_new_routes(rb)
        costs = cost_obj(st).cost.detach().cpu().tolist()
        for gi, c in zip(idx[s:s + batch], costs):
            out[gi] = float(c)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default=str(DEFAULT_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--force", action="store_true", help="overwrite output CSV if it exists")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {out_path} (use --force)")

    device = "cpu"
    cost_obj = build_cost_obj(device)
    dataset_dir = Path(args.dataset_dir)
    graphs, seed_routes = load_raw_graphs_and_lc_routes(
        dataset_dir / "raw_graphs_subset.pkl", dataset_dir)

    train_by_tier = train_indices_by_tier()
    all_train = sorted(gi for lst in train_by_tier.values() for gi in lst)
    print(f"train graphs: {len(all_train)} "
          f"({ {t: len(v) for t, v in train_by_tier.items()} })")
    costs = per_graph_seed_cost(cost_obj, graphs, seed_routes, all_train, device)

    rows = []
    for label, tiers in STAGES:
        active = [gi for t in tiers for gi in train_by_tier[t]]
        vals = np.array([costs[gi] for gi in active])
        rows.append({
            "curriculum_stage": label,
            "until_epoch": STAGE_UNTIL[label],
            "n_graphs": len(active),
            "train_seed_preedit_mean": round(float(vals.mean()), 6),
            "p25": round(float(np.percentile(vals, 25)), 6),
            "p75": round(float(np.percentile(vals, 75)), 6),
        })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
