#!/usr/bin/env python
"""Redraw the "Mean cost" training figure (train/val curves + clean-LC baseline,
with curriculum-stage shading) from the CSVs a training run leaves behind.

This reproduces ``*_cost_train_val.svg`` without touching the original: outputs
get a ``_repro`` suffix so an existing figure is never overwritten.

Two figure VARIANTS are produced (``--variant static|mock|both``):

  * ``static`` -- the honest, apples-to-apples picture. The "initial routes"
    line is the STATIC full-network pre-edit cost at fixed 0.5/0.5 eval weights,
    drawn identically on both panels (train pool vs val pool): flat/stepped and
    SOLID. Nothing breathes.
  * ``mock``   -- identical to ``static`` EXCEPT the validation "initial routes"
    line is recomputed under resampled cost weights (the same sampler training
    uses), so it "breathes" like the train curve. The train pre-edit line stays
    static (train has no per-graph components saved to reweight).

Honest labelling of the recorded train seed curve
-------------------------------------------------
The recorded ``train_seed_cost`` is NOT the untouched input cost. The rollout
edits route slots sequentially and logs, per slot, the network cost at that
slot's START -- which already contains the model's edits to the earlier slots
(``current_start_cost``, improvement_learning.py ~1045/1188). So it declines as
the model learns. It is therefore labelled as a per-edit-step cost that carries
the previous step's edits, NOT as "initial routes". The true model-independent
initial level is the separate static pre-edit line.

Inputs:
  * ``<run>_training_history_partial.csv`` -- per-epoch train/val seed & final
    cost columns plus ``curriculum_stage``.
  * ``<run>_clean_lc_baseline_cost.csv``   -- one clean-LC cost per curriculum
    stage.
  * ``<run>_train_seed_preedit.csv`` (from compute_train_seed_preedit.py)
    -- per-stage STATIC pre-edit train-pool seed cost (fixed 0.5/0.5).
  * ``<run>_val_seed_per_graph.csv`` (from compute_val_seed_components.py)
    -- per-val-graph ``route_comp``/``conn_comp``; used only for the ``mock``
    variant's resampled-weights validation line.

Usage:
  python render_cost_train_val_figure.py                      # both variants
  python render_cost_train_val_figure.py --variant static
  python render_cost_train_val_figure.py --dir DIR --run NAME
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Repo default: the run whose figure we are reproducing.
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "paper_final" / "training"
DEFAULT_RUN = "NEW_lc_copytiers_curric_noadj_v1"

# The cost curves live in TWO different measurement systems, so they are drawn on
# two separate panels rather than a single shared axis:
#   * TRAIN  -- per-batch randomly sampled cost weights, stochastic policy.
#   * VAL    -- fixed eval weights, greedy policy, fixed validation graphs.
# Absolute levels / gaps are only comparable *within* a panel. The clean-LC
# baseline shares the validation operating point, so it belongs on the val panel.

# Recorded train seed curve: a per-edit-step cost, NOT the untouched input (see
# module docstring). Labelled honestly.
TRAIN_STEP = ("train_seed_cost",
              "cost at each edit step (input = previous step's routes)",
              "#7c3aed")
TRAIN_FINAL = ("train_final_cost", "after model edits", "#0072bc")
VAL_FINAL = ("val_final_cost", "after model edits", "#f39c12")

# The one "true initial" quantity, drawn identically on both panels: the STATIC
# full-network pre-edit cost at fixed 0.5/0.5 weights. Train pool uses the
# per-stage pre-edit means; val pool uses the recorded (flat/stepped) val seed.
INITIAL_COLOR = "#4b1a99"     # dark purple
INITIAL_LABEL = "initial routes, pre-edit (static, fixed 0.5/0.5)"
INITIAL_MOCK_LABEL = "val initial, resampled-weights (mock, fixed graphs)"

CLEAN_LC_COLOR = "#dc2626"
CLEAN_LC_LABEL = "clean-LC heuristic baseline"
SHADE_COLORS = ["#eaf3ff", "#eafbea", "#fff6e6", "#fdeaea", "#f0eaff"]

TRAIN_TITLE = "Training rollout  —  sampled cost weights, stochastic policy"
VAL_TITLE = "Validation  —  fixed eval weights, greedy policy"

# Curriculum tier order: stage i (in span order) activates TIER_ORDER[i], so a
# tier's validation graphs first enter the active set at that stage's start.
TIER_ORDER = ["copy_full", "copy_boundary", "copy_mixed", "covered_dup", "lc_clean"]

# Weight sampler parameters for the edit run (demand disabled -> enabled =
# {route, conn}); mirrors cost_obj.sample_variable_weights with pp/op/mcw below.
SAMPLER_PP, SAMPLER_OP, SAMPLER_MCW = 0.0, 0.3, 0.3


def sample_route_weight(rng):
    """One (w_route, w_conn) draw, replicating sample_variable_weights over the
    two enabled components. Returns w_route (w_conn = 1 - w_route)."""
    r = rng.random()
    if r < SAMPLER_PP:                       # passenger-perspective (demand) -> masked out
        return 0.5                           # unreachable with PP=0, kept for parity
    if r < SAMPLER_PP + SAMPLER_OP:          # operator-perspective: route only
        return 1.0
    if r < SAMPLER_PP + SAMPLER_OP + SAMPLER_MCW:  # connectivity only
        return 0.0
    a, b = rng.random(), rng.random()        # intermediate: uniform on 2-simplex
    return a / (a + b) if (a + b) > 0 else 0.5


def stage_spans(history):
    """Contiguous (start_epoch, end_epoch, label) spans from curriculum_stage."""
    spans = []
    rows = history.dropna(subset=["curriculum_stage"])[["epoch", "curriculum_stage"]]
    for epoch, label in rows.itertuples(index=False, name=None):
        epoch = int(epoch)
        if spans and spans[-1][2] == label:
            spans[-1] = (spans[-1][0], epoch, label)
        else:
            spans.append((epoch, epoch, label))
    return spans


def _shade_stages(ax, spans):
    """Curriculum-stage background shading + dotted boundary lines."""
    for k, (start, _end, _label) in enumerate(spans):
        end = _end
        ax.axvspan(start, end, color=SHADE_COLORS[k % len(SHADE_COLORS)], alpha=0.6, zorder=0)
        ax.axvline(start, color="gray", lw=0.6, ls=":", zorder=1)


def _plot_curve(ax, history, col, label, color, smooth):
    """Plot one cost curve (with optional centred rolling mean) onto ``ax``."""
    y = pd.to_numeric(history[col], errors="coerce")
    mask = y.notna()
    x, yv = history["epoch"][mask], y[mask]
    if smooth and smooth > 1:
        yv = yv.rolling(smooth, center=True, min_periods=1).mean()
    ax.plot(x, yv, color=color, lw=1.9, label=label, zorder=3)


def _plot_stage_flat(ax, spans, by_stage, color, label):
    """Flat/stepped SOLID line: one horizontal segment per curriculum stage."""
    labelled = False
    for start, end, lab in spans:
        if lab not in by_stage:
            continue
        ax.plot([start, end], [by_stage[lab]] * 2, color=color, lw=2.0, ls="-",
                zorder=4, label=None if labelled else label)
        labelled = True


def _plot_stage_step(ax, spans, by_stage, color, label):
    """Single CONNECTED stepped line across all stages (no smoothing): one
    horizontal run per stage, joined by vertical steps at the boundaries."""
    xs, ys = [], []
    for start, end, lab in spans:
        if lab not in by_stage:
            continue
        xs += [start, end]
        ys += [by_stage[lab], by_stage[lab]]
    if xs:
        ax.plot(xs, ys, color=color, lw=2.0, ls="-", zorder=4, label=label)


def _val_seed_by_stage(history):
    """Per-stage value of the recorded (piecewise-constant) val seed cost."""
    rows = history.dropna(subset=["curriculum_stage"])
    return rows.groupby("curriculum_stage")["val_seed_cost"].median().to_dict()


def _tier_entry_epochs(spans):
    """tier -> epoch at which its validation graphs first become active."""
    return {TIER_ORDER[i]: start for i, (start, _e, _l) in enumerate(spans)
            if i < len(TIER_ORDER)}


def _val_mock_series(spans, per_graph, epochs, mock_seed):
    """Resampled-weights validation-initial series: each epoch draws ONE weight
    vector with the training sampler and scores the active seed graphs with it,
    so the line breathes like the train curve instead of stepping."""
    entry = _tier_entry_epochs(spans)
    rows = [r for r in per_graph.itertuples(index=False) if r.tier in entry]
    route = {r.tier: [] for r in rows}
    conn = {r.tier: [] for r in rows}
    for r in rows:
        route[r.tier].append(r.route_comp)
        conn[r.tier].append(r.conn_comp)
    rng = random.Random(mock_seed)
    xs, ys = [], []
    for e in epochs:
        active = [t for t, ent in entry.items() if ent <= e]
        r_all = [v for t in active for v in route[t]]
        c_all = [v for t in active for v in conn[t]]
        if not r_all:
            continue
        w = sample_route_weight(rng)
        costs = [w * rc + (1.0 - w) * cc for rc, cc in zip(r_all, c_all)]
        xs.append(e)
        ys.append(sum(costs) / len(costs))
    return xs, ys


def _stage_labels(ax, spans):
    """Curriculum-stage names just inside the top of the axis."""
    for start, end, label in spans:
        ax.text((start + end) / 2, 0.98, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#444")


def render(history_csv, baseline_csv, out_path, variant, smooth=15,
           per_graph_csv=None, mock_seed=0, preedit_csv=None):
    """Render one figure variant ('static' or 'mock')."""
    history = pd.read_csv(history_csv)
    baseline = pd.read_csv(baseline_csv)
    spans = stage_spans(history)
    clean_by_stage = dict(zip(baseline["curriculum_stage"], baseline["clean_lc_cost"]))

    preedit = (pd.read_csv(preedit_csv)
               if preedit_csv and Path(preedit_csv).exists() else None)
    preedit_by_stage = (dict(zip(preedit["curriculum_stage"],
                                 preedit["train_seed_preedit_mean"]))
                        if preedit is not None else {})

    per_graph = (pd.read_csv(per_graph_csv)
                 if per_graph_csv and Path(per_graph_csv).exists() else None)
    if variant == "mock" and per_graph is None:
        raise SystemExit(
            f"variant 'mock' needs the per-graph CSV ({per_graph_csv}); "
            "run compute_val_seed_components.py first, or use --variant static")

    fig, (ax_t, ax_v) = plt.subplots(
        2, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    fig.suptitle(
        "Mean route cost over training  —  "
        f"{'static initial (fixed 0.5/0.5)' if variant == 'static' else 'val initial under resampled weights'}\n"
        "(train and validation use different weights/policies "
        "and are NOT comparable across panels)",
        fontsize=12)

    # --- Training panel -------------------------------------------------
    _shade_stages(ax_t, spans)
    _plot_curve(ax_t, history, *TRAIN_STEP, smooth)
    _plot_curve(ax_t, history, TRAIN_FINAL[0], TRAIN_FINAL[1], TRAIN_FINAL[2], smooth)
    # The true, model-independent initial level (identical treatment to val):
    # one connected stepped line, not smoothed.
    _plot_stage_step(ax_t, spans, preedit_by_stage, INITIAL_COLOR, INITIAL_LABEL)
    ax_t.set_title(TRAIN_TITLE, fontsize=10)
    ax_t.set_ylabel("Mean cost")
    ax_t.grid(alpha=0.2)
    ax_t.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True,
                framealpha=0.95, fontsize=8, title="gap = the model's improvement")
    _stage_labels(ax_t, spans)

    # --- Validation panel (shares the clean-LC operating point) ---------
    _shade_stages(ax_v, spans)
    _plot_curve(ax_v, history, VAL_FINAL[0], VAL_FINAL[1], VAL_FINAL[2], smooth)
    # val "initial routes": static (flat/stepped) or resampled-weights mock.
    if variant == "static":
        # Recorded val seed is already the static pre-edit cost. Draw it with the
        # SAME connected-stepped, unsmoothed style as the train pre-edit line: it
        # is the SAME quantity, other split.
        _plot_stage_step(ax_v, spans, _val_seed_by_stage(history),
                         INITIAL_COLOR, INITIAL_LABEL)
    else:
        xs, ys = _val_mock_series(spans, per_graph,
                                  list(pd.to_numeric(history["epoch"])), mock_seed)
        ys = pd.Series(ys)
        if smooth and smooth > 1:
            ys = ys.rolling(smooth, center=True, min_periods=1).mean()
        ax_v.plot(xs, ys, color=INITIAL_COLOR, lw=1.9, ls="-", zorder=3,
                  label=INITIAL_MOCK_LABEL)
    _plot_stage_flat(ax_v, spans, clean_by_stage, CLEAN_LC_COLOR, CLEAN_LC_LABEL)
    ax_v.set_title(VAL_TITLE, fontsize=10)
    ax_v.set_xlabel("epochs")
    ax_v.set_ylabel("Mean cost")
    ax_v.grid(alpha=0.2)
    ax_v.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True,
                framealpha=0.95, fontsize=8,
                title="lower is better\n(final < baseline = beats heuristic)")
    _stage_labels(ax_v, spans)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"cost figure ({variant}) -> {out_path}")


def render_combined(history_csv, baseline_csv, out_path, smooth=15):
    """Single-axes overview: train result, val result and the clean-LC baseline
    on one plot, with a short horizontal legend below the figure."""
    history = pd.read_csv(history_csv)
    baseline = pd.read_csv(baseline_csv)
    spans = stage_spans(history)
    clean_by_stage = dict(zip(baseline["curriculum_stage"], baseline["clean_lc_cost"]))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    _shade_stages(ax, spans)
    _plot_curve(ax, history, "train_final_cost", "train", "#0072bc", smooth)
    _plot_curve(ax, history, "val_final_cost", "val", "#f39c12", smooth)
    _plot_stage_step(ax, spans, clean_by_stage, CLEAN_LC_COLOR, "clean-LC")
    _stage_labels(ax, spans)
    ax.set_title("Mean route cost over training", fontsize=12)
    ax.set_xlabel("epochs")
    ax.set_ylabel("Mean cost")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=True, framealpha=0.95, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"cost figure (combined) -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="directory holding the run CSVs/SVGs")
    p.add_argument("--run", default=DEFAULT_RUN, help="run name prefix")
    p.add_argument("--variant", choices=["static", "mock", "combined", "both"],
                   default="both", help="which figure(s) to render (both = all three)")
    p.add_argument("--history", type=Path, help="override path to *_training_history_partial.csv")
    p.add_argument("--baseline", type=Path, help="override path to *_clean_lc_baseline_cost.csv")
    p.add_argument("--out", type=Path, help="output SVG (single variant only; default: <run>_cost_train_val_<variant>_repro.svg)")
    p.add_argument("--smooth", type=int, default=15, help="centred rolling-mean window for train / mock curves (<=1 disables)")
    p.add_argument("--per-graph-csv", type=Path, help="override path to *_val_seed_per_graph.csv")
    p.add_argument("--mock-seed", type=int, default=0, help="RNG seed for the resampled-weights mock")
    p.add_argument("--preedit-csv", type=Path, help="override path to *_train_seed_preedit.csv")
    args = p.parse_args()

    history_csv = args.history or args.dir / f"{args.run}_training_history_partial.csv"
    baseline_csv = args.baseline or args.dir / f"{args.run}_clean_lc_baseline_cost.csv"
    per_graph_csv = args.per_graph_csv or args.dir / f"{args.run}_val_seed_per_graph.csv"
    preedit_csv = args.preedit_csv or args.dir / f"{args.run}_train_seed_preedit.csv"

    variants = (["static", "mock", "combined"] if args.variant == "both"
                else [args.variant])
    if args.out and len(variants) > 1:
        raise SystemExit("--out can only be used with a single --variant")
    for variant in variants:
        out_path = args.out or args.dir / f"{args.run}_cost_train_val_{variant}_repro.svg"
        if variant == "combined":
            render_combined(history_csv, baseline_csv, out_path, smooth=args.smooth)
        else:
            render(history_csv, baseline_csv, out_path, variant, smooth=args.smooth,
                   per_graph_csv=per_graph_csv, mock_seed=args.mock_seed,
                   preedit_csv=preedit_csv)


if __name__ == "__main__":
    main()
