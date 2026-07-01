"""EKB (Ekaterinburg) case study: NBCO run + alpha sweep.

One-off paper experiment kept out of the reusable library (like macsa /
training_lc). The data-loading / crop / plotting primitives already live in
``eval_lib.ekb``; this module holds the orchestration that used to sit inline in
``paper_combined.ipynb`` -- the active-case loader, route scoring, the metric
rows, and the two runners (single NBCO run, alpha sweep). Config comes from the
suite profile (``SUITE.ekb``); nothing here owns model state (the BCO edit bees
load the checkpoint via eval_lib's default path, set in the notebook).
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any


@dataclass
class EKBCase:
    """The active EKB instance (full network or a geographic crop)."""
    tensors: dict
    init: Any        # route tensor (seed network)
    spec: dict
    crop_meta: dict
    case_tag: str


def load_ekb_case(*, use_crop=False, crop_target_nodes=300, crop_min_route_len=11,
                  crop_keep_largest=True) -> EKBCase:
    """Load the EKB instance, optionally cropped to a geographic subgraph.

    Unifies the two identical ``_load_active_ekb_case`` / ``_ekb_sweep_load_case``
    helpers that used to be duplicated across the notebook's EKB cells.
    """
    from eval_lib import as_route_tensor
    from eval_lib.ekb import (ekb_spec, load_ekb_routes, load_ekb_tensors,
                              make_ekb_crop_case)

    full_tensors = load_ekb_tensors()
    full_init = load_ekb_routes()
    if use_crop:
        tensors, init_routes, crop_meta = make_ekb_crop_case(
            full_tensors, full_init, target_nodes=crop_target_nodes,
            min_route_len=crop_min_route_len, keep_largest_component=crop_keep_largest)
        case_tag = f"crop{crop_meta['n_nodes']}"
    else:
        tensors, init_routes = full_tensors, full_init
        crop_meta = {"crop_enabled": False}
        case_tag = "full"
    return EKBCase(tensors, as_route_tensor(init_routes), ekb_spec(init_routes),
                   crop_meta, case_tag)


def case_from_cfg(ekb_cfg) -> EKBCase:
    """Build the active EKB case from the suite's ``ekb`` config block."""
    return load_ekb_case(
        use_crop=bool(ekb_cfg.use_crop),
        crop_target_nodes=int(ekb_cfg.crop_target_nodes),
        crop_min_route_len=int(ekb_cfg.crop_min_route_len),
        crop_keep_largest=bool(ekb_cfg.crop_keep_largest_component))


def nbco_adj_target(ekb_cfg):
    """Resolve the NBCO adjustment target (null in cfg -> unified ADJ_TARGET)."""
    from connectpt.routes_generator.objectives import load_unified_objective
    t = ekb_cfg.nbco.get("adj_target")
    return float(t) if t is not None else float(load_unified_objective().adj_target)


def score_routes(case: EKBCase, routes, tag, *, adj_target, force_cpu,
                 alpha=None, seed_routes=None):
    """Score a fixed route set under the unified objective (adjustment ON).

    Unifies ``_ekb_score_fixed_routes`` (NBCO) and ``_ekb_sweep_score`` (sweep):
    ``alpha`` optionally overrides the route/connectivity weights; ``seed_routes``
    defaults to the case's initial network.
    """
    from eval_lib.baselines import _run_baseline
    from eval_lib.paper import (UNIFIED_ADJ, eval_routes_cfg, set_cfg_value,
                                unify_weights)
    from connectpt.routes_generator.objectives import load_unified_objective
    CONNECTIVITY_MODE = load_unified_objective().connectivity_mode

    cfg = unify_weights(eval_routes_cfg("EKB", case.spec))
    set_cfg_value(cfg, "run_name", f"EKB_{case.case_tag}_{tag}")
    set_cfg_value(cfg, "experiment.cpu", bool(force_cpu))
    set_cfg_value(cfg, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
    if alpha is not None:
        set_cfg_value(cfg, "experiment.cost_function.kwargs.route_time_weight", float(alpha))
        set_cfg_value(cfg, "experiment.cost_function.kwargs.median_connectivity_weight",
                      float(1.0 - alpha))
    return _run_baseline(
        None, cfg, routes, f"EKB_{case.case_tag}_{tag}_", {}, tensors=case.tensors,
        use_weighted_connectivity=True, connectivity_mode=CONNECTIVITY_MODE,
        adjustment_seed_routes=case.init if seed_routes is None else seed_routes,
        **dict(UNIFIED_ADJ, adjustment_degree_target=float(adj_target)))


def _row_base(case, method, source, metrics, routes, *, adj_target, n_iterations,
              n_bees, seq_bees, duration_s=None):
    from eval_lib import as_route_tensor
    from eval_lib.paper import paper_row

    routes = as_route_tensor(routes)
    row = paper_row("EKB", method, source, metrics, routes, case.init,
                    duration_s=duration_s)
    row.update(ekb_case=case.case_tag,
               n_nodes=int(case.tensors["node_locs"].shape[0]),
               adj_target=float(adj_target),
               n_iterations=int(n_iterations),
               n_bees=int(n_bees),
               process_neural_bees_sequentially=bool(seq_bees))
    return row


def nbco_row(case, method, source, metrics, routes, *, adj_target, n_iterations,
             n_bees, seq_bees, duration_s=None, seed_cost=None):
    import numpy as np

    row = _row_base(case, method, source, metrics, routes, adj_target=adj_target,
                    n_iterations=n_iterations, n_bees=n_bees, seq_bees=seq_bees,
                    duration_s=duration_s)
    row["n_routes_active"] = int(case.spec["n_routes"])
    if seed_cost is None:
        row["seed_cost"] = float(row["cost"])
        row["cost_delta"] = 0.0
        row["cost_delta_pct"] = 0.0
    else:
        row["seed_cost"] = float(seed_cost)
        row["cost_delta"] = float(row["cost"] - seed_cost)
        row["cost_delta_pct"] = (100.0 * row["cost_delta"] / abs(seed_cost)
                                 if abs(seed_cost) > 1e-12 else np.nan)
    return row


def sweep_row(case, method, source, metrics, routes, *, alpha, adj_target,
              n_iterations, n_bees, seq_bees, duration_s=None, seed_cost=None):
    row = _row_base(case, method, source, metrics, routes, adj_target=adj_target,
                    n_iterations=n_iterations, n_bees=n_bees, seq_bees=seq_bees,
                    duration_s=duration_s)
    row["alpha"] = float(alpha)
    if seed_cost is not None and "cost" in row:
        row["seed_cost"] = float(seed_cost)
        row["delta_cost"] = float(row["cost"]) - float(seed_cost)
    return row


def metric_subtitle(df, label):
    """Per-panel metric caption for the NBCO before/after figures."""
    import numpy as np

    if df.empty or "method" not in df:
        return ""
    sub = df[df["method"] == label]
    if sub.empty:
        return ""
    r = sub.iloc[0]
    return (f"cost={r['cost']:.3f}  d={r.get('cost_delta', np.nan):.3f}  "
            f"seed={r.get('seed_cost', np.nan):.3f}\n"
            f"RTT={r['RTT']:.0f}  WMC={r['WMC']:.2f}  adj={r['adj_vs_seed']:.2f}  "
            f"d_un={r['d_un']:.1f}%  redun={r['redun%']:.0f}%")


def nbco_table_stem(case, ekb_cfg):
    seq = "_seqbees" if bool(ekb_cfg.nbco.process_neural_bees_sequentially) else ""
    crop = f"_crop{int(ekb_cfg.crop_target_nodes)}" if bool(ekb_cfg.use_crop) else ""
    return f"final_ekb{crop}_nbco_gnn_trimextend_iter{int(ekb_cfg.nbco.iterations)}{seq}"


def sweep_table_stem(case, ekb_cfg):
    crop = f"_crop{int(ekb_cfg.crop_target_nodes)}" if bool(ekb_cfg.use_crop) else ""
    tgt = str(ekb_cfg.sweep.adj_target).replace(".", "p")
    return (f"final_ekb{crop}_alpha_sweep_target{tgt}_iter{int(ekb_cfg.sweep.iterations)}")


@dataclass
class EKBRunResult:
    df: Any
    results: dict
    history: dict
    mutation_counts: dict
    table_stem: str


def run_ekb_nbco(case: EKBCase, *, ekb_cfg, our_model_cfg, n_bees, seed,
                 model_outputs_dir) -> EKBRunResult:
    """Single NBCO run (GNN rebuild + trim/extend) on the active EKB case.

    ``our_model_cfg`` is the notebook's context-bound ``experiments.our_model_cfg``
    partial. Reads/writes the paper cache (honouring the TEMP_ prefix); saves a
    live best-solution checkpoint on every improvement. Returns the result df,
    the {label: routes} mapping, and the convergence history.
    """
    import pandas as pd
    import torch

    import eval_lib.paper as _paper
    from eval_lib import as_route_tensor, run_bco
    from eval_lib.paper import (PAPER_DIR, UNIFIED_ADJ, append_paper_row,
                                bco_cfg_set, ravel_hist, reset_paper_table,
                                save_paper_routes, save_paper_table, set_cfg_value)

    ekb = ekb_cfg
    adj_target = nbco_adj_target(ekb)
    seq = bool(ekb.nbco.process_neural_bees_sequentially)
    n_iterations = int(ekb.nbco.iterations)
    final_label = str(ekb.nbco.final_label)
    force_cpu = bool(ekb.force_cpu)

    table = nbco_table_stem(case, ekb)
    routes_path = PAPER_DIR / f"{_paper.PAPER_PREFIX}{table}_routes.pt"
    csv_path = PAPER_DIR / f"{_paper.PAPER_PREFIX}{table}.csv"
    results = {"Initial EKB routes": case.init}

    if routes_path.exists() and csv_path.exists() and not bool(ekb.nbco.force_rerun):
        dump = torch.load(routes_path, weights_only=False)
        results = {k: as_route_tensor(v) for k, v in dump["routes"].items()}
        df = pd.read_csv(csv_path)
        print(f"[EKB NBCO] loaded cached routes/table: {routes_path.name}")
        return EKBRunResult(df, results, {}, {}, table)

    reset_paper_table(table)
    rows, history, mutation_counts = [], {}, {}
    print("[EKB NBCO] scoring initial EKB routes ...", flush=True)
    seed_res = score_routes(case, case.init, "nbco_seed_eval",
                            adj_target=adj_target, force_cpu=force_cpu)
    seed_row = nbco_row(case, "Initial EKB routes", f"ekb_{case.case_tag}_nbco_seed",
                        seed_res[1], case.init, adj_target=adj_target,
                        n_iterations=n_iterations, n_bees=n_bees, seq_bees=seq)
    rows.append(seed_row)
    append_paper_row(seed_row, table, ndigits=4)

    print(f"[EKB NBCO] running {final_label} on {case.case_tag}: "
          f"iters={n_iterations}, target={adj_target} ...", flush=True)
    cfg = our_model_cfg("EKB", case.spec, adj_target=adj_target,
                        run_name_suffix=f"{case.case_tag}_", seed=int(seed),
                        use_gnn=True, force_cpu=force_cpu)
    set_cfg_value(cfg, "experiment.cpu", bool(force_cpu))
    set_cfg_value(cfg, "process_neural_bees_sequentially", seq)
    bco_cfg_set(cfg, n_iterations=n_iterations,
                **dict(UNIFIED_ADJ, adjustment_degree_target=float(adj_target)))
    set_cfg_value(cfg, "experiment.cost_function.kwargs.use_weighted_connectivity", True)

    # Live checkpoint: overwrite final_ekb_best_solution with the current BCO
    # incumbent on every improvement, so stopping early never loses the best.
    best_stem = "final_ekb_best_solution"

    def _save_best(iteration, best_networks, best_costs, best_adj, best_metrics,
                   metric_names):
        best_rt = as_route_tensor(best_networks[0]).cpu()
        mrow = {str(n): float(v) for n, v in zip(metric_names, best_metrics[0].tolist())}
        mrow.update(method=final_label, checkpoint_iteration=int(iteration),
                    objective_cost=float(best_costs[0]), adj_vs_seed=float(best_adj[0]),
                    case_tag=case.case_tag, adj_target=float(adj_target),
                    n_iterations=n_iterations)
        save_paper_table(pd.DataFrame([mrow]).round(6), best_stem)
        save_paper_routes(best_stem,
                          {"Initial EKB routes": as_route_tensor(case.init),
                           final_label: best_rt},
                          meta={"city": "EKB", "best_method": final_label,
                                "checkpoint_iteration": int(iteration),
                                "case_tag": case.case_tag,
                                "objective_cost": float(best_costs[0])})
        if writer is not None:
            writer.add_scalar("ekb/objective_cost", float(best_costs[0]), int(iteration))
            writer.add_scalar("ekb/adj_vs_seed", float(best_adj[0]), int(iteration))
            for n, v in zip(metric_names, best_metrics[0].tolist()):
                writer.add_scalar(f"ekb/{n}", float(v), int(iteration))
            writer.flush()

    from torch.utils.tensorboard import SummaryWriter
    tb_dir = model_outputs_dir / "tensorboard" / f"ekb_{case.case_tag}_iter{n_iterations}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[EKB NBCO][tensorboard] -> {tb_dir}")

    t0 = _time.perf_counter()
    try:
        res = run_bco(cfg, case.init, tensors=case.tensors,
                      run_name_scope=f"EKB_{case.case_tag}_",
                      cost_history_out=history, mutation_counts_out=mutation_counts,
                      iteration_callback=_save_best)
    except KeyboardInterrupt:
        print(f"[EKB NBCO] interrupted -- best-so-far kept in {best_stem}.*")
        raise
    dt = _time.perf_counter() - t0
    writer.close()

    out_routes = as_route_tensor(res[3])
    row_out = nbco_row(case, final_label, f"ekb_{case.case_tag}_nbco_gnn_trimextend",
                       res[1], out_routes, adj_target=adj_target,
                       n_iterations=n_iterations, n_bees=n_bees, seq_bees=seq,
                       duration_s=dt, seed_cost=seed_row["cost"])
    rows.append(row_out)
    results[final_label] = out_routes
    append_paper_row(row_out, table, ndigits=4)
    torch.save({"history": ravel_hist(history.get("history")),
                "mutation_counts": mutation_counts},
               PAPER_DIR / f"{_paper.PAPER_PREFIX}{table}_history.pt")

    df = pd.DataFrame(rows)
    save_paper_table(df.round(4), table)
    save_paper_routes(table, results, case.tensors["node_locs"],
                      case.tensors["street_adj"],
                      meta={"city": "EKB", "case_tag": case.case_tag,
                            "crop": case.crop_meta,
                            "objective": "NBCO GNN + trim/extend",
                            "adj_target": float(adj_target),
                            "n_iterations": n_iterations,
                            "process_neural_bees_sequentially": seq})
    return EKBRunResult(df, results, history, mutation_counts, table)


def run_ekb_alpha_sweep(case: EKBCase, *, ekb_cfg, our_model_cfg, n_bees, seed):
    """Alpha sweep (route/connectivity trade-off) at a fixed adjustment target.

    Returns ``(df, {label: routes})``. Honours the TEMP_ prefix + cache like the
    NBCO runner. ``our_model_cfg`` is the notebook's context-bound partial.
    """
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    import eval_lib.paper as _paper
    from eval_lib import as_route_tensor, run_bco
    from eval_lib.paper import (PAPER_DIR, UNIFIED_ADJ, append_paper_row,
                                bco_cfg_set, reset_paper_table, save_paper_routes,
                                save_paper_table, set_cfg_value)

    ekb = ekb_cfg
    adj_target = float(ekb.sweep.adj_target)
    n_iterations = int(ekb.sweep.iterations)
    alpha_grid = list(ekb.sweep.alpha_grid)
    seq = bool(ekb.sweep.process_neural_bees_sequentially)
    force_cpu = bool(ekb.force_cpu)

    table = sweep_table_stem(case, ekb)
    routes_path = PAPER_DIR / f"{_paper.PAPER_PREFIX}{table}_routes.pt"
    csv_path = PAPER_DIR / f"{_paper.PAPER_PREFIX}{table}.csv"

    if routes_path.exists() and csv_path.exists() and not bool(ekb.sweep.force_rerun):
        payload = torch.load(routes_path, map_location="cpu", weights_only=False)
        routes = {k: as_route_tensor(v) for k, v in payload.get("routes", {}).items()}
        df = pd.read_csv(csv_path)
        print(f"[EKB sweep] loaded cache: {csv_path.name}")
        return df, routes

    print(f"[EKB sweep] case={case.case_tag} spec={case.spec} alphas={alpha_grid} "
          f"target={adj_target} iters={n_iterations} cpu={force_cpu}")
    reset_paper_table(table)
    rows = []
    routes = {"Initial EKB routes": case.init}

    def _meta():
        return {"city": "EKB", "case_tag": case.case_tag, "crop": case.crop_meta,
                "alpha_grid": alpha_grid, "adj_target": float(adj_target),
                "n_iterations": n_iterations,
                "process_neural_bees_sequentially": seq}

    for alpha in tqdm(alpha_grid, desc="EKB alpha sweep"):
        seed_res = score_routes(case, case.init, f"seed_alpha{alpha:g}",
                                adj_target=adj_target, force_cpu=force_cpu,
                                alpha=alpha, seed_routes=case.init)
        seed_row = sweep_row(case, "Initial EKB routes",
                             f"ekb_{case.case_tag}_seed_alpha{alpha:g}",
                             seed_res[1], seed_res[3], alpha=alpha,
                             adj_target=adj_target, n_iterations=n_iterations,
                             n_bees=n_bees, seq_bees=seq)
        rows.append(seed_row)
        append_paper_row(seed_row, table, ndigits=4)

        cfg = our_model_cfg("EKB", case.spec, adj_target=adj_target,
                            run_name_suffix=f"{case.case_tag}_alpha{alpha:g}_",
                            seed=int(seed), use_gnn=True, force_cpu=force_cpu)
        set_cfg_value(cfg, "experiment.cpu", bool(force_cpu))
        set_cfg_value(cfg, "experiment.cost_function.kwargs.route_time_weight", float(alpha))
        set_cfg_value(cfg, "experiment.cost_function.kwargs.median_connectivity_weight",
                      float(1.0 - alpha))
        set_cfg_value(cfg, "experiment.cost_function.kwargs.use_weighted_connectivity", True)
        set_cfg_value(cfg, "process_neural_bees_sequentially", seq)
        bco_cfg_set(cfg, n_iterations=n_iterations,
                    **dict(UNIFIED_ADJ, adjustment_degree_target=float(adj_target)))
        t0 = _time.perf_counter()
        _run_name, _metrics, _unserved, out_routes, _mut = run_bco(
            cfg, case.init, tensors=case.tensors,
            run_name_scope=f"EKB_{case.case_tag}_alpha{alpha:g}_")
        dt = _time.perf_counter() - t0
        out_routes = as_route_tensor(out_routes)
        routes[f"Our NBCO alpha={alpha:g}"] = out_routes
        score_res = score_routes(case, out_routes, f"our_alpha{alpha:g}",
                                 adj_target=adj_target, force_cpu=force_cpu,
                                 alpha=alpha, seed_routes=case.init)
        row_out = sweep_row(case, "Our NBCO (GNN + trim/extend)",
                            f"ekb_{case.case_tag}_our_alpha{alpha:g}",
                            score_res[1], score_res[3], alpha=alpha,
                            adj_target=adj_target, n_iterations=n_iterations,
                            n_bees=n_bees, seq_bees=seq, duration_s=dt,
                            seed_cost=seed_row.get("cost"))
        rows.append(row_out)
        append_paper_row(row_out, table, ndigits=4)
        save_paper_routes(table, routes, case.tensors["node_locs"],
                          case.tensors["street_adj"], meta=_meta())

    df = pd.DataFrame(rows)
    save_paper_table(df.round(4), table)
    save_paper_routes(table, routes, case.tensors["node_locs"],
                      case.tensors["street_adj"], meta=_meta())
    print(f"[EKB sweep] saved -> {csv_path.name} + {routes_path.name}")
    return df, routes
