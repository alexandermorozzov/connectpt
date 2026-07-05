# -*- coding: utf-8 -*-
"""Golden-output harness for BCO optimization work.

capture: run 3 BCO variants with fixed seeds, save cost histories + routes +
         metrics to artifacts/results/_golden_bco.pt
compare: rerun and assert bit-identical outputs vs the saved golden file.

Usage: python _golden_bco.py capture|compare [tag]
"""
import sys, time, random
import torch

from eval_lib.baselines import load_benchmark_tensors, BENCHMARK_SPECS
from eval_lib.experiments import compose_experiment_cfg
from eval_lib.context import EDIT_MODEL_WEIGHTS_DIR, ARTIFACTS_DIR
from eval_lib.paper import UNIFIED_ADJ
from connectpt.routes_generator.search.cfg_run import run_bco_from_cfg

EDIT_WEIGHTS_PATH = EDIT_MODEL_WEIGHTS_DIR / \
    "improvement_lc_rttconn_adj_w10_t02_finetune100.pt"

GOLDEN_PATH = ARTIFACTS_DIR / "results" / "_golden_bco.pt"
N_ITER = 3

# Each variant loads its captured config-first YAML (nbco_variants/*); the bee
# split, cost weights, connectivity mode and adjustment block are baked into the
# YAML (byte-identical to the old build_bco_cfg output). run_variant applies only
# the per-run overrides (route bounds, n_iterations, seed).
VARIANTS = [
    ("classical_Mumford0", "Mumford0", "nbco_variants/classic_bco_mumford0"),
    ("classical_Mandl",    "Mandl",    "nbco_variants/classic_bco_mandl"),
    ("neural_Mumford0",    "Mumford0", "nbco_variants/neural_bco_mumford0"),
    ("our_Mumford0",       "Mumford0", "nbco_variants/our_nbco_mumford0"),
]


def _init_routes(city, spec, tensors):
    rng = random.Random(0)
    n = tensors["node_locs"].shape[0]
    adj = tensors["street_adj"]
    nbrs = {u: [v for v in range(n) if v != u and torch.isfinite(adj[u, v])]
            for u in range(n)}
    routes, uncov = [], set(range(n))
    for _ in range(spec["n_routes"]):
        start = rng.choice(sorted(uncov)) if uncov else rng.randrange(n)
        path = [start]
        while len(path) < spec["max_route_len"]:
            c = [v for v in nbrs[path[-1]] if v not in path]
            if not c:
                break
            pref = [v for v in c if v in uncov]
            path.append(rng.choice(pref or c))
        routes.append(path)
        uncov -= set(path)
    R = torch.full((1, spec["n_routes"], spec["max_route_len"]), -1, dtype=torch.long)
    for i, p in enumerate(routes):
        R[0, i, :len(p)] = torch.tensor(p)
    return R


def run_variant(name, city, yaml_name):
    spec = next(s for s in BENCHMARK_SPECS if s["city"] == city)
    tensors = load_benchmark_tensors(city)
    R = _init_routes(city, spec, tensors)
    cfg = compose_experiment_cfg(
        yaml_name, bounds=spec, run_name=f"golden_{name}", seed=0,
        n_iterations=N_ITER, adj=UNIFIED_ADJ, weighted_connectivity=True)
    torch.manual_seed(0)
    random.seed(0)
    ch = {}
    t0 = time.perf_counter()
    out = run_bco_from_cfg(cfg, R, tensors, run_name_scope="golden_",
                           cost_history_out=ch,
                           edit_weights_path=EDIT_WEIGHTS_PATH,
                           edit_n_adjustment_cond_feats=0)
    dt = time.perf_counter() - t0
    _, metrics, unserved, routes, _ = out
    # wall-clock duration is the only legitimately nondeterministic metric
    metrics = {k: v for k, v in (metrics or {}).items()
               if "duration" not in k}
    return {
        "unserved": (unserved.detach().cpu() if torch.is_tensor(unserved)
                     else unserved),
        "metrics": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in metrics.items()},
        "routes": (routes.detach().cpu() if torch.is_tensor(routes)
                   else torch.as_tensor(routes)),
        "history": (ch.get("history").detach().cpu()
                    if torch.is_tensor(ch.get("history")) else ch.get("history")),
        "seconds": dt,
    }


def _eq(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and \
            a.shape == b.shape and bool(torch.equal(a, b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, float):
        return a == b or (a != a and b != b)   # NaN == NaN for our purpose
    return a == b


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    results = {}
    for name, city, yaml_name in VARIANTS:
        results[name] = run_variant(name, city, yaml_name)
        print(f"[{name}] {results[name]['seconds']:.1f}s")
    if mode == "capture":
        torch.save(results, GOLDEN_PATH)
        print(f"golden saved -> {GOLDEN_PATH}")
        return
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=False)
    n_bad = 0
    for name in golden:
        g, r = golden[name], results[name]
        bad_here = 0
        for field in ("unserved", "metrics", "routes", "history"):
            ok = _eq(g[field], r[field])
            if not ok:
                bad_here += 1
                print(f"MISMATCH {name}.{field}")
                if torch.is_tensor(g[field]) and torch.is_tensor(r[field]) \
                        and g[field].shape == r[field].shape:
                    diff = (g[field].float() - r[field].float()).abs()
                    print(f"  max abs diff: {diff.max().item():.3e}")
        n_bad += bad_here
        speed = golden[name]["seconds"] / max(results[name]["seconds"], 1e-9)
        print(f"[{name}] {'OK' if bad_here == 0 else 'BAD'} | "
              f"{golden[name]['seconds']:.1f}s -> {results[name]['seconds']:.1f}s "
              f"({speed:.2f}x)")
    if n_bad:
        sys.exit(f"{n_bad} mismatches")
    print("ALL GOLDEN OUTPUTS IDENTICAL")


if __name__ == "__main__":
    main()
