"""Build the initial route network for a benchmark instance, once.

    python scripts/make_init_routes.py
    python scripts/make_init_routes.py --instances mandl --seed 1
    python scripts/make_init_routes.py --clean

Writes ``datasets/instances/<name>/Routes.txt`` (one route per line, 0-indexed).
Runs read that file and never build a network on the fly. Instances that ship
with their own initial network (MACSA scenarios, EKB) do not need this.
"""
from __future__ import annotations

import argparse

from connectpt.routes_generator.run.context import resolve_under_root
from connectpt.routes_generator.run.context import resolve_device, seed_everything
from connectpt.routes_generator.data import (INSTANCES_DIR, build_init_routes,
                                             instance_spec, load_matrices,
                                             write_routes_txt)

INSTANCES = ["mandl", "mumford0", "mumford1", "mumford2", "mumford3"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", nargs="*", default=INSTANCES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instances-dir", default=None,
                        help="default: datasets/instances")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--clean", action="store_true",
                        help="construction sample only, without the realistic tier")
    args = parser.parse_args()

    root = (resolve_under_root(args.instances_dir) if args.instances_dir
            else INSTANCES_DIR)
    device = resolve_device(cpu=args.cpu)

    for name in args.instances:
        d = root / name
        cfg = instance_spec(name)
        spec = {"city": name, **{k: int(cfg[k]) for k in
                                 ("n_routes", "min_route_len", "max_route_len")}}
        seed_everything(args.seed)
        routes = build_init_routes(spec, load_matrices(d), seed=args.seed,
                                   device=device, clean=args.clean)
        tier = "construction only" if args.clean else "+ realistic tier"
        n = write_routes_txt(
            d / "Routes.txt", routes,
            f"{name} initial network -- learned construction {tier}, seed {args.seed}")
        print(f"{name}: {n} routes -> {d / 'Routes.txt'}")


if __name__ == "__main__":
    main()
