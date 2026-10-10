"""Build the initial route network for a benchmark instance, once.

    python scripts/make_init_routes.py
    python scripts/make_init_routes.py --instances mandl --seed 1
    python scripts/make_init_routes.py --clean

Writes ``datasets/instances/<name>/Routes.txt`` (one route per line, 0-indexed).
Runs read that file and never build a network on the fly. Instances that ship
with their own initial network (MACSA scenarios, EKB) do not need this.
Config groups: ``cfg/runs/init_routes.yaml``.
"""
from __future__ import annotations
from connectpt.routes_generator.run.config import cost_params

import argparse

from omegaconf import OmegaConf

from connectpt.routes_generator.network.corruption import RealisticParams
from connectpt.routes_generator.objectives.cost import build_cost
from connectpt.routes_generator.run.config import load_experiment, policy_profile
from connectpt.routes_generator.run.context import resolve_under_root, resolve_weights_path
from connectpt.routes_generator.run.context import resolve_device, seed_everything
from connectpt.routes_generator.policy.build import build_policy
from connectpt.routes_generator.run.schemas import InitialParams, params
from connectpt.routes_generator.search.initial import build_init_routes
from connectpt.routes_generator.run.instances import load_instance
from connectpt.routes_generator.run.const import INSTANCES_DIR, MODEL_WEIGHTS_DIR
from connectpt.routes_generator.network.routes import write_routes_txt

INSTANCES = ["mandl", "mumford0", "mumford1", "mumford2", "mumford3"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", nargs="*", default=INSTANCES)
    parser.add_argument("--generator", choices=["lc", "john", "nikolic"], default="lc")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--instances-dir", default=None, help="default: datasets/instances")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--clean", action="store_true", help="construction sample only, without the realistic tier")
    args = parser.parse_args()

    root = resolve_under_root(args.instances_dir) if args.instances_dir else INSTANCES_DIR
    device = resolve_device(cpu=args.cpu)
    cfg = load_experiment("init_routes")
    network_seed = cfg.run.seed if args.seed is None else args.seed
    cost = build_cost(
        cost_params(cfg.cost), symmetric_routes=cfg.common.symmetric_routes, low_memory_mode=cfg.cost.low_memory_mode
    )
    initial = params(InitialParams, cfg.initial)
    corruption = RealisticParams.from_values(OmegaConf.to_container(cfg.corruption, resolve=True))

    for name in args.instances:
        d = root / name
        instance = load_instance(d, require_routes=False)
        spec = instance.spec
        seed_everything(network_seed)
        policy, binding, instance_cost = None, None, cost
        if args.generator == "lc":
            seed_everything(cfg.run.seed)
            policy = build_policy(policy_profile(
                cfg.model_profile.model, cfg.cost, cfg.common, device,
                checkpoint_path=resolve_weights_path(cfg.model_profile.checkpoint_path, MODEL_WEIGHTS_DIR),
            ))
            from connectpt.routes_generator.run.tasks import TASKS
            from connectpt.routes_generator.training.dataset import DatasetLimits
            limits = DatasetLimits(spec['n_routes'], spec['min_route_len'], spec['max_route_len'])
            binding = TASKS[cfg.model_profile.env.name].binding(
                policy, limits, OmegaConf.to_container(cfg.model_profile.env.params, resolve=True))
        routes = build_init_routes(
            spec,
            instance.tensors,
            generator=args.generator,
            cost=instance_cost,
            initial=initial,
            corruption=corruption,
            policy=policy,
            binding=binding,
            seed=network_seed,
            clean=args.clean,
            device=device,
        )
        tier = "construction only" if args.clean else "+ realistic tier"
        generator_label = "learned construction" if args.generator == "lc" else args.generator
        description = f"{generator_label} {tier}" if args.generator == "lc" else generator_label
        n = write_routes_txt(d / "Routes.txt", routes, f"{name} initial network -- {description}, seed {network_seed}")
        print(f"{name}: {n} routes -> {d / 'Routes.txt'}")


if __name__ == "__main__":
    main()
