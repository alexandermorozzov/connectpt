"""Batched inference through a caller-provided environment and shared rollout."""
from dataclasses import dataclass

import torch
from torch.nn.functional import pad
from torch_geometric.loader import DataLoader

from ..city.graph import to_batch
from ..env.base import EnvParams
from ..network.routes import to_routes
from ..objectives.evaluate import Evaluation
from .generate import _generate
from .rollout import Trajectory, rollout


@dataclass(frozen=True)
class Prediction:
    routes: torch.Tensor
    evaluation: Evaluation
    trajectory: Trajectory | None = None


@torch.no_grad()
def predict(policy, graphs, preferences, cost, env_factory, limits: EnvParams,
            mode, n=1, *, seed_networks=None, reference_networks=None,
            route_idx=None, max_steps=None, mask=None, sample_batch_size=None,
            current_routes=None, revisit_routes=None):
    """Infer full networks or a selected route, optionally bounded by max_steps.

    Routes are [B,R,L], -1 padded, on the policy device. Each prediction owns
    its graph/state/seed/preference tensors. No inputs are mutated. Sampling
    consumes the caller's RNG in the existing rollout order. Every module's
    train/eval mode is restored on success and failure; feature-normalization
    statistics follow the policy's existing frozen/unfrozen setting.
    env_factory uses the Env constructor contract; task rules stay in Env.
    Bounded and selected-route trajectories require n=1.
    """
    if mode not in ('greedy', 'sample') or n < 1:
        raise ValueError('prediction requires greedy/sample mode and n >= 1')
    bounded = max_steps is not None or route_idx is not None
    if bounded and n != 1:
        raise ValueError('bounded or selected-route prediction requires n=1')
    if max_steps is not None and max_steps < 0:
        raise ValueError('max_steps must be nonnegative')
    if route_idx is not None and seed_networks is None:
        raise ValueError('a selected route requires a seed network')
    device = next(policy.parameters()).device
    data = to_batch(graphs, device).clone()
    prefs = {name: torch.as_tensor(value, device=device).clone() for name, value in preferences.items()}
    seed = None if seed_networks is None else to_routes(seed_networks).to(device).expand(data.num_graphs, -1, -1).clone()
    reference = None if reference_networks is None else to_routes(reference_networks).to(device).expand(data.num_graphs, -1, -1).clone()
    current = None if current_routes is None else current_routes.clone().to(device)
    revisit = None if revisit_routes is None else to_routes(revisit_routes).to(device).clone()
    modes = [(module, module.training) for module in policy.modules()]
    try:
        policy.eval()
        env = env_factory(cost, policy.feature_set, limits, policy.action_spec,
                          node_normalizer=policy.node_norm, mask=mask)
        if bounded:
            reset_kwargs = dict(normalize=False)
            if route_idx is not None:
                reset_kwargs.update(route_idx=route_idx, reference_networks=reference)
            env.reset(data, prefs, seed, **reset_kwargs)
            if reference is not None:
                env.state.set_adjustment_seed(reference)
            if current is not None:
                env.state.set_current_routes(current)
            env.bind(env.state, single_route=route_idx is not None or env.action_spec.role == 'edit')
            trajectory = rollout(env, policy, mode == 'greedy', max_steps=max_steps, record_states=False)
            routes = env.updated_network() if route_idx is not None or env.action_spec.role == 'edit' else env.state.adjustment_routes()
            return Prediction(routes, env.result, trajectory)
        env.reset(data, prefs, seed if env.action_spec.role == 'edit' else None, normalize=False)
        if reference is not None:
            env.state.set_adjustment_seed(reference)
        routes = _generate(env, policy, mode, n, env_factory=env_factory,
                           seed_networks=seed, sample_batch_size=sample_batch_size,
                           current_routes=current, revisit_routes=revisit)
        return Prediction(routes, env.result)
    finally:
        for module, training in modes:
            module.train(training)


@torch.no_grad()
def evaluate_policy(env, policy, graphs, prefs, mode, n=1, *, seed_networks=None):
    """Evaluate batches through predict without changing the caller's environment."""
    batches = graphs if isinstance(graphs, DataLoader) else (graphs,)
    predictions = [predict(policy, data, prefs, env.cost, type(env), env.params, mode, n,
                           seed_networks=seed_networks, mask=env.mask_override) for data in batches]
    if not predictions:
        raise ValueError('evaluation requires at least one graph batch')
    n_routes = max(value.routes.shape[1] for value in predictions)
    width = max(value.routes.shape[2] for value in predictions)
    routes = torch.cat([pad(value.routes, (0, width - value.routes.shape[2],
                                          0, n_routes - value.routes.shape[1]), value=-1)
                        for value in predictions])
    return Evaluation.concatenate([value.evaluation for value in predictions]), routes
