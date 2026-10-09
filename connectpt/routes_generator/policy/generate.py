"""Internal network sampling for the public predict interface."""
import torch
from dataclasses import replace

from ..env.state import RouteGenBatchState
from ..network.routes import to_routes
from ..objectives.evaluate import Evaluation
from ..objectives.preferences import batch_value
from .rollout import rollout


def _seed(state, networks, current):
    if networks is not None:
        networks = to_routes(networks, n_routes=int(state.n_routes_to_plan.min().item()))
        networks = networks.expand(state.batch_size, -1, -1).to(state.device)
        if networks.shape[1]:
            state.add_new_routes(networks)
    if current is not None:
        current = to_routes(current, width=state.max_n_nodes).reshape(-1, state.max_n_nodes)
        current = current.expand(state.batch_size, -1).to(state.device)
        if (current >= 0).any():
            state.set_current_routes(current)


def _finish(env, policy, greedy, revisit, env_factory):
    if revisit is not None:
        revisit = to_routes(revisit, n_routes=int(env.state.n_routes_to_plan.min().item()))
        revisit = revisit.expand(env.state.batch_size, -1, -1).to(env.state.device)
        for route in revisit.unbind(1):
            has_route = (route >= 0).any(-1)
            if not has_route.any():
                continue
            if not has_route.all():
                raise NotImplementedError("revisiting requires equal route counts across a batch")
            env.state.set_current_routes(route)
            route_env = env_factory(env.cost, policy.feature_set,
                                    replace(env.params, max_trim_actions_per_route=1), policy.action_spec,
                                    node_normalizer=policy.node_norm, mask=env.mask_override)
            route_env.bind(env.state, single_route=True)
            rollout(route_env, policy, greedy, record_states=False)
    env.bind(env.state)
    if not env.state.is_done().all():
        rollout(env, policy, greedy, record_states=False)
    return env.state


@torch.no_grad()
def _generate(
    env,
    policy,
    mode,
    n=1,
    *,
    env_factory,
    sample_batch_size=None,
    seed_networks=None,
    current_routes=None,
    revisit_routes=None,
):
    """Generate greedily or select the cheapest of n independent samples."""
    if mode not in ("greedy", "sample") or n < 1:
        raise ValueError("generation requires greedy/sample mode and n >= 1")
    if env.action_spec.role == "edit":
        return _generate_edit(env, policy, mode, n, env.network if seed_networks is None else seed_networks)
    state = env.state
    if mode == "greedy":
        _seed(state, seed_networks, current_routes)
        _finish(env, policy, True, revisit_routes, env_factory)
        return to_routes(state.routes).to(state.device)
    graphs = state.graph_data.to_data_list() * n
    batch_size = state.batch_size
    if sample_batch_size is None:
        sample_batch_size = batch_size if batch_size > 1 else n
    if sample_batch_size < 1:
        raise ValueError("sample_batch_size must be positive")
    seeds = None if seed_networks is None else to_routes(seed_networks).repeat(n, 1, 1)
    revisits = None if revisit_routes is None else to_routes(revisit_routes).repeat(n, 1, 1)
    currents = None
    if current_routes is not None:
        currents = to_routes(current_routes, width=state.max_n_nodes).reshape(-1, state.max_n_nodes)
        currents = currents.expand(batch_size, -1).to(state.device).repeat(n, 1)
    all_routes, all_costs, evaluations = [], [], []
    for start in range(0, len(graphs), sample_batch_size):
        end = min(start + sample_batch_size, len(graphs))
        indices = torch.arange(start, end, device=state.device) % batch_size
        sample_state = RouteGenBatchState(
            graphs[start:end],
            env.cost.constants,
            state.n_routes_to_plan[indices],
            state.min_route_len[indices],
            state.max_route_len[indices],
            preferences={name: batch_value(value, batch_size, state.device)[indices].clone()
                         for name, value in state.cost_weights.items()},
        )
        if state.adjustment_seed.numel():
            sample_state.set_adjustment_seed(state.adjustment_seed[indices])
        _seed(
            sample_state, None if seeds is None else seeds[start:end], None if currents is None else currents[start:end]
        )
        sample_env = env_factory(env.cost, policy.feature_set, env.params, policy.action_spec,
                                node_normalizer=policy.node_norm, mask=env.mask_override)
        sample_env.bind(sample_state, normalize=False)
        _finish(sample_env, policy, False, None if revisits is None else revisits[start:end], env_factory)
        all_routes.extend(sample_state.routes)
        evaluations.append(sample_env.result)
        all_costs.append(sample_env.result.cost)
    costs = torch.cat(all_costs).reshape(n, batch_size)
    best = costs.argmin(0)
    routes = to_routes(all_routes).to(state.device)[best * batch_size + state.batch_indices]
    routes = routes[..., :int((routes >= 0).sum(-1).max())]
    state.replace_routes(routes.to(state.device))
    selected = Evaluation.concatenate(evaluations).select(best * batch_size + state.batch_indices)
    env.bind(state, normalize=False, evaluation=selected)
    return routes.to(state.device)


def _generate_edit(env, policy, mode, n, networks):
    original = env.state
    seed = to_routes(networks).to(original.device)
    reference = original.adjustment_seed if original.adjustment_seed.numel() else seed
    candidates, costs, evaluations = [], [], []
    for _ in range(n if mode == "sample" else 1):
        working = seed.clone()
        for route_idx in range(seed.shape[1]):
            env.reset(
                original.graph_data,
                original.cost_weights,
                working,
                route_idx=route_idx,
                normalize=False,
                reference_networks=reference,
            )
            env.bind(env.state, single_route=True)
            rollout(env, policy, mode == "greedy", record_states=False)
            working = env.updated_network()
        final = RouteGenBatchState(
            original.graph_data,
            env.cost.constants,
            seed.shape[1],
            original.min_route_len,
            original.max_route_len,
            preferences=original.cost_weights,
        )
        final.add_new_routes(working)
        final.set_adjustment_seed(reference)
        candidates.append(working)
        result = env.cost(final)
        evaluations.append(result)
        costs.append(result.cost)
    best = torch.stack(costs).argmin(0)
    routes = torch.stack(candidates)[best, original.batch_indices]
    final = RouteGenBatchState(
        original.graph_data,
        env.cost.constants,
        seed.shape[1],
        original.min_route_len,
        original.max_route_len,
        preferences=original.cost_weights,
    )
    final.add_new_routes(routes)
    final.set_adjustment_seed(reference)
    selected = Evaluation.concatenate(evaluations).select(best * original.batch_size + original.batch_indices)
    env.bind(final, normalize=False, evaluation=selected)
    return routes
