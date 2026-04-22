import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from .citygraph_dataset import (
    STOP_KEY,
    DemandScaleTransform,
    InsertPosFeatures,
    SpaceScaleTransform,
)
from .transit_time_estimator import (
    ROUTE_ACTION_HALT,
    RouteGenBatchState,
)
from .torch_utils import get_batch_tensor_from_routes


def load_lc_result_routes(results_dir, n_graphs):
    """Load per-graph LC route tensors from examples/lc_results."""
    results_dir = Path(results_dir)
    route_sets = []
    for graph_idx in range(n_graphs):
        route_path = results_dir / f"graph_{graph_idx:04d}" / \
            f"lc_lc_graph_{graph_idx:04d}_routes_routes.pkl"
        with route_path.open("rb") as file:
            routes = pickle.load(file)
        if isinstance(routes, list):
            routes = routes[0]
        if routes.ndim == 3 and routes.shape[0] == 1:
            routes = routes.squeeze(0)
        route_sets.append(routes.long())
    return torch.stack(route_sets, dim=0)


def load_raw_graphs_and_lc_routes(raw_graphs_path, lc_results_dir):
    with Path(raw_graphs_path).open("rb") as file:
        graphs = pickle.load(file)
    seed_routes = load_lc_result_routes(lc_results_dir, len(graphs))
    return graphs, seed_routes


def make_improvement_batch(graphs, seed_routes, indices, device, training=False,
                           space_scale=None, demand_scale=None,
                           insert_pos=None):
    if space_scale is None:
        space_scale = SpaceScaleTransform(0.95, 1.05)
    if demand_scale is None:
        demand_scale = DemandScaleTransform(0.9, 1.1)
    if insert_pos is None:
        insert_pos = InsertPosFeatures()

    batch_graphs = []
    for idx in indices:
        graph = graphs[int(idx)].clone()
        if training:
            graph = space_scale(graph)
            graph = demand_scale(graph)
        graph[STOP_KEY].x = graph[STOP_KEY].pos.clone()
        graph = insert_pos(graph)
        batch_graphs.append(graph)

    graph_batch = Batch.from_data_list(batch_graphs).to(device)
    route_batch = seed_routes[torch.as_tensor(indices, dtype=torch.long)].to(
        device)
    return graph_batch, route_batch


def _clone_cost_weights(cost_weights):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in cost_weights.items()
    }


def rollout_lc_improvement(model, cost_obj, graph_batch, route_batch,
                           min_route_len, max_route_len, greedy=False,
                           cost_weights=None, return_actions=False):
    if cost_weights is None:
        cost_weights = cost_obj.sample_variable_weights(graph_batch.num_graphs,
                                                        graph_batch[STOP_KEY].x.device)

    n_routes = route_batch.shape[1]
    seed_state = RouteGenBatchState(
        graph_batch, cost_obj, n_routes, min_route_len, max_route_len,
        cost_weights=_clone_cost_weights(cost_weights))
    seed_state.add_new_routes(route_batch)
    seed_result = cost_obj(seed_state)

    state = RouteGenBatchState(
        graph_batch, cost_obj, n_routes, min_route_len, max_route_len,
        cost_weights=_clone_cost_weights(cost_weights))

    route_logits = []
    route_entropies = []
    route_actions = []
    route_action_kinds = []
    for route_idx in range(n_routes):
        state.set_current_routes(route_batch[:, route_idx])
        actions, logits, entropy = model.plan_new_route(state, greedy=greedy)
        route_logits.append(logits)
        route_entropies.append(entropy)
        if return_actions:
            route_actions.append(actions.detach().clone())
            if hasattr(model, "last_route_action_kinds"):
                route_action_kinds.append(
                    model.last_route_action_kinds.detach().clone())
            else:
                route_action_kinds.append(None)

    final_result = cost_obj(state)
    output = (
        state,
        seed_result,
        final_result,
        torch.stack(route_logits, dim=1),
        torch.stack(route_entropies, dim=1),
    )
    if return_actions:
        output = output + (route_actions, route_action_kinds)
    return output


def _backward_on_sampled_actions(model, cost_obj, graph_batch, route_batch,
                                 min_route_len, max_route_len, cost_weights,
                                 route_actions, route_action_kinds,
                                 advantages, entropy_weight):
    """Replay sampled actions and backprop before mutating the state.

    The sampled rollout mutates RouteGenBatchState many times.  Backpropagating
    through a graph that still references those mutable LongTensor indices can
    trigger version-counter errors.  This helper replays fixed actions and calls
    backward on each step before applying that step to the replay state.
    """
    n_routes = route_batch.shape[1]
    state = RouteGenBatchState(
        graph_batch, cost_obj, n_routes, min_route_len, max_route_len,
        cost_weights=_clone_cost_weights(cost_weights))
    supports_route_actions = getattr(model, "supports_trim_actions", False)
    objective_values = []

    for route_idx in range(n_routes):
        state.set_current_routes(route_batch[:, route_idx])
        state = model.setup_planning(state)
        actions_for_route = route_actions[route_idx]
        kinds_for_route = route_action_kinds[route_idx]
        ended = torch.zeros((state.batch_size,), dtype=torch.bool,
                            device=state.device)

        for step_idx in range(actions_for_route.shape[1]):
            if ended.all():
                break

            sampled_actions = actions_for_route[:, step_idx].clone()
            if supports_route_actions:
                sampled_kinds = kinds_for_route[:, step_idx].clone()
                active = ~ended
                score_mask = active & (sampled_kinds != ROUTE_ACTION_HALT)
                if score_mask.any():
                    score_idxs = torch.where(score_mask)[0]
                    score_state = state.index_select(score_idxs)
                    score_state = model.setup_planning(score_state)
                    _, _, logits, entropy = model.step_route_action(
                        score_state, greedy=False,
                        actions=sampled_actions[score_mask],
                        action_kinds=sampled_kinds[score_mask])
                    objective = \
                        (advantages[score_mask] * logits).mean()
                    objective = objective + entropy_weight * entropy.mean()
                    (-objective).backward()
                    objective_values.append(objective.detach())

                apply_kinds = sampled_kinds.detach().clone()
                apply_actions = sampled_actions.detach().clone()
                apply_kinds[ended] = ROUTE_ACTION_HALT
                apply_actions[ended] = -1
                just_ended = apply_kinds == ROUTE_ACTION_HALT
                apply_actions[just_ended] = -1
                with torch.no_grad():
                    state.apply_route_actions(apply_kinds, apply_actions)
                ended = ended | just_ended
            else:
                active = ~ended
                sampled_halts = sampled_actions[:, 0] == -1
                score_mask = active & ~sampled_halts
                if score_mask.any():
                    score_idxs = torch.where(score_mask)[0]
                    score_state = state.index_select(score_idxs)
                    score_state = model.setup_planning(score_state)
                    _, logits, entropy = model.step(
                        score_state, greedy=False,
                        actions=sampled_actions[score_mask])
                    objective = \
                        (advantages[score_mask] * logits).mean()
                    objective = objective + entropy_weight * entropy.mean()
                    (-objective).backward()
                    objective_values.append(objective.detach())

                apply_actions = sampled_actions.detach().clone()
                apply_actions[ended] = -1
                just_ended = apply_actions[:, 0] == -1
                with torch.no_grad():
                    state.shortest_path_action(apply_actions)
                ended = ended | just_ended

    if len(objective_values) == 0:
        return torch.zeros((), device=graph_batch[STOP_KEY].x.device)
    return torch.stack(objective_values).mean()


@torch.no_grad()
def evaluate_lc_improvement(model, cost_obj, graphs, seed_routes, indices,
                            device, min_route_len, max_route_len,
                            batch_size=8):
    model.eval()
    seed_costs = []
    final_costs = []
    route_outputs = []
    eval_weights = cost_obj.get_weights(device)

    for batch_indices in tqdm(list(indices.split(batch_size)),
                              desc="eval", leave=False):
        graph_batch, route_batch = make_improvement_batch(
            graphs, seed_routes, batch_indices, device, training=False)
        state, seed_result, final_result, _, _ = rollout_lc_improvement(
            model, cost_obj, graph_batch, route_batch, min_route_len,
            max_route_len, greedy=True, cost_weights=eval_weights)
        seed_costs.append(seed_result.cost.cpu())
        final_costs.append(final_result.cost.cpu())
        route_outputs.append(get_batch_tensor_from_routes(state.routes).cpu())

    seed_costs = torch.cat(seed_costs)
    final_costs = torch.cat(final_costs)
    return {
        "seed_cost": seed_costs.mean().item(),
        "final_cost": final_costs.mean().item(),
        "delta": (seed_costs - final_costs).mean().item(),
        "win_rate": (final_costs < seed_costs).float().mean().item(),
        "routes": route_outputs,
    }


def train_lc_improvement(model, cost_obj, graphs, seed_routes, device,
                         output_dir, run_name,
                         train_fraction=0.9, batch_size=8, n_epochs=20,
                         lr=1e-4, weight_decay=1e-4, entropy_weight=1e-3,
                         min_route_len=2, max_route_len=8,
                         warmup_batches=4, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    all_indices = torch.randperm(len(graphs))
    train_size = int(train_fraction * len(all_indices))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    model.train()
    with torch.no_grad():
        warmup = list(train_indices.split(batch_size))[:warmup_batches]
        for batch_indices in tqdm(warmup, desc="feature norm"):
            graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, batch_indices, device, training=True)
            rollout_lc_improvement(model, cost_obj, graph_batch, route_batch,
                                   min_route_len, max_route_len, greedy=False)
    model.update_and_freeze_feature_norms()

    best_val_cost = float("inf")
    best_model_path = output_dir / f"{run_name}.pt"
    history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_indices = train_indices[torch.randperm(len(train_indices))]
        train_seed_costs = []
        train_final_costs = []

        for batch_indices in tqdm(list(epoch_indices.split(batch_size)),
                                  desc=f"epoch {epoch + 1}/{n_epochs}"):
            graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, batch_indices, device, training=True)
            cost_weights = cost_obj.sample_variable_weights(
                graph_batch.num_graphs, device)
            with torch.no_grad():
                _, seed_result, final_result, _, _, route_actions, \
                    route_action_kinds = rollout_lc_improvement(
                        model, cost_obj, graph_batch, route_batch,
                        min_route_len, max_route_len, greedy=False,
                        cost_weights=cost_weights, return_actions=True)

            improvement = seed_result.cost - final_result.cost
            advantages = improvement.detach() - improvement.detach().mean()
            if advantages.numel() > 1:
                advantages = advantages / (advantages.std() + 1e-8)

            optimizer.zero_grad()
            objective = _backward_on_sampled_actions(
                model, cost_obj, graph_batch, route_batch, min_route_len,
                max_route_len, cost_weights, route_actions,
                route_action_kinds, advantages, entropy_weight)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_seed_costs.append(seed_result.cost.detach().cpu())
            train_final_costs.append(final_result.cost.detach().cpu())

        train_seed = torch.cat(train_seed_costs).mean().item()
        train_final = torch.cat(train_final_costs).mean().item()
        val = evaluate_lc_improvement(
            model, cost_obj, graphs, seed_routes, val_indices, device,
            min_route_len, max_route_len, batch_size=batch_size)

        if val["final_cost"] < best_val_cost:
            best_val_cost = val["final_cost"]
            torch.save(model.state_dict(), best_model_path)

        row = {
            "epoch": epoch + 1,
            "train_seed_cost": train_seed,
            "train_final_cost": train_final,
            "train_delta": train_seed - train_final,
            "val_seed_cost": val["seed_cost"],
            "val_final_cost": val["final_cost"],
            "val_delta": val["delta"],
            "val_win_rate": val["win_rate"],
        }
        history.append(row)
        print(
            f"epoch={row['epoch']:02d} "
            f"train_seed={row['train_seed_cost']:.4f} "
            f"train_final={row['train_final_cost']:.4f} "
            f"train_delta={row['train_delta']:.4f} "
            f"val_seed={row['val_seed_cost']:.4f} "
            f"val_final={row['val_final_cost']:.4f} "
            f"val_delta={row['val_delta']:.4f} "
            f"win_rate={row['val_win_rate']:.2%}"
        )

    return {
        "best_model_path": best_model_path,
        "history": history,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
