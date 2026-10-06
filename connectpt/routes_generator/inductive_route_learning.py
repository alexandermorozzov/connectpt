import copy
import math
import random
import logging as log
import time
from pathlib import Path
from itertools import product

from tqdm import tqdm
import torch
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
import optuna
from omegaconf import DictConfig, OmegaConf
import hydra

# need to import CityGraphData here to allow unpickling of new datasets
from .citygraph_dataset import CityGraphData, CityGraphDataset, \
    get_default_train_and_eval_split, get_dynamic_training_set, STOP_KEY, \
    DEMAND_KEY
from .transit_time_estimator import (COST_WEIGHT_KEY_ORDER,
                                     ROUTE_ACTION_EXTEND, RouteGenBatchState)
from . import utils as lrnu
from .models import FeatureNorm, get_mlp, WEIGHT_FEATURE_KEYS
from .eval_route_generator import eval_model


BLMODE_NONE = "none"
BLMODE_GREEDY = "greedy"
BLMODE_ROLL = "rolling average"
BLMODE_NN = "neural"


class Baseline:
    def update(self, mean_cost):
        raise NotImplementedError

    def get_baseline(self, graph_data, n_routes, cost_weights, sumwriter=None, 
                     n_eps=None):
        raise NotImplementedError
    

class FixedBaseline:
    def __init__(self, baseline=0, *args, **kwargs):
        self.baseline = baseline

    def update(self, costs):
        self.baseline = costs.mean()

    def get_baseline(self, graph_data, n_routes, cost_weights, sumwriter=None,
                     n_eps=None):
        if sumwriter is not None:
            sumwriter.add_scalar("baseline", self.baseline, n_eps)
        batch_size = graph_data.num_graphs
        return self.baseline[None, None].expand(batch_size, n_routes)
    

class RollingBaseline:
    def __init__(self, alpha):
        assert 0 < alpha <= 1
        self.alpha = alpha
        self.baseline = None

    def update(self, costs):
        mean_cost = costs.mean()
        if self.baseline is None:
            self.baseline = mean_cost
        else:
            self.baseline = self.alpha * mean_cost + (1 - self.alpha) * \
                self.baseline
    
    def get_baseline(self, graph_data, cost_weights, sumwriter=None, 
                     n_eps=None):
        if sumwriter is not None:
            if self.baseline is None:
                sumwriter.add_scalar("baseline", 0, n_eps)
            else:
                sumwriter.add_scalar("baseline", self.baseline, n_eps)

        if self.baseline is None:
            return 0
        else:
            return self.baseline


class NNBaseline:
    """Scalar value baseline.

    When ``actor_model`` is provided and exposes ``get_critic_features``,
    the baseline shares the actor's GNN encoder (stop-gradient) and
    consumes pooled node embeddings instead of hand-crafted summary
    scalars — a far richer, edit-aware state representation. The value
    head input dim is then known only at first forward, so the head is
    built lazily. When ``actor_model`` is None it uses the legacy
    21-feature input (unchanged for construction PPO).
    """

    # Feature specs (checkpoint-locked, declared like on the actor models):
    # order of the cost-weight slots inside the shared global state features,
    # and inside this baseline's own legacy input tail (slot -1-ii = key ii).
    weight_feature_keys = WEIGHT_FEATURE_KEYS
    weight_input_keys = COST_WEIGHT_KEY_ORDER

    def __init__(self, learning_rate=0.0005, decay=0.01, actor_model=None,
                 normalize_returns=False, huber=False, huber_delta=1.0,
                 value_clip=None, return_norm_momentum=0.05, device=None,
                 weight_feature_keys=None, holliday_features=False):
        self.learning_rate = learning_rate
        self._explicit_weight_features = weight_feature_keys is not None
        if weight_feature_keys is not None:
            self.weight_feature_keys = tuple(weight_feature_keys)
            self.weight_input_keys = tuple(k for k in COST_WEIGHT_KEY_ORDER
                                           if k in self.weight_feature_keys)
        self.holliday_features = holliday_features
        self.device = device if device is not None else globals().get('DEVICE', torch.device('cpu'))
        self.decay = decay
        self.optim = None
        self.model = None
        self._curr_estimate = None
        self.loss_fn = torch.nn.MSELoss()
        self.actor_model = actor_model
        self._shared_critic = actor_model is not None and \
            hasattr(actor_model, "get_critic_features")
        # --- Optional improved-critic mode (default OFF -> legacy behaviour) ---
        # normalize_returns: regress the value head against running-standardized
        #   returns (PopArt-style value normalization) and de-normalize on
        #   from_state so GAE keeps working in raw units. Stabilizes the critic
        #   when the return scale/ tails shift (e.g. adjustment penalty in cost).
        # huber: SmoothL1 instead of MSE -> robust to heavy-tailed returns.
        # value_clip: PPO-style clipped value loss around the collection-time
        #   value (in normalized units when normalize_returns, else raw).
        self.normalize_returns = bool(normalize_returns)
        self.huber = bool(huber)
        self.huber_delta = float(huber_delta)
        self.value_clip = None if value_clip is None else float(value_clip)
        self._ret_momentum = float(return_norm_momentum)
        self._ret_mean = torch.zeros((), device=self.device)
        self._ret_var = torch.ones((), device=self.device)
        self._ret_std = torch.ones((), device=self.device)
        self._ret_initialized = False
        if not self._shared_critic:
            self._build_model(self.legacy_input_dim)

    def _build_model(self, input_dim):
        self.model = torch.nn.Sequential(
            FeatureNorm(input_dim, 0.001, unbiased=self.holliday_features),
            get_mlp(3, input_dim * 2, in_dim=input_dim, out_dim=1,
                    dropout=0.0)
        ).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.decay)

    @property
    def legacy_input_dim(self):
        # total demand, num demand edges, mean and std edge demand,
         # mean and std edge time weighted by demand on edge,
         # max num of mean stop features, # routes so far and # routes left,
         # weights of the two cost components
        return 10 + 9 + len(self.weight_feature_keys) if self._explicit_weight_features else 21

    @property
    def input_dim(self):
        # Back-compat alias for the legacy hand-crafted feature dim.
        return self.legacy_input_dim

    def _value_loss(self, pred, target):
        if self.huber:
            return torch.nn.functional.smooth_l1_loss(
                pred, target, beta=self.huber_delta, reduction='none')
        return (pred - target) ** 2

    def _update_return_stats(self, costs):
        batch_mean = costs.mean()
        batch_var = costs.var(unbiased=False)
        if not self._ret_initialized:
            self._ret_mean = batch_mean.detach().clone()
            self._ret_var = batch_var.detach().clone()
            self._ret_initialized = True
        else:
            m = self._ret_momentum
            self._ret_mean = (1 - m) * self._ret_mean + m * batch_mean.detach()
            self._ret_var = (1 - m) * self._ret_var + m * batch_var.detach()
        self._ret_std = (self._ret_var + 1e-6).sqrt()

    def update(self, costs, old_values=None):
        """Run a critic update step.

        Returns a dict with detached snapshots so the trainer can log
        critic-quality metrics (MSE, value-vs-return calibration, residuals)
        without needing to re-run the model. ``old_values`` (collection-time
        value estimates, raw units) enables PPO value-clipping when
        ``value_clip`` is set.
        """
        self.optim.zero_grad()
        pred = self._curr_estimate
        costs = costs.to(pred.dtype)

        if self.normalize_returns:
            mean, std = self._ret_mean, self._ret_std
            target = (costs - mean) / std
            old_n = None if old_values is None else \
                (old_values.to(pred.dtype) - mean) / std
        else:
            target = costs
            old_n = None if old_values is None else old_values.to(pred.dtype)

        if self.value_clip is not None and old_n is not None:
            pred_clipped = old_n + torch.clamp(
                pred - old_n, -self.value_clip, self.value_clip)
            loss = torch.maximum(self._value_loss(pred, target),
                                 self._value_loss(pred_clipped, target)).mean()
        else:
            loss = self._value_loss(pred, target).mean()

        # Snapshots in RAW units (de-normalized) so explained-variance / MSE
        # diagnostics stay comparable across modes.
        if self.normalize_returns:
            raw_values = pred.detach() * self._ret_std + self._ret_mean
        else:
            raw_values = pred.detach()
        values_snapshot = raw_values.clone()
        targets_snapshot = costs.detach().clone()

        loss.backward()
        self.optim.step()
        self._curr_estimate = None
        # update the feature normalization statistics
        self.model[0].update()
        # update running return statistics AFTER using them for this step
        if self.normalize_returns:
            self._update_return_stats(costs.detach())
        return {
            "loss": float(loss.detach().item()),
            "values": values_snapshot,
            "targets": targets_snapshot,
        }

    def inputs_from_data(self, graph_data, cost_weights):
        dev = graph_data[STOP_KEY].x.device
        input_data = torch.zeros(graph_data.num_graphs, self.input_dim, 
                                 dtype=torch.float, device=dev)
        input_data[:, 0] = graph_data.demand.sum(dim=(1,2))

        for bi, dl in enumerate(graph_data.to_data_list()):
            dmd_graph = dl[DEMAND_KEY]
            input_data[bi, 1] = dmd_graph.num_edges
            input_data[bi, 2] = dmd_graph.edge_attr[:, 0].mean()
            if dmd_graph.num_edges > 1:
                input_data[bi, 3] = dmd_graph.edge_attr[:, 0].std()
            else:
                input_data[bi, 3] = 0

            dmd_weighted_times = dmd_graph.edge_attr[:, 0] * \
                dmd_graph.edge_attr[:, 1]
            input_data[bi, 4] = dmd_weighted_times.mean()
            if self.holliday_features:
                # Match the preserved upstream NNBaseline statistic exactly.
                input_data[bi, 5] = dmd_weighted_times.mean()
            elif dmd_graph.num_edges > 1:
                input_data[bi, 5] = dmd_weighted_times.std()
            else:
                input_data[bi, 5] = 0
            x_dim = dl[STOP_KEY].x.shape[1]
            input_data[bi, 6:6+x_dim] = dl[STOP_KEY].x.mean(dim=0)

        # Named dict -> fixed slots by the explicit key spec (never dict
        # iteration order). A key absent from the dict (e.g. the {} passed by
        # set_input_norm) leaves its slot at zero.
        for ii, key in enumerate(self.weight_input_keys):
            if key in cost_weights:
                input_data[:, -(1 + ii)] = cost_weights[key]

        return input_data

    def set_input_norm(self, all_graph_data):
        if not isinstance(all_graph_data, Batch):
            all_graph_data = Batch.from_data_list(all_graph_data)
        input_data = self.inputs_from_data(all_graph_data, {})
        self.model[0].running_mean[...] = input_data.mean(dim=0)
        self.model[0].running_var[...] = input_data.var(dim=0)
 
        self.model[0].running_mean[-15:] = 0
        self.model[0].running_var[-15:] = 1
    
    def _features_from_state(self, state):
        if self._shared_critic:
            return self.actor_model.get_critic_features(state)
        input_data = self.inputs_from_data(state.graph_data,
                                           state.cost_weights)
        glob_feats = state.get_global_state_features(
            weight_feature_keys=self.weight_feature_keys)
        input_data[..., -glob_feats.shape[-1]:] = glob_feats
        return input_data

    def from_state(self, state):
        input_data = self._features_from_state(state)
        if self.model is None:
            self._build_model(input_data.shape[-1])
        pred = self.model(input_data).squeeze(-1)
        # ``_curr_estimate`` is what update() regresses: the normalized
        # prediction when return-normalization is on, else the raw value.
        self._curr_estimate = pred
        if self.normalize_returns:
            baseline = pred * self._ret_std + self._ret_mean
        else:
            baseline = pred

        assert baseline.isfinite().all()

        return baseline.detach()
    
    def get_baseline(self, graph_data, n_routes, cost_weights, state_vecs,
                     sumwriter=None, n_eps=None):
        # set up the input data
        assert self._curr_estimate is None, "Baseline estimate already exists!"
        
        input_data = self.inputs_from_data(graph_data, cost_weights)
        # add extra features and batch dimension for the numbers of routes
        input_data = input_data[:, None].repeat(1, state_vecs.shape[-2], 1)
        input_data[..., -state_vecs.shape[-1]:] = state_vecs
        
        baseline = self.model(input_data).squeeze(-1)
        self._curr_estimate = baseline

        sumwriter.add_scalar("baseline", baseline.mean(), n_eps)
        sumwriter.add_scalar("baseline std", baseline.std(), n_eps)

        assert baseline.isfinite().all()

        return baseline.detach()
    

def eval_model_over_nroutes(model, eval_dl, n_routes_range, eval_cfg, cost_obj, 
                            sumwriter=None, n_eps=0, n_samples=None, 
                            silent=False, device=None):
    # evaluate the model over a variety of n_routes, and take the mean
    eval_costs = []
    all_metrics = None
    cached_eval_n_routes = eval_cfg.get('n_routes', None)
    for n_routes in n_routes_range:
        eval_cfg.n_routes = n_routes
        mean_cost, metrics = \
            eval_model(model, eval_dl, eval_cfg, cost_obj, silent=silent, 
                       n_samples=n_samples, device=device if device is not None else next(model.parameters()).device)
        eval_costs.append(mean_cost)

        if all_metrics is None:
            all_metrics = metrics
        else:
            for key, val in metrics.items():
                all_metrics[key] = torch.cat((all_metrics[key], val))

    if cached_eval_n_routes is not None:
        # restore the cached routes, just in case
        eval_cfg.n_routes = cached_eval_n_routes

    mean_eval_cost = sum(eval_costs) / len(eval_costs)
    if sumwriter is not None:
        sumwriter.add_scalar("val cost", mean_eval_cost, n_eps)
        sumwriter.add_scalar('validation/cost', mean_eval_cost, n_eps)
        for metric_name in ['ATT', 'RTT', '$d_{un}$', 
                            '# disconnected node pairs', 'median_connectivity', 'median_connectivity_weighted']:
            mean_value = all_metrics[metric_name].mean()
            sumwriter.add_scalar(f"val {metric_name}", mean_value, n_eps)
            if metric_name in ('ATT', 'RTT'):
                sumwriter.add_scalar(f'validation/{metric_name.lower()}_minutes', mean_value, n_eps)

    return mean_eval_cost


def render_scenario(graph, routes, show_demand=False):
    # draw the routes on top of the city
    num_colours = max(2, int(np.ceil((2 * len(routes)) ** (1/3))))
    dim_points = np.linspace(0.1, 0.9, num_colours)
    # filter out grayscale colours
    colours = [cc for cc in product(dim_points, dim_points, dim_points)
                if len(np.unique(cc)) > 1]
    colours = iter(colours)

    nx_graph = nx.DiGraph()
    for ii in range(graph.num_nodes):
        nx_graph.add_node(ii)
    posdict = {ii: pos.cpu().numpy() 
               for ii, pos in enumerate(graph[STOP_KEY].pos)}
    nx.draw_networkx_nodes(nx_graph, pos=posdict, node_size=10)
    widths = torch.linspace(10, 2, len(routes))
    for width, colour, route in zip(widths, colours, routes):
        route_graph = nx.DiGraph()
        if type(route) is torch.Tensor:
            route = [ss.item() for ss in route]
        for jj in range(len(route) - 1):
            route_graph.add_edge(route[jj], route[jj+1])
        nx.draw_networkx_edges(route_graph, pos=posdict, width=width, 
                               edge_color=colour)
    
    if show_demand:
        dmd_graph = nx.DiGraph()
        for ii, jj in graph[DEMAND_KEY].edge_index.T:
            dmd_graph.add_edge(ii.item(), jj.item())
        demands = graph.demand[graph[DEMAND_KEY].edge_index[0], 
                               graph[DEMAND_KEY].edge_index[1]]
        widths = demands.cpu().numpy() * 5                               
        nx.draw_networkx_edges(dmd_graph, pos=posdict, width=widths,
                               edge_color='red', style="dotted")

    plt.show()


class DummySummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass


def _collect_episode_statistics(rewards, dones, reward_sums, lengths):
    """Collect finished episodes, retaining unfinished totals across rollouts."""
    completed_rewards, completed_lengths = [], []
    for step_rewards, step_dones in zip(rewards, dones):
        reward_sums.add_(step_rewards)
        lengths.add_(1)
        completed_rewards.append(reward_sums[step_dones])
        completed_lengths.append(lengths[step_dones])
        reward_sums[step_dones] = 0
        lengths[step_dones] = 0
    return torch.cat(completed_rewards), torch.cat(completed_lengths)


def _train_construction(model, min_n_routes, max_n_routes, cfg, optimizer,
              train_dataloader, val_dataloader, cost_obj, eval_fn, 
              return_scale=1, sumwriter=None, optuna_trial=None, device=None, 
              checkpoint_dir=None, algorithm='ppo',
              max_seconds=None, iteration_callback=None, evaluation_callback=None):
    if algorithm not in ('ppo', 'decomposed_ppo', 'd3po'):
        raise ValueError(f'Unknown construction algorithm: {algorithm}')
    if device is None:
        device = next(model.parameters()).device
    if cfg.batch_size < 2 or min(cfg.ppo.horizon, cfg.ppo.n_epochs,
                                 cfg.ppo.n_iterations, cfg.ppo.val_period) < 1:
        raise ValueError('Construction training requires batch_size >= 2 and positive PPO budgets')
    decomposed = algorithm != 'ppo'
    if cfg.ppo.get('advantage_normalization', 'minibatch') != 'minibatch':
        raise ValueError('Construction uses Holliday minibatch advantage normalization')
    from .improvement_learning import (
        D3POValueModule, _compute_ppo_returns_and_advantages,
        _update_lc_improvement_cfg_d3po_from_rollout, _normalize_minibatch_advantages,
    )
    objective_indices = tuple(i for i, enabled in enumerate(cost_obj.enabled_component_mask)
                              if enabled) if decomposed else ()
    batch_size = train_dataloader.batch_size
    assert batch_size == cfg.ppo.minibatch_size, \
        "For now, minibatch size must be the same as the batch size, due to " \
        "an unresolved issue with concatenating states."
    # assert cfg.ppo.minibatch_size % batch_size == 0, \
    #     "minibatch size must be a multiple of the batch size"
    assert (cfg.ppo.horizon * batch_size) % cfg.ppo.minibatch_size == 0, \
        "minibatch size must wholly divide the number of samples per epoch"

    if sumwriter is None:
        # instantiate a dummy summary writer so that the calls to it still work
        sumwriter = DummySummaryWriter()
    
    # set up neural network baseline
    critic_kwargs = dict(learning_rate=cfg.baseline_lr, device=device,
                         weight_feature_keys=model.weight_feature_keys,
                         holliday_features=bool(cfg.ppo.get('holliday_critic_features', False)))
    val_module = (D3POValueModule(n_objectives=len(objective_indices), **critic_kwargs)
                  if decomposed else NNBaseline(**critic_kwargs))
    if cfg.get('action_seed') is not None:
        torch.manual_seed(int(cfg.action_seed))
    # val_module.set_input_norm(train_dataloader.dataset)

    best_val_cost = float('inf')
    n_routes = max_n_routes
    model.eval()
    supports_route_actions = getattr(model, 'supports_trim_actions', False)
    n_states_per_minibatch = cfg.ppo.minibatch_size // batch_size
    gamma = cfg.discount_rate
    epsilon = cfg.ppo.epsilon
    horizon = cfg.ppo.horizon

    # set up progress bar
    pbar = tqdm(total=cfg.ppo.n_iterations)
    
    train_gen = iter(train_dataloader)
    n_routes = None
    ep_finished = None
    started = time.monotonic()
    completed_iterations = 0
    graph_instances_loaded = 0
    graph_batches_loaded = 0
    for iteration in range(cfg.ppo.n_iterations):
        if iteration > 0 and max_seconds is not None and time.monotonic() - started >= max_seconds:
            break
        iteration_started = time.monotonic()
        if iteration % cfg.ppo.val_period == 0:
            calibrate_first = bool(cfg.ppo.get('calibrate_before_validation', True))
            if iteration == 0 and calibrate_first:
                # Calibrate first, then score the frozen actor that is actually
                # saved as best_model and used by subsequent validation.
                eval_fn(model, val_dataloader, None, pbar.n)
                model.update_and_freeze_feature_norms()
            avg_val_cost = eval_fn(model, val_dataloader, sumwriter, pbar.n)
            if iteration == 0 and not calibrate_first:
                # Original order: one validation pass, then freeze its norms.
                model.update_and_freeze_feature_norms()
            if evaluation_callback is not None:
                evaluation_callback(model, val_dataloader, pbar.n)

            if optuna_trial is not None:
                optuna_trial.report(avg_val_cost, iteration)
                if iteration > 0 and optuna_trial.should_prune():
                    # don't prune just because of a bad initialization
                    raise optuna.exceptions.TrialPruned()

            # Kool et al. does this with a paired t-test, we're keeping it simple
            # for the moment.
            if avg_val_cost < best_val_cost:
                # update the best model and the baseline
                best_model = copy.deepcopy(model)
                best_val_cost = avg_val_cost

        if n_routes is None:
            # use the max n routes first, so that if we're going to run out
             # of memory, we do so right away
            n_routes = max_n_routes
        else:
            # pick a random number of routes in the evaluation range
            n_routes = random.randint(min_n_routes, max_n_routes)

        if ep_finished is None or ep_finished.all():
            # all eps in this batch have finished, so get the next batch
            try:
                data = next(train_gen)
            except StopIteration:
                # we ran out of data, so iterate over it again
                train_gen = iter(train_dataloader)
                data = next(train_gen)
            data = data.to(device)
            graph_instances_loaded += data.num_graphs
            graph_batches_loaded += 1

            # sample cost weights
            cost_weights = cost_obj.sample_variable_weights(data.num_graphs, device)
            state = RouteGenBatchState(data, cost_obj, n_routes, 
                                       cfg.eval.min_route_len,
                                       cfg.eval.max_route_len,
                                       cost_weights=cost_weights)
            state = model.setup_planning(state)
            ep_finished = state.is_done()
            # A new graph batch starts new episodes; keep these across rollouts
            # only while the current batch is retained.
            episode_reward_sums = torch.zeros(state.batch_size, device=device)
            episode_lengths = torch.zeros(state.batch_size, dtype=torch.long, device=device)

            # compute the cost of the scenarios with no transit
            if cfg.diff_reward:
                with torch.no_grad():
                    base_result = cost_obj(state)
                    base_cost = (cost_obj.get_cost_components(state, base_result)[:, objective_indices]
                                 if decomposed else base_result.cost)
                    prev_cost = base_cost
            
        # set up the buffers to hold the rollout
        reward_shape = (horizon, state.batch_size) + ((len(objective_indices),) if decomposed else ())
        buf_rewards = torch.zeros(reward_shape,
                                  device=device)
        buf_val_ests = buf_rewards.clone()
        buf_logits = torch.zeros((horizon, state.batch_size), device=device)
        done_at_step_mask = torch.zeros_like(buf_logits, dtype=torch.bool)
        buf_actions = torch.zeros((horizon, state.batch_size, 2),
                                  device=device, dtype=torch.long)

        buf_actions = torch.full((horizon, state.batch_size, 2), -1,
                                 device=device, dtype=torch.long)
        buf_action_kinds = torch.full((horizon, state.batch_size),
                                      ROUTE_ACTION_EXTEND,
                                      device=device, dtype=torch.long)
        buf_states = []
        terminal_metrics = {key: [] for key in ('cost', 'att_minutes', 'rtt_minutes',
                                                'feasible_fraction', 'unserved_fraction', 'alpha')}

        # roll out the specified number of steps
        with torch.no_grad():    
            for tt in range(horizon):
                val_ests = val_module.from_state(state)
                buf_val_ests[tt] = val_ests
                buf_states.append(state.clone().to_device('cpu'))
                if supports_route_actions:
                    action_kinds, actions, logits, _ = \
                        model.step_route_action(state)
                    buf_action_kinds[tt] = action_kinds
                else:
                    actions, logits, _ = model.step(state)
                buf_actions[tt] = actions
                # update the states with the actions
                if supports_route_actions:
                    state.apply_route_actions(action_kinds, actions)
                else:
                    state.shortest_path_action(actions)
                done = state.is_done().clone()
                done_at_step_mask[tt] = done
    
                buf_logits[tt] = logits
                if cfg.diff_reward:
                    # reward is decrease in cost
                    result = cost_obj(state)
                    reward_cost = (cost_obj.get_cost_components(state, result)[:, objective_indices]
                                   if decomposed else result.cost)
                    rewards = (prev_cost - reward_cost) * return_scale
                    # Completed elements are rebased after reset/resampling.
                    prev_cost = reward_cost.clone()
                elif done.any():
                    # reward is the decrease in cost at last step, 0 otherwise
                    result = cost_obj(state)
                    reward_cost = (cost_obj.get_cost_components(state, result)[:, objective_indices]
                                   if decomposed else result.cost)
                    finished = done[:, None] if decomposed else done
                    rewards = -reward_cost * return_scale * finished
                else:
                    # rewards are all 0 since no episode is done
                    rewards = torch.zeros_like(buf_rewards[0])

                buf_rewards[tt] = rewards
                if done.any():
                    terminal_metrics['cost'].append(result.cost[done])
                    terminal_metrics['att_minutes'].append(result.mean_demand_time[done] / 60)
                    terminal_metrics['rtt_minutes'].append(result.total_route_time[done] / 60)
                    terminal_metrics['feasible_fraction'].append((~result.are_constraints_violated())[done].float())
                    terminal_metrics['unserved_fraction'].append((result.unserved_demand / result.total_demand)[done])
                    terminal_metrics['alpha'].append(state.cost_weights['demand_time_weight'][done])
                
                # mark which episodes have reached the end at least once
                ep_finished |= done
                state.reset_dones()
                if cfg.ppo.get('resample_preferences_on_terminal', decomposed) and done.any():
                    # D3PO changes preference at episode boundaries, using the
                    # same cost sampler as PPO. PPO keeps Holliday's batch weights.
                    sampled = cost_obj.sample_variable_weights(int(done.sum()), device)
                    merged = {key: value.clone() for key, value in state.cost_weights.items()}
                    for key, value in sampled.items():
                        merged[key][done] = value
                    state.set_cost_weights(merged)
                if cfg.diff_reward and done.any():
                    # New weights change the empty-network cost. Rebase only
                    # reset elements, without changing the terminal reward.
                    reset_result = cost_obj(state)
                    reset_cost = (cost_obj.get_cost_components(state, reset_result)[:, objective_indices]
                                  if decomposed else reset_result.cost)
                    prev_cost[done] = reset_cost[done]
            
            # compute the returns and advantages
            final_val_ests = val_module.from_state(state)
            buf_returns, buf_advantages = _compute_ppo_returns_and_advantages(
                buf_rewards, buf_val_ests, done_at_step_mask, final_val_ests,
                gamma, cfg.ppo.use_gae, cfg.ppo.gae_lambda)
            
        # log the average return and baseline on the rolled-out episodes
        if decomposed:
            weights = torch.stack([cost_obj.get_preference_weights(ss, normalize=True)[:, objective_indices]
                                   for ss in buf_states]).to(device)
            scalar_rewards = (buf_rewards * weights).sum(-1)
            sumwriter.add_scalar("avg return", scalar_rewards.mean(), pbar.n)
            sumwriter.add_scalar("baseline", (buf_val_ests * weights).sum(-1).mean(), pbar.n)
            for oi, ci in enumerate(objective_indices):
                name = ('ATT', 'RTT', 'connectivity')[ci]
                sumwriter.add_scalar(f"avg return {name}", buf_rewards[..., oi].mean(), pbar.n)
                sumwriter.add_scalar(f"baseline {name}", buf_val_ests[..., oi].mean(), pbar.n)
        else:
            scalar_rewards = buf_rewards
            sumwriter.add_scalar("avg return", buf_rewards.mean(), pbar.n)
            sumwriter.add_scalar("baseline", buf_val_ests.mean(), pbar.n)
        completed_rewards, completed_lengths = _collect_episode_statistics(
            scalar_rewards, done_at_step_mask, episode_reward_sums, episode_lengths)
        update_metrics = []
        critic_values, critic_targets = [], []

        if decomposed:
            rollout = dict(states=buf_states, rewards=buf_rewards, logits=buf_logits,
                           actions=buf_actions, action_kinds=buf_action_kinds,
                           preferences=weights,
                           diversity_preferences=torch.stack([
                               cost_obj.get_preference_weights(ss, normalize=True) for ss in buf_states]).to(device),
                           active_masks=torch.ones_like(done_at_step_mask),
                           check_initial_replay=True,
                           enabled_component_mask=cost_obj.enabled_component_mask)
            dcfg = cfg.d3po
            d3po_stats = _update_lc_improvement_cfg_d3po_from_rollout(
                model, optimizer, val_module, rollout, buf_returns, buf_advantages,
                cfg.ppo.n_epochs, cfg.ppo.minibatch_size, epsilon, cfg.entropy_weight, device,
                diversity_weight=float(dcfg.diversity_weight) if algorithm == 'd3po' else 0.0,
                diversity_alpha=float(dcfg.diversity_alpha),
                preference_noise_sigma=float(dcfg.preference_noise_sigma),
                normalize_advantages_per_objective=bool(dcfg.normalize_advantages_per_objective))
            stats = {'objective': d3po_stats['objective'],
                     'entropy': d3po_stats['entropy'],
                     'initial ratio error': d3po_stats['initial_ratio_error'],
                     'value loss': d3po_stats['critic_mse_mean'],
                     'clip fraction': d3po_stats['clipped_count'] / max(d3po_stats['ratio_count'], 1),
                     'diversity loss': d3po_stats['diversity_loss']}
            for oi, value in enumerate(d3po_stats['critic_explained_variance_per_obj'] or []):
                sumwriter.add_scalar(f'critic explained variance {oi}', value, iteration)
                stats[f'critic/explained_variance_{("att", "rtt", "connectivity")[objective_indices[oi]]}'] = value
        else:
            # train on the rolled-out episodes
            train_orders = [torch.randperm(len(buf_states), device=device)
                            for _ in range(cfg.ppo.n_epochs)]
            train_order = torch.cat(train_orders)
            for idxs in torch.split(train_order, n_states_per_minibatch):
                mb_states = [buf_states[ii] for ii in idxs]
                if n_states_per_minibatch == 1:
                    mb_states = mb_states[0]
                else:
                    mb_states = RouteGenBatchState.batch_from_list(mb_states)
                # move states to chosen device
                mb_states = mb_states.to_device(device)
                mb_acts = buf_actions[idxs].flatten(0, 1)
                if supports_route_actions:
                    mb_action_kinds = buf_action_kinds[idxs].flatten(0, 1)
                mb_old_logits = buf_logits[idxs].flatten(0, 1)
                mb_returns = buf_returns[idxs].flatten(0, 1)
                mb_advs = buf_advantages[idxs].flatten(0, 1)

                # compute V(s) loss and update the value estimator
                val_module.from_state(mb_states)
                value_stats = val_module.update(mb_returns)
                critic_values.append(value_stats['values'])
                critic_targets.append(value_stats['targets'])

                # step model forward
                if supports_route_actions:
                    _, _, logits, entropy = model.step_route_action(
                        mb_states, actions=mb_acts, action_kinds=mb_action_kinds)
                else:
                    _, logits, entropy = model.step(mb_states, actions=mb_acts)
            
                # normalize advantages by the minibatch statistics
                mb_advs = _normalize_minibatch_advantages(mb_advs)

                # compute clip loss
                ratios = (logits - mb_old_logits).exp()
                if (ratios == 0).any():
                    log.warning(f"Some ratios are zero: {ratios}")
                if ratios.isinf().any():
                    log.warning(f"Some ratios are very extreme: {ratios}")

                assert ratios.isfinite().all()
                
                clipped_ratios = ratios.clamp(1 - epsilon, 1 + epsilon)
                ratio_error = float((ratios.detach() - 1).abs().max())
                if not update_metrics and ratio_error > 2e-4:
                    raise RuntimeError(f'Initial PPO replay ratio differs from 1: {ratio_error}')
                clip_obj = torch.minimum(ratios * mb_advs,
                                         clipped_ratios * mb_advs)
                objective = clip_obj.mean()
                objective += entropy.mean() * cfg.entropy_weight
                # do backprop and weight updates
                optimizer.zero_grad()
                objective.backward()
                # based on Andrychowicz et al. 2021, we clip the gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5,
                                               error_if_nonfinite=True)
                optimizer.step()
                update_metrics.append([float(clip_obj.detach().mean()), float(objective.detach()),
                    float(entropy.detach().mean()), value_stats['loss'],
                    float(((ratios.detach() - 1).abs() > epsilon).float().mean()),
                    ratio_error])

            means = np.mean(update_metrics, axis=0)
            stats = dict(zip(('route obj', 'objective', 'entropy', 'value loss',
                              'clip fraction'), means[:5].tolist()))
            stats['initial ratio error'] = update_metrics[0][-1]
            targets = torch.cat(critic_targets)
            residuals = targets - torch.cat(critic_values)
            variance = targets.var(unbiased=False)
            stats['critic/explained_variance'] = float(1-residuals.var(unbiased=False)/variance) if variance > 0 else float('nan')

        # save a checkpoint every val period
        if checkpoint_dir is not None and iteration % cfg.ppo.val_period == 0:
            ckpt_filename = checkpoint_dir / f"iter{iteration}.pt"
            torch.save(model.state_dict(), ckpt_filename)

        pbar.update(1)
        completed_iterations = iteration + 1
        stats['train seconds'] = time.monotonic() - iteration_started
        stats['elapsed seconds'] = time.monotonic() - started
        stats['env transitions'] = completed_iterations * horizon * batch_size
        stats['graph instances loaded'] = graph_instances_loaded
        stats['graph batches loaded'] = graph_batches_loaded
        stats['completed networks'] = int(done_at_step_mask.sum())
        # No finished episodes means no measured mean, rather than zero or a
        # stale value carried forward by MetricWriter into the CSV.
        stats['mean episode reward'] = float(completed_rewards.mean()) if completed_rewards.numel() else float('nan')
        stats['mean episode length'] = float(completed_lengths.float().mean()) if completed_lengths.numel() else float('nan')
        stats['return target mean'] = float(buf_returns.mean())
        stats['train/reward'] = stats['mean episode reward']
        stats['train/episode_length'] = stats['mean episode length']
        stats['train/step_reward'] = float(scalar_rewards.mean())
        stats['train/completed_networks'] = stats['completed networks']
        for key, values in terminal_metrics.items():
            if key != 'alpha':
                stats[f'train/{key}'] = float(torch.cat(values).mean()) if values else float('nan')
        alpha = torch.cat(terminal_metrics['alpha']) if terminal_metrics['alpha'] else None
        stats['train/completed_alpha_mean'] = float(alpha.mean()) if alpha is not None else float('nan')
        for name, condition in (('alpha_0', lambda a: a == 0), ('alpha_1', lambda a: a == 1),
                                ('alpha_intermediate', lambda a: (a > 0) & (a < 1))):
            stats[f'train/completed_{name}_fraction'] = float(condition(alpha).float().mean()) if alpha is not None else float('nan')
        stats['ppo/clip_fraction'] = stats['clip fraction']
        stats['critic/value_loss'] = stats['value loss']
        stats['performance/iteration_seconds'] = stats['train seconds']
        stats['performance/env_transitions'] = stats['env transitions']
        if 'entropy' in stats:
            stats['ppo/entropy'] = stats['entropy']
        for key, value in stats.items():
            sumwriter.add_scalar(key, value, iteration)
        if iteration_callback is not None:
            iteration_callback(completed_iterations, stats, model, val_module, optimizer)

    # final eval
    avg_val_cost = eval_fn(model, val_dataloader, sumwriter, pbar.n)
    if evaluation_callback is not None:
        evaluation_callback(model, val_dataloader, pbar.n, final=True)
    if avg_val_cost < best_val_cost:
        # update the best model and the baseline
        best_model = copy.deepcopy(model)
        best_val_cost = avg_val_cost
    pbar.close()

    return best_model, best_val_cost


def train_ppo(*args, **kwargs):
    """Holliday construction PPO, sharing its rollout with construction D3PO."""
    return _train_construction(*args, algorithm='ppo', **kwargs)


def train_d3po(*args, **kwargs):
    """Construction rollout with the existing improvement D3PO critic/update."""
    return _train_construction(*args, algorithm='d3po', **kwargs)


def train_decomposed_ppo(*args, **kwargs):
    """D3PO ablation using the same update with diversity disabled."""
    return _train_construction(*args, algorithm='decomposed_ppo', **kwargs)


def train(model, min_n_routes, max_n_routes, cfg, optimizer, train_dataloader, 
          val_dataloader, cost_obj, eval_fn, sumwriter=None, optuna_trial=None,
          device=None, checkpoint_dir=None):
    """Train a model on a dataset.
    
    Args:
    eval_model -- a function that takes:
        - a model, 
        - a dataloader, 
        - a summary writer,
        - and the number of episodes so far
        as input, and returns a scalar cost.
    """
    if sumwriter is None:
        # instantiate a dummy summary writer so that the calls to it still work
        sumwriter = DummySummaryWriter()

    best_model = copy.deepcopy(model)
    best_cost = float('inf')
    try:
        graphs_per_epoch = len(train_dataloader.dataset)
    except TypeError:
        graphs_per_epoch = 9 * len(val_dataloader.dataset)
    batches_per_epoch = \
        math.ceil(graphs_per_epoch / train_dataloader.batch_size)
    pbar = tqdm(total=graphs_per_epoch * cfg.reinforce.n_epochs)

    bl_module = None
    if device is None:
        device = next(model.parameters()).device
    if cfg.baseline_mode == BLMODE_NONE:
        baseline = 0
    elif cfg.baseline_mode == BLMODE_ROLL:
        bl_module = RollingBaseline(cfg.bl_alpha)
    elif cfg.baseline_mode == BLMODE_NN:
        bl_module = NNBaseline(device=device)
        bl_module.set_input_norm(train_dataloader.dataset)
    elif cfg.baseline_mode == BLMODE_GREEDY:
        bl_module = FixedBaseline()
    
    log.info(f"discount rate is {cfg.discount_rate}")
    supports_route_actions = getattr(model, 'supports_trim_actions', False)

    n_routes = None
    for epoch in range(cfg.reinforce.n_epochs):
        avg_cost = eval_fn(model, val_dataloader, sumwriter, pbar.n)
        if epoch == 0:
            model.update_and_freeze_feature_norms()

        if optuna_trial is not None:
            optuna_trial.report(avg_cost, epoch)
            if epoch > 0 and optuna_trial.should_prune():
                # don't prune just because of a bad initialization
                raise optuna.exceptions.TrialPruned()

        # Kool et al. does this with a paired t-test, we're keeping it simple
         # for the moment.
        if avg_cost < best_cost:
            # update the best model and the baseline
            best_model = copy.deepcopy(model)
            best_cost = avg_cost
            if cfg.baseline_mode == BLMODE_GREEDY:
                bl_module.update(-avg_cost * cfg.reward_scale)

        dl_iter = iter(train_dataloader)
        for _ in range(batches_per_epoch):
            if n_routes is None:
                # use the max n routes first, so that if we're going to run out
                 # of memory, we do so right away
                n_routes = max_n_routes
            else:
                # pick a random number of routes in the evaluation range
                n_routes = random.randint(min_n_routes, max_n_routes)
            data = next(dl_iter)
            if device.type != 'cpu':
                data = data.cuda()

            # sample cost weights
            cost_weights = \
                cost_obj.sample_variable_weights(data.num_graphs, device)
            state = RouteGenBatchState(data, cost_obj, n_routes, cfg.eval.min_route_len,
                                       cfg.eval.max_route_len, cost_weights=cost_weights)
            state = model.setup_planning(state)
            
            # run those graphs though the net
            all_logits = []
            all_entropy = 0
            n_routes_so_far = []
            state_vecs = []

            if cfg.diff_reward:
                base_result = cost_obj(state)
                base_cost = base_result.cost
                prev_cost = base_cost
                rewards = []

            while not state.is_done().all():
                n_routes_so_far.append(state.n_finished_routes)
                state_vecs.append(state.get_global_state_features(
                    weight_feature_keys=model.weight_feature_keys))
                if supports_route_actions:
                    action_kinds, actions, logits, entropy = \
                        model.step_route_action(state)
                    state.apply_route_actions(action_kinds, actions)
                else:
                    actions, logits, entropy = model.step(state)
                    state.shortest_path_action(actions)
                if cfg.diff_reward:
                    result = cost_obj(state)
                    reward = (prev_cost - result.cost) * cfg.reward_scale
                    rewards.append(reward)
                    prev_cost = result.cost * ~state.is_done()
             
                all_logits.append(logits)
                all_entropy += entropy

            logits = torch.stack(all_logits, dim=1)

            if cfg.diff_reward:
                rewards = torch.stack(rewards, dim=1)
                # apply the discount factor, gamma, to get the returns
                rev_rwds = rewards.flip(-1)
                prev_rr = None
                rev_rtrns = []
                for rr in rev_rwds.transpose(1, 0):
                    if prev_rr is not None:
                        rr = rr + prev_rr * cfg.discount_rate
                    rev_rtrns.append(rr)
                    prev_rr = rr
                returns = torch.stack(rev_rtrns, dim=1).flip(-1)
            else:
                # all returns are just the final return
                result = cost_obj(state)
                costs = result.cost
                costs = costs[:, None].expand(-1, logits.shape[1])
                returns = -costs * cfg.reward_scale
                
            n_routes_so_far = torch.stack(n_routes_so_far, dim=1)
            state_vecs = torch.stack(state_vecs, dim=1)

            route_lens = [len(rr) for br in state.routes for rr in br]
            sumwriter.add_scalar("avg # stops on a route", 
                                 np.mean(route_lens), pbar.n)

            sumwriter.add_scalar("avg train cost", costs[:, -1].mean(), pbar.n)
            
            # Log some stats with the summary writer
            avg_rtrn = returns.mean().item()
            sumwriter.add_scalar("avg return", avg_rtrn, pbar.n)

            for metric_name, metric in result.get_metrics().items():
                mean_value = metric.mean()
                sumwriter.add_scalar(f"avg {metric_name}", mean_value, pbar.n)

            # Log some stats with the summary writer
            # if plan_out.route_est_vals is not None:
            #     baseline = plan_out.route_est_vals
            #     sumwriter.add_scalar("baseline", baseline.mean().item(), n_eps)
            #     returns -= baseline.detach()
            #     sumwriter.add_scalar("avg return vs baseline", returns.mean(),
            #                           n_eps)
            #     sumwriter.add_scalar("std return vs baseline", returns.std(), 
            #                           n_eps)
            if bl_module is not None:
                baseline = bl_module.get_baseline(data, n_routes, 
                                                  state.cost_weights,
                                                  state_vecs, sumwriter, 
                                                  pbar.n)
                no_action_mask = n_routes_so_far == n_routes
                # don't make any updates on these entries
                returns[no_action_mask] = baseline[no_action_mask]
                if cfg.baseline_mode != BLMODE_GREEDY:
                    bl_module.update(returns)
                returns -= baseline

                sumwriter.add_scalar("avg return vs baseline", returns.mean(),
                                     pbar.n)
                sumwriter.add_scalar("std return vs baseline", returns.std(), 
                                     pbar.n)

            objective = 0
            # compute the training signal on the logits
            route_signal = returns
            route_obj = (route_signal * logits).mean()
            sumwriter.add_scalar("route obj", route_obj.item(), pbar.n)
            objective += route_obj
            # add entropy to the signal
            objective += cfg.entropy_weight * entropy.mean()
            sumwriter.add_scalar("entropy", entropy.mean().item(), pbar.n)

            sumwriter.add_scalar("objective", objective.item(), pbar.n)

            # backprop and weight update
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.update(data.num_graphs)

        if checkpoint_dir is not None:
            ckpt_filename = checkpoint_dir / f"epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_filename)

    # final eval
    avg_cost = eval_fn(model, val_dataloader, sumwriter, pbar.n)
    if avg_cost < best_cost:
        # update the best model and the baseline
        best_model = copy.deepcopy(model)

    pbar.close()

    return best_model, best_cost


def setup_and_train(cfg: DictConfig, trial: optuna.trial.Trial = None, *,
                    model=None, cost_obj=None, device=None, run_name=None,
                    sum_writer=None, output_dir=None, iteration_callback=None,
                    return_training_result=False):
    if model is None:
        device, run_name, sum_writer, cost_obj, model = \
            lrnu.process_standard_experiment_cfg(cfg, 'inductive_')
    # model always has legal route lengths by construction, so we don't need
     # to penalize it for illegal routes
    cost_obj.ignore_stops_oob = True

    optimizer_type = getattr(torch.optim, cfg.optimizer)
    optimizer = optimizer_type(model.parameters(), lr=cfg.lr, 
                               weight_decay=cfg.decay, 
                               maximize=True)
    val_batch_size = cfg.batch_size * 4
    
    if cfg.dataset.type == 'pickle':
        train_ds, val_ds = get_default_train_and_eval_split(
            **cfg.dataset.kwargs)
        if len(train_ds) < cfg.batch_size or len(val_ds) == 0:
            raise ValueError('Construction dataset needs a full training batch and nonempty validation split')
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, 
                              shuffle=True, drop_last=True)
    elif cfg.dataset.type == 'dynamic':
        train_ds = get_dynamic_training_set(**cfg.dataset.kwargs)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size)
        val_ds = CityGraphDataset(cfg.dataset.val_path)
    elif cfg.dataset.type == 'mumford':
        do_scaling = cfg.dataset.get('scale_dynamically', True)
        data = CityGraphData.from_mumford_data(cfg.dataset.path, 
                                               cfg.dataset.city,
                                               scale_dynamically=do_scaling)
        train_ds = [data] * cfg.batch_size
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size)
        val_ds = [data] * val_batch_size
    else:
        raise ValueError(f'Unknown construction dataset type: {cfg.dataset.type}')

    evaluation_cfg = cfg.get('evaluation', {})
    evaluation_count = evaluation_cfg.get('graph_count')
    validation_indices = list(range(len(val_ds)))
    if evaluation_count is not None:
        if int(evaluation_count) < 1:
            raise ValueError('evaluation.graph_count must be positive or null')
        generator = torch.Generator().manual_seed(int(evaluation_cfg.get('seed', 9001)))
        validation_indices = torch.randperm(len(val_ds), generator=generator)[:int(evaluation_count)].sort().values.tolist()
        val_ds = [val_ds[i] for i in validation_indices]
        val_batch_size = min(val_batch_size, len(val_ds))
    val_dl = DataLoader(val_ds, batch_size=val_batch_size)
    if sum_writer is not None:
        if cfg.dataset.type != 'dynamic':
            sum_writer.add_scalar('training graph count', len(train_ds), 0)
        sum_writer.add_scalar('validation graph count', len(val_ds), 0)

    # run training
    model.to(device)
    # set a range of random cost parameters to use during validation
    val_cost_obj = copy.deepcopy(cost_obj)
    val_weights = val_cost_obj.sample_variable_weights(val_batch_size, device)
    val_cost_obj.set_weights(**val_weights)
    
    eval_n_routes = cfg.eval.n_routes
    if type(eval_n_routes) is int:
        eval_n_routes = [eval_n_routes]
        
    # define evaluation function
    eval_fn = lambda mm, dd, sw, ne: \
        eval_model_over_nroutes(mm, dd, eval_n_routes, cfg.eval, val_cost_obj, 
                                sw, ne, n_samples=1, silent=True, device=device)

    if output_dir is not None:
        output_dir = Path(output_dir)
    elif 'outdir' in cfg:
        output_dir = Path(cfg.outdir)
    else:
        output_dir = Path('output')

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    evaluation_callback = None
    if evaluation_cfg.get('pareto_enabled', False):
        from .training.construction_experiment import evaluate_pareto
        from .core.artifacts import ArtifactStore
        if sum_writer is None:
            raise ValueError('Pareto monitoring requires a TensorBoard writer')
        if int(evaluation_cfg.pareto_period) < 1:
            raise ValueError('evaluation.pareto_period must be positive')
        ArtifactStore(output_dir).save_json(dict(indices=validation_indices,
            seed=int(evaluation_cfg.seed), reference=list(evaluation_cfg.hv_reference),
            split='held-out validation; independent test reserved for final evaluation'),
            'pareto/validation_protocol.json')

        def evaluation_callback(actor, loader, step, final=False):
            if final or step % int(evaluation_cfg.pareto_period) == 0:
                evaluate_pareto(actor, loader.dataset, cfg, cost_obj, output_dir,
                                device, sum_writer, step)

    min_n_routes = min(eval_n_routes)
    max_n_routes = max(eval_n_routes)
    checkpoint_dir = output_dir / f'{run_name}_checkpoints'
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    trainer = cfg.get('trainer', 'ppo' if 'ppo' in cfg else 'reinforce')
    train_fn = {'ppo': train_ppo, 'd3po': train_d3po,
                'decomposed_ppo': train_decomposed_ppo, 'reinforce': train}[trainer]
    extra = {} if trainer == 'reinforce' else dict(
        max_seconds=cfg.get('train_loop', {}).get('max_seconds'),
        return_scale=float(cfg.get('reward_scale', 1.0)),
        iteration_callback=iteration_callback, evaluation_callback=evaluation_callback)
    best_model, best_cost = train_fn(
        model, min_n_routes, max_n_routes, cfg, optimizer, train_dl,
        val_dl, cost_obj, eval_fn, sumwriter=sum_writer, device=device,
        optuna_trial=trial, checkpoint_dir=checkpoint_dir, **extra)

    if return_training_result:
        return best_model, best_cost
    torch.save(best_model.state_dict(), output_dir / (run_name + '.pt'))
    return best_cost
