"""Bee-colony selection and acceptance, in the original RNG order."""
import logging as log
import torch
from tqdm import tqdm
from ..env.state import RouteGenBatchState
from ..common.logging import log as log_metrics
from ..objectives.metrics import evaluate_network, monitor_metrics
from .records import SearchBestRecord, SearchRecord


def _ensure_mutation_stats_bucket(mutation_counts_out, bucket_name, bee_names):
    if mutation_counts_out is None:
        return None
    bucket = mutation_counts_out.setdefault(bucket_name, {})
    for name in bee_names:
        bucket.setdefault(name, 0)
    return bucket


def _record_attempted_mutations(mutation_counts_out, attempted_counts):
    """Record attempted mutations per bee name.

    ``attempted_counts`` maps a bee name (the plan group's name, e.g. the
    declarative BeeSpec name) to how many bees ran that operator this step --
    ``plan.summary()``. Stats are keyed by bee name, not by legacy type slot.
    """
    if mutation_counts_out is None:
        return
    attempted = _ensure_mutation_stats_bucket(mutation_counts_out, "attempted", attempted_counts.keys())
    for name, inc in attempted_counts.items():
        attempted[name] += int(inc)
        # Flat alias at the top level so callers reading the bare name see the
        # running attempted total.
        mutation_counts_out[name] = attempted[name]


def _record_mutation_mask(mutation_counts_out, bucket_name, mutation_types, mask, bee_names):
    """Add ``mask`` hits to ``bucket_name``, keyed by bee name.

    ``bee_names`` is the plan's ordered group names; ``mutation_types`` holds
    each bee's 1-based group slot, so slot ``i`` maps to ``bee_names[i - 1]``.
    """
    if mutation_counts_out is None:
        return
    bucket = _ensure_mutation_stats_bucket(mutation_counts_out, bucket_name, bee_names)
    if mutation_types is None:
        return

    for slot, name in enumerate(bee_names, start=1):
        type_mask = mutation_types == slot
        if not type_mask.any():
            continue
        bucket[name] += int(mask[:, type_mask].sum().item())


def _record_accepted_mutations(mutation_counts_out, mutation_types, accepted_mask, bee_names):
    _record_mutation_mask(mutation_counts_out, "accepted", mutation_types, accepted_mask, bee_names)


def _ensure_selection_stats_bucket(mutation_counts_out):
    if mutation_counts_out is None:
        return None
    bucket = mutation_counts_out.setdefault("selection", {})
    bucket.setdefault("parent_copies", 0)
    bucket.setdefault("nonbest_parent_copies", 0)
    bucket.setdefault("worse_parent_copies", 0)
    return bucket


def _record_selection_stats(mutation_counts_out, parent_idxs, bee_costs):
    """Record which population members survive the selection step."""
    if mutation_counts_out is None:
        return
    bucket = _ensure_selection_stats_bucket(mutation_counts_out)
    parent_costs = bee_costs.gather(1, parent_idxs)
    best_costs = bee_costs.min(dim=1, keepdim=True).values

    bucket["parent_copies"] += int(parent_idxs.numel())
    bucket["nonbest_parent_copies"] += int((parent_costs > best_costs).sum().item())
    bucket["worse_parent_copies"] += int((parent_costs > bee_costs).sum().item())


def _choose_route_indices(bee_networks, n_routes):
    """Sample one route index per bee for mutation, uniformly."""
    return torch.randint(
        high=int(n_routes.item()),
        size=bee_networks.shape[:2],
        device=bee_networks.device,
    )


class Colony:
    """Own the population, evaluation and ordered colony operations."""

    def __init__(self, state, cost_obj, init_network, *, method, n_iterations, plan,
                 silent, logger_list, mutation_counts_out):
        self.state = state
        self.cost_obj = cost_obj
        self.init_network = init_network
        self.passes_per_it = method.passes_per_it
        self.mod_steps_per_pass = method.mod_steps_per_pass
        self.method = method
        self.n_iterations = n_iterations
        self.plan = plan
        self.silent = silent
        self.logger_list = logger_list
        self.mutation_counts_out = mutation_counts_out
        self.n_bees = plan.total_bees
        self.initialize()

    def initialize(self):
        self.plan.validate(self.n_bees)
        self.batch_size = self.state.batch_size
        self.dev = self.state.device
        self.batch_idxs = torch.arange(self.batch_size, device=self.dev)
        self.max_n_nodes = self.state.max_n_nodes
        self.n_routes = self.state.n_routes_to_plan
        self.best_networks = torch.full((self.batch_size, self.n_routes, self.max_n_nodes), -1, device=self.dev)
        n_init_routes = self.init_network.shape[1]
        self.best_networks[:, :n_init_routes, :self.init_network.shape[-1]] = self.init_network
        self.reference_networks = self.best_networks.clone()
        self.state.set_adjustment_seed(self.reference_networks)
        self.bee_idxs = torch.arange(self.n_bees, device=self.dev)
        log.debug('starting BCO')
        exp_states = sum([[substate] * self.n_bees for substate in self.state.batch_to_list()], [])
        self.bee_states = RouteGenBatchState.batch_from_list(exp_states)
        self.plan.prepare(self.state, self.bee_states, self.cost_obj, self.method)
        self.bee_networks = self.best_networks[:, None].repeat(1, self.n_bees, 1, 1)
        self.bee_evaluation = self.evaluate(self.bee_networks)
        self.bee_costs = self.bee_evaluation.cost.reshape(self.batch_size, self.n_bees).clone()
        self.best_costs, self.best_idxs = self.bee_costs.min(1)
        self.best_evaluation = self.bee_evaluation.select(self.batch_idxs * self.n_bees + self.best_idxs)
        self.log_best(0)
        self.cost_history = torch.zeros((self.batch_size, self.n_iterations + 1), device=self.dev)
        self.cost_history[:, 0] = self.best_costs
        log_metrics(self.logger_list, SearchRecord(0.0), 0)

    def evaluate(self, bee_networks):
        self.bee_states.replace_routes(bee_networks.flatten(0, 1))
        return self.cost_obj(self.bee_states)

    def log_best(self, step):
        result = self.best_evaluation
        metrics = evaluate_network(result)
        record = SearchBestRecord(
            float(result.cost.mean()),
            {name: float(result.goals[:, i].mean()) for i, name in enumerate(result.goal_names)},
            {name: float(result.penalties[:, i].mean()) for i, name in enumerate(result.penalty_names)},
            {name: float(value.mean()) for name, value in
             monitor_metrics(metrics, [name for name in metrics if name in self.cost_obj.monitor]).items()},
        )
        log_metrics(self.logger_list, record, step)

    def mutate(self):
        chosen_route_idxs = _choose_route_indices(self.bee_networks, self.n_routes)
        _record_attempted_mutations(self.mutation_counts_out, self.plan.summary())
        return self.plan.get_mutants(self.bee_networks, chosen_route_idxs)

    def accept(self, new_bee_networks, mutation_types):
        new_evaluation = self.evaluate(new_bee_networks)
        new_bee_costs = new_evaluation.cost.reshape(self.batch_size, self.n_bees)
        # Keep only strict improvements, including equal-cost rejection.
        accepted_idxs = new_bee_costs - self.bee_costs < 0
        _record_accepted_mutations(self.mutation_counts_out, mutation_types, accepted_idxs, self.plan.group_names())
        self.bee_networks[accepted_idxs] = new_bee_networks[accepted_idxs]
        self.bee_costs[accepted_idxs] = new_bee_costs[accepted_idxs]
        self.bee_evaluation = self.bee_evaluation.updated(new_evaluation, accepted_idxs.flatten())
        return int(accepted_idxs.sum())

    def best(self, iteration):
        current_best_objective_cost, current_best_idx = self.bee_costs.min(1)
        is_improvement = current_best_objective_cost < self.best_costs
        improvement_idx = current_best_idx[is_improvement]
        self.best_costs[is_improvement] = current_best_objective_cost[is_improvement]
        self.best_networks[is_improvement] = self.bee_networks[is_improvement, improvement_idx]
        candidate_best = self.bee_evaluation.select(self.batch_idxs * self.n_bees + current_best_idx)
        self.best_evaluation = self.best_evaluation.updated(candidate_best, is_improvement)
        self.cost_history[:, iteration + 1] = self.best_costs

    def recruit(self):
        max_bee_costs, _ = self.bee_costs.max(dim=1)
        min_bee_costs, _ = self.bee_costs.min(dim=1)
        spread = max_bee_costs - min_bee_costs
        spread[spread == 0] = 1
        qualities = (max_bee_costs[:, None] - self.bee_costs) / spread[:, None]
        min_quality, _ = qualities.min(1)
        follow_probs = (-qualities + min_quality[:, None]).exp()
        # The same draw is broadcast across graphs, as in the original colony.
        are_recruiters = follow_probs < torch.rand(self.n_bees, device=self.dev)
        if not are_recruiters.any():
            return
        denom = (qualities * are_recruiters).sum(1)
        denom[denom == 0] = 1
        recruit_probs = qualities / denom[:, None]
        recruit_probs[~are_recruiters] = 0
        no_valid_recruiters = recruit_probs.sum(-1) == 0
        recruit_probs[no_valid_recruiters] = 1
        recruiters = recruit_probs.multinomial(self.n_bees, replacement=True)
        recruiters[no_valid_recruiters] = self.bee_idxs
        recruiters[are_recruiters] = self.bee_idxs[None].expand(self.batch_size, -1)[are_recruiters]
        _record_selection_stats(self.mutation_counts_out, recruiters, self.bee_costs)
        self.bee_networks = self.bee_networks[self.batch_idxs[:, None], recruiters]
        self.bee_costs = self.bee_costs[self.batch_idxs[:, None], recruiters]
        self.bee_evaluation = self.bee_evaluation.select((self.batch_idxs[:, None] * self.n_bees + recruiters).flatten())

    def run(self):
        for iteration in tqdm(range(self.n_iterations), disable=self.silent):
            accepted_count = 0
            for _ in range(self.passes_per_it):
                for _ in range(self.mod_steps_per_pass):
                    networks, kinds = self.mutate()
                    accepted_count += self.accept(networks, kinds)
                self.best(iteration)
                self.recruit()
            self.log_best(iteration + 1)
            log_metrics(self.logger_list, SearchRecord(
                accepted_count / (self.batch_size * self.n_bees * self.passes_per_it * self.mod_steps_per_pass)),
                iteration + 1)
        self.state.replace_routes(self.best_networks)
        return self.state, self.best_evaluation, self.cost_history
