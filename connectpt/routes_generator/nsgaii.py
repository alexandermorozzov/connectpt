# Based on the NSGA-II implementation of https://github.com/smkalami/nsga2-in-python
#
# Ported into connectpt from AHolliday/transit_learning (learning/nsgaii.py).
# Changes vs. the original:
#   * import paths remapped to connectpt.routes_generator;
#   * the Hydra CLI main() entry point was dropped;
#   * `nx.from_numpy_matrix` -> `nx.from_numpy_array` (removed in modern
#     networkx) in husselmann_init;
#   * `cost_obj(state)` calls go through the `_get_costs` adapter, because
#     connectpt's MultiObjectiveCostModule.forward returns a cost-helper object
#     whereas the ancestor returned `(costs, violations)` directly.
#
# NSGA-II is a multi-objective optimizer: with the default MultiObjectiveCost
# Module it optimizes two objectives (mean demand time, total route time) and
# returns a Pareto front rather than a single solution.
import logging as log
from itertools import combinations

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

from .transit_time_estimator import (RouteGenBatchState,
                                      MultiObjectiveCostModule,
                                      network_is_connected)
from .bee_colony import get_neural_variants
from . import initialization as init
from . import heuristics as hs
from . import utils as lrnu
from . import torch_utils as tu
from .models import RandomPathCombiningRouteGenerator


class NSGAII:
    """A class to implement the NSGA-II multi-objective optimization algorithm"""

    def __init__(self, cost_obj, init_models, mutators, n_iterations=200,
                 pop_size=200, p_crossover=0.9, p_mutation=0.9,
                 mutator_p_t=0.03, batch_size=None,
                 device=torch.device('cpu')):
        """Constructor for the NSGA-II object.

        ``device`` is explicit (the ancestor read a module-global set by its
        Hydra main(); callers now pass the resolved device directly).
        """

        assert isinstance(cost_obj, MultiObjectiveCostModule), \
            "cost_obj must be a MultiObjectiveCostModule"
        self.device = torch.device(device)
        self.cost_obj = cost_obj
        self.init_models = init_models
        self.mutators = mutators
        self.n_iterations = n_iterations
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        if batch_size is None:
            self.batch_size = pop_size
        else:
            self.batch_size = batch_size
        self.p_mutation = p_mutation
        self.mutator_p_t = mutator_p_t

    def _get_costs(self, state):
        """Adapter: connectpt's MultiObjectiveCostModule.forward returns a
        cost-helper object; NSGA-II expects ``(costs, violations)``."""
        cho = self.cost_obj(state)
        return self.cost_obj.get_cost(cho)

    def run(self, state: RouteGenBatchState, init_mode: str, sum_writer=None,
            *, seed_routes=None):
        """Runs the NSGA-II algorithm on a given problem.

        init_mode: 'model', 'john', or 'husselmann'.
        seed_routes: optional ``[R, L]`` or ``[1, R, L]`` tensor of initial
            routes to inject into the starting population as one explicit
            member. When given, the rest of ``pop_size`` is still filled via
            ``init_mode`` so NSGA-II keeps its diversity; the seed just
            establishes a known floor (e.g. the NX-heuristic init shared with
            the rest of the benchmark sweep). ``None`` keeps the original
            behaviour (build the entire population via ``init_mode``).
        """

        assert state.batch_size == 1, "NSGA-II only supports batch_size=1"

        # NSGA-II (Husselmann init, crossover, the hs.* mutators) mixes the
        # state's graph tensors with CPU-built network tensors, so it must run
        # on CPU to avoid cross-device errors. The cost objective is stateless
        # and evaluates fine on CPU.
        if state.device.type != 'cpu':
            state = state.to_device('cpu')

        # Initialize population
        pop = self.get_init_population(state, init_mode,
                                       seed_routes=seed_routes)

        # Non-dominated Sorting
        pop, pareto_front = non_dominated_sorting(pop)

        # Calculate Crowding Distance
        pop = calc_crowding_distance(pop, pareto_front)

        # Sort Population
        pop, pareto_front = self.sort_population(pop)

        # Truncate Extra Members, in case init returned too many
        pop, pareto_front = self.truncate_population(pop, pareto_front)

        # log statistics of the initial population
        log_stats(0, pop, pareto_front, state.symmetric_routes, sum_writer)

        # set up variables for the main loop
        p_mutators = np.ones(len(self.mutators)) / len(self.mutators)
        mutator_use_counts = np.zeros(len(self.mutators))
        max_route_len = state.max_route_len[0].cpu()
        n_nodes = state.n_nodes[0].cpu()
        adj_mat = state.street_adj[0].isfinite().cpu()
        # initialize the population state
        exp_states = [state] * self.pop_size
        pop_states = RouteGenBatchState.batch_from_list(exp_states)
        pop_states = pop_states.to_device(state.device)
        init_weights = self.cost_obj.sample_weights(self.pop_size)
        pop_states.set_cost_weights(init_weights)

        # Main Loop
        for it in tqdm(range(self.n_iterations)):

            # Crossover
            n_to_crossover = int(self.pop_size * self.p_crossover)
            # first, copy the requisite fraction of parents directly
            n_to_copy = self.pop_size - n_to_crossover
            clones = []
            while len(clones) < n_to_copy:
                candidates = np.random.choice(pop, size=4, replace=False)
                parents = [min(pair, key=self.ranking_function)
                           for pair in (candidates[:2], candidates[2:])]
                clone = parents[np.random.randint(2)]['routes'].clone()
                clones.append(clone)

            assert len(clones) == n_to_copy
            child_networks = clones

            # do crossover for the remaining children
            all_parent_pairs = []
            while len(child_networks) < self.pop_size:
                # keep doing crossover until we have the right number of children
                all_parent_pairs = []

                while len(all_parent_pairs) < n_to_crossover:
                    # perform binary tournament selection to pick the parents
                    # choose both tournament pairs at once to avoid double-picking
                    candidates = np.random.choice(pop, size=4, replace=False)
                    # winner is the one with a lower ranking (would be sorted earlier)
                    parents = [min(pair, key=self.ranking_function)
                                for pair in (candidates[:2], candidates[2:])]
                    all_parent_pairs.append(parents)

                parents1, parents2 = zip(*all_parent_pairs)
                parents1 = torch.stack([pp['routes'] for pp in parents1])
                parents2 = torch.stack([pp['routes'] for pp in parents2])
                new_child_networks = \
                    parallel_crossover(parents1, parents2, n_nodes,
                                       max_route_len, adj_mat)
                n_to_keep = self.pop_size - len(child_networks)
                child_networks += new_child_networks[:n_to_keep]

            child_networks = torch.stack(child_networks)
            # shuffle the clones and children to avoid any bias
            shuffled_idcs = torch.randperm(child_networks.shape[0])
            child_networks = child_networks[shuffled_idcs]

            # Mutation
            # child_networks = torch.stack(child_networks)
            mutated_networks, mutator_idxs = \
                self.mutate(state, child_networks, p_mutators)
            # log number of uses of each mutator
            for mi in range(self.n_mutators):
                n_uses = (mutator_idxs == mi).sum()
                mutator_use_counts[mi] += n_uses

            # Compute costs for children and mutants
            child_states = pop_states.clone()
            child_states.replace_routes(mutated_networks.to(state.device))
            child_costs, has_violation = self._get_costs(child_states)
            # replace invalid mutants with un-mutated children
            has_violation = has_violation.cpu().numpy()
            if has_violation.any():
                # replace the invalid mutants with the original children
                mutated_networks[has_violation] = child_networks[has_violation]
                # clear the mutator indices for invalid mutants
                mutator_idxs[has_violation] = -1
                child_states.replace_routes(mutated_networks.to(self.device))
                # recompute costs
                child_costs, has_violation = self._get_costs(child_states)
            if has_violation.any():
                # Children themselves are invalid -- usually because they
                # inherited from an invalid seed network injected into the
                # init pop (see `get_init_population(seed_routes=...)`).
                # Repair by replacing each invalid child with a clone of an
                # arbitrary current pop member; by iteration N>0 the pop is
                # mostly valid networks, so this almost always lands on a
                # valid donor. If even the donor is invalid we keep the
                # network anyway -- Pareto sorting dominates it out next gen.
                if pop:
                    repl_idxs = np.where(has_violation)[0]
                    for ri in repl_idxs:
                        donor = pop[np.random.randint(len(pop))]
                        mutated_networks[ri] = donor['routes']
                    mutator_idxs[has_violation] = -1
                    child_states.replace_routes(
                        mutated_networks.to(state.device))
                    child_costs, has_violation = self._get_costs(child_states)
                    has_violation = has_violation.cpu().numpy()
                if has_violation.any():
                    log.warning(
                        f"NSGA-II iter {it}: {int(has_violation.sum())}/"
                        f"{self.pop_size} children violate constraints even "
                        f"after fallback; keeping them, Pareto sorting will "
                        f"drop them.")

            # Create Merged Population
            child_costs = child_costs.cpu().numpy()
            children = [{'routes': child_networks[ii],
                         'cost': child_costs[ii],
                         'rank': None,
                         'crowding_distance': None}
                        for ii in range(self.pop_size)]
            pop = pop + children

            # Non-dominated Sorting
            pop, pareto_front = non_dominated_sorting(pop)

            # update mutator probabilities
            nondom_children = [ii - self.pop_size for ii in pareto_front[0]
                               if ii >= self.pop_size]
            p_mutators = self.update_mutation_probs(p_mutators, mutator_idxs,
                                                    nondom_children)

            # Calculate Crowding Distance
            pop = calc_crowding_distance(pop, pareto_front)

            # Sort Population
            pop, pareto_front = self.sort_population(pop)

            # Truncate Extra Members
            pop, pareto_front = self.truncate_population(pop, pareto_front)

            # Show Iteration Information
            log.debug(f'Iteration {it + 1}: '
                      f'Number of Pareto Members = {len(pareto_front[0])}')

            log_stats(it+1, pop, pareto_front, state.symmetric_routes,
                      sum_writer)

        # Pareto Front Population
        pareto_pop = [pop[i] for i in pareto_front[0]]

        return {
            'pop': pop,
            'F': pareto_front,
            'pareto_pop': pareto_pop,
            'mutator_use_counts': mutator_use_counts
        }

    def get_init_population(self, state, mode='model', *, seed_routes=None):
        """Valid modes are 'model', 'john', and 'husselmann'.

        ``seed_routes`` (optional ``[R, L]`` or ``[1, R, L]`` route tensor) is
        injected as one explicit population member before the configured
        ``mode`` fills the rest. The seed is padded/truncated to the state's
        ``max_route_len``, scored, and added to the population **unconditionally**
        -- constraint violations (``n_stops_oob`` / ``n_disconnected_demand_edges``
        / ``n_duplicate_stops``) are logged but do NOT drop the seed. NSGA-II's
        non-dominated sorting + selection will Pareto-dominate a bad seed
        naturally, but you still get the requested starting point. This matters
        for large/restrictive benchmarks like Mumford2 where the NX-heuristic
        init does not satisfy demand-coverage but is still the desired anchor.
        Use this to start NSGA-II from the same initial network as the rest of
        the benchmark sweep (e.g. the NX-heuristic init).
        """
        ii = 0
        pop = []
        # then, while the population is not full, generate random networks
        max_route_len = state.max_route_len[0].cpu()

        # Inject explicit seed network as the first population member. The
        # remaining pop_size - 1 slots are filled via the configured mode
        # below, so NSGA-II keeps its usual diversity behaviour on top of a
        # known-good (or known-bad) starting point.
        if seed_routes is not None:
            seed_t = seed_routes.detach().cpu()
            if seed_t.ndim == 2:
                seed_t = seed_t.unsqueeze(0)
            cur_len = int(seed_t.shape[-1])
            tgt_len = int(max_route_len)
            if cur_len < tgt_len:
                seed_t = torch.nn.functional.pad(
                    seed_t, (0, tgt_len - cur_len), value=-1)
            elif cur_len > tgt_len:
                seed_t = seed_t[..., :tgt_len]
            seed_state = RouteGenBatchState.batch_from_list([state])
            seed_state = seed_state.to_device(state.device)
            seed_state.replace_routes(seed_t.to(state.device))
            seed_cost, seed_invalid = self._get_costs(seed_state)
            seed_is_invalid = bool(seed_invalid[0].item())
            # Always inject the seed regardless of constraint validity. Bad
            # seeds will be Pareto-dominated and selected against in the
            # surviving-network stage; what matters is that the search starts
            # from the requested network.
            pop.append({
                'routes': seed_t[0].cpu().clone(),
                'cost': seed_cost[0].cpu().numpy(),
                'rank': None, 'crowding_distance': None,
            })
            if seed_is_invalid:
                log.warning(
                    "NSGA-II: seed_routes violates a cost-module constraint "
                    "(n_stops_oob / n_disconnected_demand_edges / "
                    "n_duplicate_stops); injecting anyway -- NSGA-II will "
                    "dominate it with valid networks in subsequent iterations. "
                    f"Seed cost={pop[-1]['cost']}")
            else:
                log.info(
                    f"NSGA-II: injected seed network into init pop "
                    f"(cost={pop[-1]['cost']})")

        if mode == 'model':
            exp_states = [state] * self.batch_size
            gen_states = RouteGenBatchState.batch_from_list(exp_states)
            gen_states = gen_states.to_device(self.device)
            # set cost function weights to a spread
            gen_weights = self.cost_obj.sample_weights(self.batch_size)
            gen_states.set_cost_weights(gen_weights)

            rpc_weights = {}
            rpc_weights['demand_time_weight'] = torch.ones(self.batch_size,
                                                           device=self.device)
            rpc_weights['route_time_weight'] = torch.zeros(self.batch_size,
                                                           device=self.device)

        while len(pop) < self.pop_size:
            if mode == 'model':
                # use greedy generation for the first batch, random for the rest
                greedy = ii == 0
                networks = []
                for model in self.init_models:
                    is_random = isinstance(model,
                                           RandomPathCombiningRouteGenerator)
                    if is_random:
                        gen_states.set_cost_weights(rpc_weights)
                    else:
                        gen_states.set_cost_weights(gen_weights)

                    if greedy and not is_random:
                        # use greedy generation for part of the batch
                        gen_states = model(gen_states, greedy=True).state
                    else:
                        gen_states = model(gen_states, greedy=False).state

                    networks += gen_states.routes
                    gen_states.clear_routes()
                networks = tu.get_batch_tensor_from_routes(
                    networks, max_route_len=max_route_len)

            elif mode == 'john':
                networks = init.john_init(state, show_pbar=True)
            elif mode == 'husselmann':
                # The ancestor hard-coded n_networks=2000, which makes the
                # initialisation unusably slow for notebook-scale pop sizes;
                # scale it to pop_size instead (still ~20x candidates after
                # the 5x mutate expansion and the validity filter).
                networks = husselmann_init(
                    state, n_networks=max(self.pop_size * 4, 100),
                    batch_size=self.batch_size)
            else:
                raise ValueError(f"Invalid initialization mode: {mode}")

            # compute costs of all
            for batch_ntwks in torch.split(networks, self.batch_size):
                batch_size = batch_ntwks.shape[0]
                cost_states = [state] * batch_size
                cost_states = RouteGenBatchState.batch_from_list(cost_states)
                cost_states = cost_states.to_device(state.device)
                cost_states.replace_routes(batch_ntwks.to(state.device))
                costs, are_invalid = self._get_costs(cost_states)
                costs = costs.cpu().numpy()
                zipped = zip(batch_ntwks.cpu(), costs, are_invalid)
                pop += [{'routes': nn.clone(), 'cost': cc,
                        'rank': None, 'crowding_distance': None}
                        for (nn, cc, is_inv) in zipped if not is_inv]

            ii += 1
            log.info(f"{len(pop)} networks generated after {ii} iterations.")

        log.info(f"Initial pop. of {len(pop)} networks generated after "
                 f"{ii} iterations.")
        return pop

    def mutate(self, state, child_networks, mutator_probs=None):
        # to start with, just ignore hyperheuristics and pick mutators randomly
        if mutator_probs is None:
            mutator_probs = np.ones(self.n_mutators) / self.n_mutators

        # torch's mult
        chosen_mutators = np.random.choice(self.n_mutators, len(child_networks),
                                           p=mutator_probs)
        # ...but don't mutate some children
        do_not_mutate = np.random.rand(self.pop_size) > self.p_mutation
        chosen_mutators[do_not_mutate] = -1

        mutated_networks = child_networks.clone()
        for mi, mutator in enumerate(self.mutators):
            mi_mask = chosen_mutators == mi
            if not mi_mask.any():
                continue
            to_mutate = child_networks[mi_mask]
            i_mutants = mutator(state, to_mutate)
            mutated_networks[mi_mask] = i_mutants

        return mutated_networks, chosen_mutators

    def update_mutation_probs(self, old_probs, mutator_idxs, success_idxs):
        # first, count how many times each mutator led to a feasible network
        use_counts = np.zeros(self.n_mutators)
        for mi in range(self.n_mutators):
            use_counts[mi] = (mutator_idxs == mi).sum()

        if (use_counts > 0).any():
            # then, count how many uses yielded non-dominated solutions
            success_counts = np.zeros(len(self.mutators))
            if len(success_idxs) > 0:
                success_mutator_idxs = mutator_idxs[np.array(success_idxs)]
                for mi in range(len(self.mutators)):
                    success_counts[mi] = (success_mutator_idxs == mi).sum()

            if (success_counts > 0).any():
                log.debug(f"Use counts: {use_counts} = {use_counts.sum()}")
                log.debug(f"Success counts: {success_counts}")
                # finally, update the probabilities
                success_ratios = success_counts / (use_counts + 1e-6)
                scores = success_ratios / (success_ratios.sum() + 1e-6)
                # remove probability mass equal to the thresholds we'll add
                scores *= (1 - self.n_mutators * self.mutator_p_t)
                # add the thresholds
                p_mutators = self.mutator_p_t + scores
                assert np.abs(p_mutators.sum() - 1.0) < 0.001, \
                    "Mutation probabilities should sum to 1!"

                # ensure the probabilities sum as close as possible to 1
                p_mutators /= p_mutators.sum()
                return p_mutators

        # if we get here, don't update the probabilities
        return old_probs

    def sort_population(self, pop):
        """Sorts a population based on rank (in asceding order) and
            crowding distance (in descending order)"""
        pop = sorted(pop, key=self.ranking_function)

        max_rank = pop[-1]['rank']
        pareto_front = []
        for r in range(max_rank + 1):
            front = [i for i in range(len(pop)) if pop[i]['rank'] == r]
            pareto_front.append(front)

        return pop, pareto_front

    def truncate_population(self, pop, pareto_front, pop_size=None):
        """Truncates a population to a given size"""

        if pop_size is None:
            pop_size = self.pop_size

        if len(pop) <= pop_size:
            return pop, pareto_front

        # Truncate the population
        pop = pop[:pop_size]

        # Remove the extra members from the Pareto fronts
        for k in range(len(pareto_front)):
            pareto_front[k] = [i for i in pareto_front[k] if i < pop_size]

        return pop, pareto_front

    @property
    def n_mutators(self):
        return len(self.mutators)

    @staticmethod
    def ranking_function(individual):
        """returns the value by which individuals are ranked in NSGA-II.

        That is, first by ascending rank, then to break rank ties, by
        descending crowding distance."""
        return (individual['rank'], -individual['crowding_distance'])


# mutation functions.  These are required to return only valid mutant networks.


def model_mutator(model, state, networks, greedy=False, weight_mode='random',
                  device=torch.device('cpu')):
    """weight_mode: 'random', 'all_passenger', or 'all_operator'"""
    # choose a random route to replace
    with torch.no_grad():
        state = model.setup_planning(state.to_device(device))
    n_routes = networks.shape[-2]
    if networks.ndim == 2:
        networks = networks.unsqueeze(0)
    n_networks = networks.shape[0]
    routes_idxs = torch.randint(n_routes, size=(n_networks,))

    # expand state to match the number of networks
    exp_states = [state] * n_networks
    exp_states = RouteGenBatchState.batch_from_list(exp_states)
    exp_states = model.setup_planning(exp_states)

    # select cost weights
    if weight_mode == 'random':
        demand_time_weights = torch.rand(n_networks, device=device)
    elif weight_mode == 'all_passenger':
        demand_time_weights = torch.ones(n_networks, device=device)
    elif weight_mode == 'all_operator':
        demand_time_weights = torch.zeros(n_networks, device=device)
    weights_dict = {
        'demand_time_weight': demand_time_weights,
        'route_time_weight': 1 - demand_time_weights
    }
    exp_states.set_cost_weights(weights_dict)

    with torch.no_grad():
        mutated = get_neural_variants(model, exp_states, networks.to(device),
                                      routes_idxs.to(device), greedy=greedy)

    return mutated.cpu()


def log_stats(it, pop, pareto_front, symmetric_routes, sum_writer):
    if not sum_writer:
        return

    pareto_costs = np.stack([pop[ii]['cost']
                                for ii in pareto_front[0]])
    # costs are in seconds but we want them in minutes
    min_costs = pareto_costs.min(axis=0) / 60

    # log the minimum value of each cost component
    sum_writer.add_scalar('min avg demand time (minutes)', min_costs[0], it)
    rtt = min_costs[1]
    if symmetric_routes:
        rtt /= 2
    sum_writer.add_scalar('min total route time (minutes)', rtt, it)
    # log an image of the pareto fronts
    img = plot_pareto_fronts(pop, pareto_front)
    sum_writer.add_image('pareto front', img, it)


def plot_pareto_fronts(pop, pareto_fronts, cost_maxes=None, n_fronts=1,
                       return_array=True):
    """By default, only plot the first pareto front (the best one)."""
    fig, ax = plt.subplots()
    if n_fronts:
        # only plot the first n_fronts of the pareto fronts
        pareto_fronts = pareto_fronts[:n_fronts]
    for front in pareto_fronts:
        if len(front) == 0:
            continue
        costs = np.stack([pop[ii]['cost'] for ii in front])
        # minutes
        costs[:, 0] /= 60
        # minutes in one direction (assumes symmetric)
        costs[:, 1] /= 120
        if cost_maxes is not None:
            ax.set_xlim(0, cost_maxes[0])
            ax.set_ylim(0, cost_maxes[1])
        ax.plot(costs[:, 0], costs[:, 1], 'o')

    if return_array:
        # convert
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        ncols, nrows = fig.canvas.get_width_height()
        img = img.reshape(nrows, ncols, 3).transpose(2, 0, 1)
        plt.close(fig)
        return img
    else:
        plt.show()


def husselmann_init(state: RouteGenBatchState, n_networks=2000,
                    n_routes_per_nodepair=50, nrpn_to_keep=15, batch_size=500):
    """Constructs a network based on the algorithm of Husselmann et al. (2024).

    nrpn_to_keep: number of routes per node pair to keep after finding them.
        called curly-l in the paper.
    """
    # first, do the K-shortest-paths maximizing missing vertices, with K=50.
    n_nodes = state.max_n_nodes
    # husselmann_init builds CPU network tensors throughout; keep the graph
    # tensors on CPU too so parallel_crossover / repair_network stay consistent.
    weighted_adj_mat = state.street_adj[0].clone().cpu()
    weighted_adj_mat[weighted_adj_mat.isinf()] = 0
    graph = nx.from_numpy_array(weighted_adj_mat.cpu().numpy())
    dmd_mat = state.demand[0].cpu().numpy()
    min_route_len = state.min_route_len[0]
    max_route_len = state.max_route_len[0]
    paths = []

    # iterate over all node pairs in both directions.  We may get different
     # results for i->j and j->i because multiple paths have the same length.
    n_comb = int((n_nodes * (n_nodes-1) / 2))
    for ii, jj in tqdm(combinations(range(n_nodes), r=2), total=n_comb):
        ij_paths = lrnu.yen_k_shortest_paths(graph, ii, jj,
                                                n_routes_per_nodepair)
        ij_paths = [pp for pp in ij_paths if len(pp) >= min_route_len and
                                                len(pp) <= max_route_len]
        if len(ij_paths) == 0:
            continue

        path_lens = [nx.path_weight(graph, pp, weight='weight')
                    for pp in ij_paths]
        sat_dmds = [dmd_mat[pp[:-1], pp[1:]].sum() for pp in ij_paths]
        # sort by ascending route length, breaking ties by descending demand
        sorted_paths = sorted(zip(path_lens, sat_dmds, ij_paths),
                                key=lambda xx: (xx[0], -xx[1]))
        kept_paths = list(zip(*sorted_paths[:nrpn_to_keep]))[2]
        paths += kept_paths

    paths = tu.get_batch_tensor_from_routes(paths, max_route_len=max_route_len)
    paths.squeeze_(0)

    # then use algorithm 4 to build a network from these
    adj_mat = weighted_adj_mat > 0
    n_routes_in_ntwk = state.n_routes_to_plan.item()
    if batch_size is None:
        batch_size = n_networks
    networks = []
    remaining = n_networks
    pbar = tqdm(total=n_networks)
    while remaining > 0:
        new_networks = parallel_crossover(paths, paths, n_nodes, max_route_len,
                                          adj_mat, n_routes_in_ntwk, batch_size)
        new_networks = new_networks[:remaining]
        networks.append(new_networks)
        remaining -= new_networks.shape[0]
        pbar.update(new_networks.shape[0])
    pbar.close()

    networks = torch.cat(networks)

    all_route_lens = (networks > -1).sum(dim=2)
    all_ntwks = [networks]
    # make n_networks be the actual number of networks, not the number we
     # tried to produce
    n_networks = networks.size(0)
    bar_format = '{l_bar}{bar}| {n:.2f}/{total_fmt}[{elapsed}<{remaining}]'
    pbar = tqdm(total=4 * n_routes_in_ntwk, bar_format=bar_format)
    for mutator in [hs.cost_based_trim, hs.cost_based_grow]:
        # apply full cost-based trim / grow
        new_mut_ntwks = networks
        old_mut_ntwks = None
        route_idcs = torch.zeros(n_networks, dtype=torch.long)
        # do this until no more changes are possible
        is_done = torch.zeros(n_networks, dtype=bool)
        pbar_count = 0
        while not is_done.all():
            old_mut_ntwks = new_mut_ntwks
            new_mut_ntwks = mutator(state, old_mut_ntwks, route_idcs)
            ntwks_are_unchanged = \
                (new_mut_ntwks == old_mut_ntwks).all(-1).all(-1)
            route_idcs[ntwks_are_unchanged] += 1
            is_done = route_idcs == n_routes_in_ntwk

            new_pbar_count = route_idcs.float().mean()
            pbar.update((new_pbar_count - pbar_count).item())
            pbar_count = new_pbar_count

            route_idcs.clamp_(0, n_routes_in_ntwk - 1)
        all_ntwks.append(new_mut_ntwks)

        # apply random all cost-based trim / grow
        new_mut_ntwks = networks
        old_mut_ntwks = None
        route_order = torch.randperm(n_routes_in_ntwk)
        meta_route_idcs = torch.zeros(n_networks, dtype=torch.long)
        route_idcs = route_order[meta_route_idcs]
        route_lens = all_route_lens.gather(1, route_idcs[:, None]).squeeze(-1)
        n_applications = (torch.rand(n_networks) * route_lens).int()
        is_done = torch.zeros(n_networks, dtype=bool)
        pbar_count = 0
        while not is_done.all():
            old_mut_ntwks = new_mut_ntwks
            new_mut_ntwks = mutator(state, old_mut_ntwks, route_idcs)
            ntwks_are_unchanged = \
                (new_mut_ntwks == old_mut_ntwks).all(-1).all(-1)
            n_applications -= 1
            # update the meta-route indices
            to_update = (n_applications == 0) | ntwks_are_unchanged
            meta_route_idcs[to_update] += 1

            # determine if we're done
            is_done = meta_route_idcs == n_routes_in_ntwk

            new_pbar_count = meta_route_idcs.float().mean()
            pbar.update((new_pbar_count - pbar_count).item())
            pbar_count = new_pbar_count

            # update the actual route indices
            meta_route_idcs.clamp_(0, n_routes_in_ntwk - 1)
            route_idcs = route_order[meta_route_idcs]

            # update the number of applications for changed route indices
            route_lens = all_route_lens.gather(1, route_idcs[:, None]).squeeze(-1)
            new_n_applications = (torch.rand(n_networks) * route_lens).int()
            n_applications[to_update] = new_n_applications[to_update]

        all_ntwks.append(new_mut_ntwks)
    pbar.close()

    # combine the above four sets of networks with the original set
    all_ntwks = torch.cat(all_ntwks, dim=0)

    return all_ntwks


def parallel_crossover(routes1, routes2, n_nodes, max_route_len, adj_mat,
                       goal_n_routes=None, n_children=None):
    # check the dimension of routes1 and routes2 to make sure they make sense
    if routes1.ndim == 2:
        routes1 = routes1.unsqueeze(0)
    if routes2.ndim == 2:
        routes2 = routes2.unsqueeze(0)

    assert routes1.shape[0] == routes2.shape[0]

    parents = torch.stack([routes1, routes2])
    parent_route_lens = (parents > -1).sum(-1)
    # do this before expanding, so we don't have to duplicate the memory
    parents[parents == -1] = n_nodes
    n_cdts = parents.shape[2]

    nodes_are_on_routes = []
    for parent in parents:
        parent_naor = tu.get_nodes_on_routes_mask(n_nodes, parent)
        nodes_are_on_routes.append(parent_naor)
    nodes_are_on_routes = torch.stack(nodes_are_on_routes, dim=0)

    if n_children is None:
        n_children = parents.shape[1]
    if n_children > parents.shape[1]:
        parents = parents.expand(-1, n_children, -1, -1)
        parent_route_lens = parent_route_lens.expand(-1, n_children, -1)
        nodes_are_on_routes = nodes_are_on_routes.expand(-1, n_children, -1, -1)

    if goal_n_routes is None:
        goal_n_routes = routes1.shape[1]

    children_route_idcs = []
    children_parent_idcs = []
    parent_routes_used = torch.zeros(parents.shape[:3], dtype=bool)
    nodes_are_in_children = torch.zeros((n_children, n_nodes+1), dtype=bool)

    ntwk_idcs = torch.arange(n_children)
    for _ in range(goal_n_routes):
        if len(children_route_idcs) == 0:
            # pick the first routes at random
            parent_idcs = torch.randint(2, size=(n_children,))
            route_idcs = torch.randint(n_cdts, size=(n_children,))

        else:
            # switch to the other parent
            parent_idcs = 1 - parent_idcs
            # identify which candidate routes overlap with current ones
            naic_mask = nodes_are_in_children[:, None]
            # n_newtorks x n_cdts x n_nodes + 1
            nodes_are_on_parent_routes = nodes_are_on_routes[parent_idcs,
                                                             ntwk_idcs]

            # score candidate routes based on how many new nodes they cover
            # n_networks x n_cdts x n_nodes + 1
            nodes_are_new = (~naic_mask) & nodes_are_on_parent_routes
            # n_networks x n_cdts
            n_new_nodes = nodes_are_new[..., :-1].sum(-1)
            chosen_parent_route_lens = parent_route_lens[parent_idcs,
                                                         ntwk_idcs]
            scores = n_new_nodes / chosen_parent_route_lens
            # make scores of non-overlapping routes -1 so they aren't chosen...
            nodes_overlap = naic_mask & nodes_are_on_parent_routes
            routes_overlap = nodes_overlap[..., :-1].any(-1)
            scores[~routes_overlap] = -1
            # ...unless all other routes are already in the network
            chosen_parent_routes_used = parent_routes_used[parent_idcs,
                                                           ntwk_idcs]
            scores[chosen_parent_routes_used] = -2

            # add a random value to the best scores to break ties randomly
            is_best_option = scores == scores.max(-1)[0][:, None]
            n_bests = is_best_option.sum()
            scores[is_best_option] = torch.rand(n_bests)

            # select highest-scoring routes to be added to the children
            route_idcs = scores.argmax(-1)
            if (scores[ntwk_idcs, route_idcs] < 0).any():
                # in general, this should not happen
                log.warning("Some selected routes did not overlap!")

        # add selected routes to the children
        children_route_idcs.append(route_idcs)
        children_parent_idcs.append(parent_idcs)
        parent_routes_used[parent_idcs, ntwk_idcs, route_idcs] = True
        # n_networks x 1 x max_route_len
        chosen_parents = parents[parent_idcs, ntwk_idcs]
        route_idcs = route_idcs[:, None, None].expand(-1, -1, max_route_len)
        chosen_routes = chosen_parents.gather(1, route_idcs).squeeze(1)
        nodes_are_in_children.scatter_(1, chosen_routes, True)

    # assemble the children into a networks tensor
    # n_networks x n_routes x max_route_len
    children_route_idcs = torch.stack(children_route_idcs, dim=1)
    children_parent_idcs = torch.stack(children_parent_idcs, dim=1)

    children = parents[children_parent_idcs, ntwk_idcs[:, None],
                       children_route_idcs]

    # set n_nodes back to -1
    children = children.clone()
    children[children == n_nodes] = -1

    for ni in range(n_children):
        # check for fully-overlapping routes and replace them
        children[ni] = replace_overlapped_routes(children[ni], parents[:, ni],
                                                 nodes_are_in_children[ni],
                                                 nodes_are_on_routes[:, ni],
                                                 parent_routes_used[:, ni])

    children[children == n_nodes] = -1

    # do repairs
    repaired_children = []
    repaired_scores = []
    for ni in range(n_children):
        child = repair_network(children[ni].clone(), max_route_len, adj_mat)
        nodes_are_on_routes = tu.get_nodes_on_routes_mask(n_nodes, child)
        nodes_are_in_child = nodes_are_on_routes.any(-2).squeeze(0)[..., :-1]

        is_connected = network_is_connected(child, n_nodes)

        if nodes_are_in_child.all() and is_connected.all():
            # the child is valid, so return it
            repaired_children.append(child)
            repaired_scores.append(scores[ni])

    # return it
    if len(repaired_children) > 0:
        # all children were invalid, so return the original children
        children = torch.stack(repaired_children)
    else:
        children = children[:0]
    return children


def crossover(routes1, routes2, n_nodes, max_route_len, adj_mat,
              goal_n_routes=None):
    # convert routes1 and routes2, which are tensors, to lists of tensors
    if goal_n_routes is None:
        goal_n_routes = len(routes1)

    # seed the child with a random route
    parent_index = np.random.randint(0, 2, size=1).item()
    parents = torch.stack([routes1, routes2])

    nodes_are_on_parent_routes = tu.get_nodes_on_routes_mask(n_nodes, parents)
    parent_routes_nodes_would_be_new = nodes_are_on_parent_routes.clone()
    parent_routes_chosen = torch.zeros(parents.shape[:2], dtype=bool)

    cur_parent = parents[parent_index]
    # select a random route and remove it from the parent
    route_index = np.random.randint(0, len(cur_parent), size=1).item()
    # route = cur_parent.pop(route_index)
    route = cur_parent[route_index]
    child = [route]
    nodes_are_in_child = torch.zeros(n_nodes + 1, dtype=bool)
    nodes_are_in_child[route] = True
    # treat dummy node as always covered
    nodes_are_in_child[-1] = True
    # set the nodes in the new route to False for both parents
    parent_routes_nodes_would_be_new[:, :, route] = False
    parent_route_lens = (parents > -1).sum(-1)
    parent_routes_chosen[parent_index, route_index] = True

    while len(child) < goal_n_routes:
        # switch to the other parent
        parent_index = 1 - parent_index
        cur_parent = parents[parent_index]
        n_new_nodes = \
            parent_routes_nodes_would_be_new[parent_index, :, :-1].sum(-1)

        scores = n_new_nodes / parent_route_lens[parent_index]
        nodes_overlap = nodes_are_in_child & \
            nodes_are_on_parent_routes[parent_index]
        routes_overlap = nodes_overlap[..., :-1].any(-1)
        scores[~routes_overlap] = -1
        # *never* choose routes that are already in the child
        scores[parent_routes_chosen[parent_index]] = -2

        route_index = torch.argmax(scores)
        route = cur_parent[route_index]
        child.append(route)
        nodes_are_in_child[route] = True
        parent_routes_chosen[parent_index, route_index] = True
        parent_routes_nodes_would_be_new[:, :, route] = False

    # conditions for feasibility:
    # - all routes are simple paths (true if true of parents)
    # - all routes are of valid lengths (true if true of parents)
    # - number of routes is correct (true by construction)
    # - all nodes are covered (may or may not be true)
    # - graph must be connected (may or may not be true, but implies above)
    # So we only need to check for the last one
    child = torch.stack(child)

    child = replace_overlapped_routes(child, parents, nodes_are_in_child,
                                      nodes_are_on_parent_routes,
                                      parent_routes_chosen)

    # try to cover any still-uncovered nodes by extending routes to them
        # (if all nodes are covered, this does nothing)
    # try extending routes from both ends
    child = repair_network(child, max_route_len, adj_mat)

    if nodes_are_in_child[:-1].all() and \
        network_is_connected(child, n_nodes).all():
        # TODO this might be a problem, if we want to ignore edges with
            # no demand
        # we have a valid child
        return child
    else:
        # crossover failed
        return None


def replace_overlapped_routes(child, parents, nodes_are_in_child,
                              nodes_are_on_parent_routes,
                              parent_routes_chosen):
    # check for fully-overlapping routes
    # n_routes x n_nodes
    n_nodes = nodes_are_in_child.shape[-1] - 1
    children_nodes_on_routes = tu.get_nodes_on_routes_mask(n_nodes, child[None])
    children_nodes_on_routes = children_nodes_on_routes.squeeze(0)
    # n_routes x n_routes x n_nodes
    cnor_all_ors = children_nodes_on_routes[:, None] | \
                    children_nodes_on_routes[None]
    # n_routes x n_routes
    i_contains_j = (cnor_all_ors == children_nodes_on_routes[:, None]).all(-1)
    i_contains_j.fill_diagonal_(False)

    # identify uncontained, unused routes we could replace them with
    # do this in a for-loop because the tensors get too big otherwise
    # n_parents x parent_size x n_routes x n_nodes
    xx = (nodes_are_on_parent_routes[:, :, None] |
          children_nodes_on_routes[None, None])
    # n_parents x parent_size x n_routes
    xx = (xx == children_nodes_on_routes[None, None]).all(-1)
    # n_parents x parent_size
    not_subset_of_any_child_route = ~ xx.any(-1)
    # n_parents x parent_size
    is_valid_choice = not_subset_of_any_child_route & (~ parent_routes_chosen)

    # parent_size * n_parents x max_route_len
    all_cdt_routes = parents.flatten(0, 1)
    # parent_size * n_parents
    is_valid_choice = is_valid_choice.flatten(0, 1)
    # parent_size * n_parents x n_nodes
    flat_nodes_are_on_routes = nodes_are_on_parent_routes.flatten(0, 1)

    # now try to replace any route that is fully overlapped by another route
    has_been_replaced = torch.zeros(child.shape[-2], dtype=bool)
    for ri, rj in i_contains_j.nonzero():
        if ri == rj:
            # it just matched with itself, doesn't count
            continue

        if has_been_replaced[ri] or has_been_replaced[rj]:
            # this route has already been replaced, so skip it
            continue

        if not is_valid_choice.any():
            # can't replace any route, so don't do anything
            break

        # first check if rj is a strict subsequence of ri
        route1 = child[ri]
        route1 = route1[route1 > -1]
        route2 = child[rj]
        route2 = route2[route2 > -1]

        for r2_order in [route2, route2.flip(0)]:
            start_idx = torch.where(route1 == r2_order[0])[0].item()
            end_idx = start_idx + len(r2_order)
            same_seq = (end_idx <= len(route1)) and \
                       (route1[start_idx:end_idx] == r2_order).all()
            if same_seq:
                break

        if not same_seq:
            # route 2 is *not* a subsequence of route 1, just a subset
            continue

        # route 2 is a strict subsequence of route 1, so replace it
        idx = is_valid_choice.float().multinomial(1).squeeze(-1)
        nodes_are_on_new_route = flat_nodes_are_on_routes[idx]
        xx = (flat_nodes_are_on_routes | nodes_are_on_new_route[None])
        overlapped_by_new = (xx == nodes_are_on_new_route[None]).all(-1)
        is_valid_choice[idx] = False
        is_valid_choice &= ~overlapped_by_new

        has_been_replaced[rj] = True
        # new_route = new_route.clone()
        # new_route[new_route == n_nodes] = -1
        child[rj] = all_cdt_routes[idx].clone()
        nodes_are_in_child[child[rj]] = True
    return child


def repair_network(network, max_route_len, adj_mat):
    n_nodes = adj_mat.shape[0]
    nodes_are_on_routes = tu.get_nodes_on_routes_mask(n_nodes, network)
    nodes_are_in_network = nodes_are_on_routes.any(-2).squeeze(0)[..., :-1]
    if nodes_are_in_network.all():
        # nothing to do
        return network

    # try extending routes from both ends
    route_lens = (network > -1).sum(dim=1, keepdim=True)
    for ext_from in ('end', 'start'):
        # get the terminal nodes at the end we're extending from
        if ext_from == 'end':
            terminal_nodes = network.gather(1, route_lens - 1).squeeze(1)
        else:
            terminal_nodes = network[:, 0]

        # loop over terminal nodes, extending if possible
        ii = 0
        while not nodes_are_in_network.all():
            # for each terminal node:
            term = terminal_nodes[ii]
            are_neighbours = adj_mat[term]
            is_valid_ext = are_neighbours & ~nodes_are_in_network
            network_route_len = (network[ii] > -1).sum()
            if network_route_len < max_route_len and is_valid_ext.any():
                # pick a node that's not covered and make it the new term
                _, new_term = is_valid_ext.max(0)
                if ext_from == 'end':
                    network[ii, route_lens[ii]] = new_term
                else:
                    network[ii, 1:] = network[ii, :-1].clone()
                    network[ii, 0] = new_term
                terminal_nodes[ii] = new_term
                # mark the new node as covered
                nodes_are_in_network[new_term] = True
                # extend the route's length
                route_lens[ii] += 1
            else:
                # can't extend this route any further, so move on to next
                ii += 1

            if ii == len(network):
                # we've covered all the routes
                break

    return network


def non_dominated_sorting(pop):
    """Perform Non-dominated Sorting on a Population"""
    pop_size = len(pop)

    # Initialize Domination Stats
    domination_set = [[] for _ in range(pop_size)]
    dominated_count = [0 for _ in range(pop_size)]

    # Initialize Pareto Fronts
    pareto_front = [[]]

    # Find the first Pareto Front
    for i in range(pop_size):
        for j in range(i+1, pop_size):
            # Check if i dominates j
            if dominates(pop[i], pop[j]):
                domination_set[i].append(j)
                dominated_count[j] += 1

            # Check if j dominates i
            elif dominates(pop[j], pop[i]):
                domination_set[j].append(i)
                dominated_count[i] += 1

        # If i is not dominated at all
        if dominated_count[i] == 0:
            pop[i]['rank'] = 0
            pareto_front[0].append(i)

    # Pareto Counter
    k = 0

    while True:

        # Initialize the next Pareto front
        Q = []

        # Find the members of the next Pareto front
        for i in pareto_front[k]:
            for j in domination_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    pop[j]['rank'] = k + 1
                    Q.append(j)

        # Check if the next Pareto front is empty
        if not Q:
            break

        # Append the next Pareto front
        pareto_front.append(Q)

        # Increment the Pareto counter
        k += 1

    return pop, pareto_front


def dominates(pp, qq):
    """Checks if p dominates q"""
    return all(pp['cost'] <= qq['cost']) and any(pp['cost'] < qq['cost'])


def calc_crowding_distance(pop, pareto_front):
        """Calculate the crowding distance for a given population"""

        # Number of Pareto fronts (ranks)
        parto_count = len(pareto_front)

        # Number of Objective Functions
        n_obj = len(pop[0]['cost'])

        # Iterate over Pareto fronts
        for k in range(parto_count):
            costs = np.array([pop[i]['cost'] for i in pareto_front[k]])
            n = len(pareto_front[k])
            d = np.zeros((n, n_obj))

            # Iterate over objectives
            for j in range(n_obj):
                idx = np.argsort(costs[:, j])
                d[idx[0], j] = np.inf
                d[idx[-1], j] = np.inf

                for i in range(1, n-1):
                    d[idx[i], j] = costs[idx[i+1], j] - costs[idx[i-1], j]
                    denom = costs[idx[-1], j] - costs[idx[0], j]
                    # add epsilon to avoid division by 0
                    d[idx[i], j] /= denom + 1e-6

            # Calculate Crowding Distance
            for i in range(n):
                pop[pareto_front[k][i]]['crowding_distance'] = sum(d[i, :])

        return pop
