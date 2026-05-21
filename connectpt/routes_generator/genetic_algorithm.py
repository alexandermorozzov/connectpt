# Copyright 2023 Andrew Holliday
#
# This file is part of the Transit Learning project.
#
# Transit Learning is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Transit Learning is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Transit Learning. If not, see <https://www.gnu.org/licenses/>.
#
# Ported into connectpt from AHolliday/transit_learning
# (learning/genetic_algorithm.py). Changes vs. the original:
#   * import paths remapped to connectpt.routes_generator;
#   * the Hydra CLI main() entry point was dropped;
#   * the no-model initial population is now seeded from the `init_network`
#     supplied by `test_method` (the original called the now-absent
#     `bee_colony.build_init_network`).

import logging as log
import copy

import torch
from tqdm import tqdm

from .torch_utils import reconstruct_all_paths, get_batch_tensor_from_routes
from .transit_time_estimator import RouteGenBatchState
from .initialization import get_direct_sat_dmd
from .bee_colony import get_bee_1_variants


def run(state, cost_obj, pop_size=10, shorten_prob=0.2, n_iterations=400,
        n_type1_mut=None, silent=False, init_network=None,
        force_linking_unlinked=False, mut_model=None, sum_writer=None,
        mutate_parents=True):
    """A genetic algorithm based on the BCO method of Nikolic and Teodorovic
        (2013), but modified and expanded.

    state -- A RouteGenBatchState object representing the initial state.
    cost_obj -- A function that determines the cost (badness) of a scenario. In
        the paper, this is wieghted total travel time.
    pop_size -- The size of the population.
    shorten_prob -- The probability that type-2 mutation will shorten a route,
        called P in the BCO paper. In their experiments, they use 0.2.
    n_iterations -- The number of iterations to perform, called IT in the paper.
    n_type1_mut -- There are 2 types of mutators used in the algorithm, which
        modify the solution in different ways. This parameter determines the
        balance between them. The paper isn't clear how many of each they use,
        so by default we make it half-and-half.
    silent -- if true, no tqdm output or printing.
    init_network -- the initial route network (a [batch, n_routes, len] tensor)
        used to seed every individual when no neural mutator is given. Supplied
        by `test_method` from `cfg.init`.
    force_linking_unlinked -- if true, force the type-1 mutator to link
        unlinked nodes.
    mut_model -- if a torch model is provided, use it as the type-1 mutator.
    sum_writer -- if provided, use it to log metrics.
    """
    if n_type1_mut is None:
        # children all get mutated before selection
        n_type1_mut = (1 + mutate_parents) * pop_size // 2

    assert state.batch_size == 1, "only batch size 1 is supported"
    batch_size = state.batch_size

    dev = state.device
    max_n_nodes = state.max_n_nodes

    # get all shortest paths
    shortest_paths, _ = reconstruct_all_paths(state.nexts)

    # "multiply" the batch for the individuals
    exp_states = [state] * pop_size * (1 + mutate_parents)
    states = RouteGenBatchState.batch_from_list(exp_states)
    if mut_model is not None:
        states = mut_model.setup_planning(states)

    demand = torch.zeros((batch_size, max_n_nodes + 1, max_n_nodes + 1),
                         device=dev)
    demand[:, :-1, :-1] = state.demand
    n_routes = state.n_routes_to_plan

    # generate the initial population
    log.info("generating initial scenario")
    exp_states = [state] * pop_size
    init_states = RouteGenBatchState.batch_from_list(exp_states)
    if mut_model is None:
        # heuristic path: seed every individual from the init_network supplied
        # by test_method (built from cfg.init). The ancestor repo built this
        # with bee_colony.build_init_network, which no longer exists.
        assert init_network is not None, \
            "genetic_algorithm.run requires an init_network when mut_model " \
            "is None"
        init_scenario = tensor_scenario_to_tuples(init_network)
        population = [copy.copy(init_scenario) for _ in range(pop_size)]
        init_states.replace_routes(population)
    else:
        plan_out = mut_model(init_states, greedy=False)
        population = [tensor_scenario_to_tuples(scen)
                      for scen in plan_out.state.routes]

    result = cost_obj(init_states)
    best_cost, best_index = result.cost.min(0)
    best_scenario = population[best_index]

    # delete this
    population = [copy.copy(best_scenario) for _ in range(pop_size)]

    ind_costs = result.cost
    ind_metrics = result.get_metrics_tensor()

    # compute can-be-directly-satisfied demand matrix
    direct_sat_dmd = get_direct_sat_dmd(demand, shortest_paths,
                                        cost_obj.symmetric_routes)

    # set up required matrices
    street_node_neighbours = (state.street_adj.isfinite() &
                              (state.street_adj > 0))[0]

    log.debug("starting GA")

    metric_names = cost_obj.get_metric_names()

    def batched_cost_fn(individuals):
        n_real = len(individuals)
        if n_real < states.batch_size:
            # pad the batch with the best scenario
            dummy = [[0, 1]] * n_routes
            individuals = individuals + \
                [dummy] * (states.batch_size - len(individuals))
        states.replace_routes(individuals)

        result = cost_obj(states)

        other_metrics = result.get_metrics_tensor()[:n_real]
        return result.cost[:n_real], other_metrics

    cost_history = torch.zeros((batch_size, n_iterations + 1), device=dev)
    cost_history[:, 0] = best_cost

    for iteration in tqdm(range(n_iterations), disable=silent):
        set_population = [set(individual) for individual in population]
        # crossover
        # CPU tensor: i1 only indexes Python lists and is compared with the
        # CPU-side i2 below -- keeping it on `dev` would mismatch i2.
        first_parents = torch.arange(pop_size)

        children = []
        # perform crossover
        for i1 in first_parents:
            i2 = i1
            # avoid self-mating
            while i2 == i1:
                i2 = torch.randint(pop_size - 1, size=(1,))
                if i2 >= i1:
                    i2 += 1

            parent1 = copy.copy(population[i1])
            parent2 = copy.copy(population[i2])
            child = []
            adding_duplicates = False
            child_covers_nodes = torch.zeros(max_n_nodes, dtype=bool)

            while len(child) < n_routes:
                # select half of the routes from each parent
                if len(parent1) == 0 and len(parent2) == 0:
                    # we've run out of routes to add, so just add duplicates
                    adding_duplicates = True
                    parent1 = copy.copy(population[i1])
                    parent2 = copy.copy(population[i2])

                if len(parent1) == 0:
                    parent = parent2
                elif len(parent2) == 0:
                    parent = parent1
                else:
                    parent = [parent1, parent2][len(child) % 2]
                # select a route that isn't already in the offspring
                all_covered = child_covers_nodes.all()
                if all_covered or len(child) == 0:
                    # just choose randomly
                    route_idx = torch.randint(len(parent), size=(1,))
                else:
                    # pick one that both overlaps with existing routes and
                     # extends coverage
                    # iterating over parent's routes in random order:
                    for route_idx in torch.randperm(len(parent)):
                        # get the next route
                        route = list(parent[route_idx])
                        child_covers_nodes_on_route = child_covers_nodes[route]
                        # check if it overlaps with existing routes
                        overlaps = child_covers_nodes_on_route.any()
                        # check if it extends coverage of child
                        extends_coverage = ~(child_covers_nodes_on_route.all())
                        if overlaps and extends_coverage:
                            # we found a satisfactory route, so stop searching
                            break
                        else:
                            route = None

                    if route is None:
                        # we couldn't find a route that both extends coverage
                         # and is connected to existing routes, so just choose
                         # randomly
                        route_idx = torch.randint(len(parent), size=(1,))

                # don't choose the same route again
                route = parent.pop(route_idx)
                if adding_duplicates or route not in child:
                    child.append(route)
                    child_covers_nodes[list(route)] = True

            assert len(child) == n_routes, "could not generate child!"
            children.append(child)

        # mutation
        if mutate_parents:
            # mutate the parents
            children += population

        children = mutate(children, n_type1_mut, direct_sat_dmd,
                          shorten_prob, street_node_neighbours, shortest_paths,
                          force_linking_unlinked, mut_model, states)

        non_duplicate_children = []
        for child in children:
            if set(child) not in set_population:
                non_duplicate_children.append(child)

        children = non_duplicate_children

        new_costs, new_metrics = batched_cost_fn(children)

        # selection: combine parents and children and take the top pop-size to
         # the next gen.
        # combine...
        population += children
        all_costs = torch.cat((ind_costs, new_costs))
        all_metrics = torch.cat((ind_metrics, new_metrics))
        # record the best solution found so far
        best_cost, best_idx = all_costs.min(0)
        best_metrics = all_metrics[best_idx]
        # write to column iteration+1: column 0 holds the initial best cost,
        # so the final column (n_iterations) gets filled on the last pass.
        # The ancestor wrote column `iteration`, leaving the last column 0.
        cost_history[:, iteration + 1] = best_cost
        best_scenario = population[best_idx]

        pop_and_children_data = list(zip(population + children,
                                         all_costs, all_metrics))
        # ...sort and select the best pop_size...
        sorted_pacd = sorted(pop_and_children_data, key=lambda x: x[1])
        top_half = sorted_pacd[:pop_size]
        # ...and finally unpack
        population, ind_costs, ind_metrics = zip(*top_half)
        population = list(population)
        ind_costs = torch.stack(ind_costs)
        ind_metrics = torch.stack(ind_metrics)

        if sum_writer is not None:
            # log the various metrics
            for name, val in zip(metric_names, best_metrics.unbind(-1)):
                sum_writer.add_scalar(f'best {name}', val, iteration)

    # return the best solution
    state.replace_routes(best_scenario)
    return state, cost_history


def mutate(population, n_type1, direct_sat_dmd, shorten_prob,
           street_node_neighbours, shortest_paths,
           force_linking_unlinked, model=None, env_state=None):
    population = copy.deepcopy(population)
    n_routes = len(population[0])
    chosen_route_idxs = torch.randint(n_routes, size=(len(population),))
    # after this, "individuals" have only the unmodified routes
    routes_to_modify = []
    for individual, chosen_route_idx in zip(population, chosen_route_idxs):
        route = individual.pop(chosen_route_idx)
        routes_to_modify.append(route)

    # select which individuals will get type-1 and type-2 mutations
    shuffled_idxs = torch.randperm(len(population))
    type1_idxs = shuffled_idxs[:n_type1]
    type2_idxs = shuffled_idxs[n_type1:]

    # modify type 1 routes
    env_state.replace_routes(population)
    if model is not None:
        # run it on all individuals...
        model.step(env_state, greedy=False)
        # insert the type-1 modified scenarios back into the population
        for ii in type1_idxs:
            population[ii] = [route_tensor_to_tuple(route)
                              for route in env_state.routes[ii]]
    else:
        # here we plan new routes for all individuals but only take the type-1
         # selections...
        modroutes_tensor = get_batch_tensor_from_routes(routes_to_modify,
                                                        device=env_state.device)
        new_type1_routes = get_bee_1_variants(env_state, modroutes_tensor,
                                              direct_sat_dmd, shortest_paths,
                                              force_linking_unlinked)
        new_type1_routes = [route_tensor_to_tuple(route)
                            for route in new_type1_routes[0]]
        # insert the type-1 modified routes back into their individuals
        for ii in type1_idxs:
            population[ii].append(new_type1_routes[ii])

    # modify type 2 routes
    # ...whereas here, we only plan new routes for the type-2 individuals
    type2_routes_to_modify = [routes_to_modify[ii] for ii in type2_idxs]
    new_type2_routes = get_type2_variants(type2_routes_to_modify,
                                          shorten_prob, street_node_neighbours,
                                          env_state.min_route_len[0],
                                          env_state.max_route_len[0])
    # insert the type-2 modified routes back into their individuals
    for ii, new_route in zip(type2_idxs, new_type2_routes):
        population[ii].append(new_route)

    assert all([len(individual) == n_routes for individual in population])

    return population


def get_type2_variants(routes_to_modify, shorten_prob, are_neighbours,
                       min_route_len, max_route_len):
    """
    routes_to_modify: a list of tuple routes to modify
    shorten_prob: a scalar probability of shortening each route
    are_neighbours: a batch_size x n_nodes x n_nodes boolean tensor of whether
        each node is a neighbour of each other node

    """
    modified_routes = []
    for route in routes_to_modify:
        # select which terminal will be modified
        modify_start = torch.rand(1) > 0.5
        if modify_start:
            extend_choices = are_neighbours[route[0]]
        else:
            extend_choices = are_neighbours[route[-1]]
        # can't add nodes already on the route
        extend_choices[list(route)] = False

        can_extend = extend_choices.any() and len(route) < max_route_len
        can_shorten = len(route) > min_route_len
        if not (can_extend or can_shorten):
            # we can't modify this route, so just leave it
            modified_routes.append(route)
            continue

        if not can_extend or (can_shorten and shorten_prob > torch.rand(1)):
            # cut off the chosen end
            if modify_start:
                route = route[1:]
            else:
                route = route[:-1]
        elif can_extend:
            # add a random node at the chosen end
            extension = extend_choices.to(dtype=torch.float32).multinomial(1)
            if modify_start:
                route = (extension.item(),) + route
            else:
                route = route + (extension.item(),)
        # else, do nothing, as we can neither extend nor shorten

        modified_routes.append(route)

    return modified_routes


def tensor_scenario_to_tuples(scenario):
    if scenario.ndim == 3:
        assert scenario.shape[0] == 1, "only batch size 1 is supported"
        scenario = scenario[0]
    return [route_tensor_to_tuple(route) for route in scenario]


def route_tensor_to_tuple(route):
    return tuple([ss.item() for ss in route[route > -1]])
