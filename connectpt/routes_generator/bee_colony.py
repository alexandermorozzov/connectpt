import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import logging as log

import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
import hydra

from .models import check_extensions_add_connections
from . import torch_utils as tu
from .citygraph_dataset import get_dataset_from_config
from .transit_time_estimator import (
    ROUTE_ACTION_HALT,
    RouteGenBatchState,
)
from .import utils as lrnu
from .initialization import get_direct_sat_dmd


MUTATION_TYPE_NAMES = (
    'type1', 'type2', 'type3', 'type4', 'type5', 'type6', 'type7')


def _reverse_padded_routes(routes):
    """Reverse the valid part of each padded route and keep -1 padding at end."""
    route_lens = (routes > -1).sum(dim=-1)
    max_n_nodes = routes.shape[-1]
    positions = torch.arange(max_n_nodes, device=routes.device)
    gather_pos = (route_lens[..., None] - 1 - positions).clamp(min=0)
    reversed_routes = routes.gather(-1, gather_pos)
    reversed_routes[positions >= route_lens[..., None]] = -1
    return reversed_routes


def needleman_wunsch(scores, gap=0.1):
    """Compute batched Needleman-Wunsch alignment scores."""
    if scores.ndim != 3:
        raise ValueError(f"Expected scores to have shape (B, N, M), got {scores.shape}")
    if not 0 <= gap < 1:
        raise ValueError(f"gap must be in [0, 1), got {gap}")

    batch_size, n_rows, n_cols = scores.shape
    dp = torch.zeros(batch_size, n_rows + 1, n_cols + 1, device=scores.device,
                     dtype=scores.dtype)
    shifted_scores = scores - gap

    for diag_idx in range(2, n_rows + n_cols + 1):
        row_min = max(1, diag_idx - n_cols)
        row_max = min(n_rows, diag_idx - 1)
        row_idxs = torch.arange(row_min, row_max + 1, device=scores.device)
        col_idxs = diag_idx - row_idxs
        up = dp[:, row_idxs - 1, col_idxs]
        left = dp[:, row_idxs, col_idxs - 1]
        diag = dp[:, row_idxs - 1, col_idxs - 1] + \
            shifted_scores[:, row_idxs - 1, col_idxs - 1]
        dp[:, row_idxs, col_idxs] = torch.maximum(torch.maximum(up, left), diag)

    return dp[:, -1, -1]


def _get_alignment_scores(candidate_routes, reference_routes, symmetric_routes,
                          gap):
    """Return best alignment scores and route lengths for each route pair."""
    if candidate_routes.shape != reference_routes.shape:
        raise ValueError(
            "candidate_routes and reference_routes must have the same shape, "
            f"got {candidate_routes.shape} and {reference_routes.shape}"
        )

    max_n_nodes = candidate_routes.shape[-1]
    flat_candidates = candidate_routes.reshape(-1, max_n_nodes)
    flat_references = reference_routes.reshape(-1, max_n_nodes)

    cand_valid = flat_candidates > -1
    ref_valid = flat_references > -1
    cand_lens = cand_valid.sum(dim=-1)
    ref_lens = ref_valid.sum(dim=-1)

    # Fast path for identical route pairs (the common case: a mutation step
    # changes one route per network, the rest still equal the reference).
    # With gap == 0 the NW score of an identical pair is exactly its length
    # (a sum of 1.0 match scores -- exact in fp32), and the reversed
    # alignment can never exceed it, so the maximum is the length itself.
    # Only the non-identical pairs go through the expensive DP; NW rows are
    # independent, so subsetting yields bit-identical scores for them.
    if gap == 0.0:
        identical = (flat_candidates == flat_references).all(dim=-1)
        if identical.any():
            alignment_scores = cand_lens.to(torch.float32)
            differing = ~identical
            if differing.any():
                diff_scores, _, _ = _get_alignment_scores(
                    flat_candidates[differing], flat_references[differing],
                    symmetric_routes, gap)
                alignment_scores = alignment_scores.clone()
                alignment_scores[differing] = diff_scores
            return alignment_scores, cand_lens, ref_lens

    pairwise_matches = flat_candidates[:, :, None] == flat_references[:, None, :]
    valid_pairwise_matches = pairwise_matches & cand_valid[:, :, None] & ref_valid[:, None, :]
    alignment_scores = needleman_wunsch(valid_pairwise_matches.to(torch.float32),
                                        gap=gap)

    if symmetric_routes:
        reversed_refs = _reverse_padded_routes(flat_references)
        reversed_valid = reversed_refs > -1
        reversed_matches = flat_candidates[:, :, None] == reversed_refs[:, None, :]
        valid_reversed_matches = reversed_matches & cand_valid[:, :, None] & \
            reversed_valid[:, None, :]
        reversed_scores = needleman_wunsch(
            valid_reversed_matches.to(torch.float32), gap=gap)
        alignment_scores = torch.maximum(alignment_scores, reversed_scores)

    cand_lens = cand_valid.sum(dim=-1)
    ref_lens = ref_valid.sum(dim=-1)
    return alignment_scores, cand_lens, ref_lens


def get_current_adjustment_degrees(candidate_routes, reference_routes,
                                   symmetric_routes, gap=0.1):
    """Return the current normalized alignment-based adjustment degree."""
    alignment_scores, cand_lens, ref_lens = _get_alignment_scores(
        candidate_routes, reference_routes, symmetric_routes, gap=gap)
    norm = torch.maximum(cand_lens, ref_lens).to(torch.float32)
    match_score = max(1.0 - gap, 1e-6)
    norm = norm * match_score
    both_empty = (cand_lens == 0) & (ref_lens == 0)
    norm[both_empty] = 1.0

    similarities = alignment_scores / norm
    similarities = similarities.clamp(min=0.0, max=1.0)
    adjustment_degrees = 1.0 - similarities
    adjustment_degrees[both_empty] = 0.0

    return adjustment_degrees.reshape(candidate_routes.shape[:-1])


def get_paper_adjustment_degrees(candidate_routes, reference_routes,
                                 symmetric_routes):
    """Return the paper-style adjustment degree based on LCS F-measure."""
    lcs_lengths, cand_lens, ref_lens = _get_alignment_scores(
        candidate_routes, reference_routes, symmetric_routes, gap=0.0)
    both_empty = (cand_lens == 0) & (ref_lens == 0)
    either_empty = (cand_lens == 0) | (ref_lens == 0)

    cand_lens = cand_lens.to(torch.float32)
    ref_lens = ref_lens.to(torch.float32)
    s_candidate = lcs_lengths / cand_lens.clamp_min(1.0)
    s_reference = lcs_lengths / ref_lens.clamp_min(1.0)

    eps = 1e-8
    valid = ~(either_empty | (s_candidate <= eps) | (s_reference <= eps))
    similarities = torch.zeros_like(lcs_lengths)
    theta = torch.zeros_like(lcs_lengths)
    theta[valid] = s_reference[valid] / s_candidate[valid]
    theta_sq = theta.square()
    numer = (1.0 + theta_sq) * s_candidate * s_reference
    denom = s_reference + theta_sq * s_candidate
    similarities[valid] = numer[valid] / denom[valid].clamp_min(eps)
    similarities = similarities.clamp(min=0.0, max=1.0)
    similarities[both_empty] = 1.0

    adjustment_degrees = 1.0 - similarities
    adjustment_degrees[both_empty] = 0.0
    return adjustment_degrees.reshape(candidate_routes.shape[:-1])


def get_adjustment_degrees(candidate_routes, reference_routes, symmetric_routes,
                           gap=0.1, mode='current'):
    """Return route-wise adjustment degree in [0, 1], 0 means unchanged."""
    if mode == 'current':
        return get_current_adjustment_degrees(
            candidate_routes,
            reference_routes,
            symmetric_routes,
            gap=gap,
        )
    if mode == 'paper':
        return get_paper_adjustment_degrees(
            candidate_routes,
            reference_routes,
            symmetric_routes,
        )
    raise ValueError(
        f"Unknown adjustment degree mode '{mode}'. Expected 'current' or 'paper'."
    )


def get_adjustment_penalties(adjustment_degrees, objective='raw', target=0.2):
    """Convert raw adjustment degrees into the penalty used in the objective."""
    if objective == 'raw':
        return adjustment_degrees
    if objective == 'target':
        if not 0.0 <= target <= 1.0:
            raise ValueError(
                f"adjustment_degree_target must be in [0, 1], got {target}"
            )
        target_tensor = adjustment_degrees.new_tensor(target)
        return (adjustment_degrees - target_tensor).abs()
    if objective == 'cap':
        # One-sided (hinge) penalty: penalize only EXCEEDING the target, i.e.
        # treat `target` as an upper bound the routes should not pass. No
        # penalty (and no pull) while adjustment <= target.
        if not 0.0 <= target <= 1.0:
            raise ValueError(
                f"adjustment_degree_target must be in [0, 1], got {target}"
            )
        target_tensor = adjustment_degrees.new_tensor(target)
        return (adjustment_degrees - target_tensor).clamp(min=0.0)
    if objective == 'cap_sq':
        # Quadratic one-sided penalty: max(0, adj - target)**2. Same upper-bound
        # semantics as 'cap' (no penalty below target), but the gradient
        # 2*(adj - target) GROWS with the overshoot -- gentle near the target
        # (does not swamp the main objective / keeps return variance low) and
        # firm on large excess. Unlike linear 'cap' (constant gradient = W),
        # the optimal adjustment depends on the target, so target/W become real
        # controllability levers under conditioning.
        if not 0.0 <= target <= 1.0:
            raise ValueError(
                f"adjustment_degree_target must be in [0, 1], got {target}"
            )
        target_tensor = adjustment_degrees.new_tensor(target)
        return (adjustment_degrees - target_tensor).clamp(min=0.0) ** 2
    raise ValueError(
        f"Unknown adjustment degree objective '{objective}'. "
        "Expected 'raw', 'target', 'cap', or 'cap_sq'."
    )


def _ensure_mutation_stats_bucket(mutation_counts_out, bucket_name):
    if mutation_counts_out is None:
        return None
    bucket = mutation_counts_out.setdefault(bucket_name, {})
    for type_name in MUTATION_TYPE_NAMES:
        bucket.setdefault(type_name, 0)
    return bucket


def _record_attempted_mutations(mutation_counts_out, *, n_type1, n_type2,
                                n_type3, n_type4, n_type5=0, n_type6=0,
                                n_type7=0):
    if mutation_counts_out is None:
        return
    attempted = _ensure_mutation_stats_bucket(mutation_counts_out, 'attempted')
    increments = {
        'type1': int(n_type1),
        'type2': int(n_type2),
        'type3': int(n_type3),
        'type4': int(n_type4),
        'type5': int(n_type5),
        'type6': int(n_type6),
        'type7': int(n_type7),
    }
    for type_name, inc in increments.items():
        attempted[type_name] += inc
        # Keep the legacy flat keys as aliases for attempted counts so older
        # callers keep seeing the same numbers.
        mutation_counts_out[type_name] = attempted[type_name]


def _record_mutation_mask(mutation_counts_out, bucket_name, mutation_types,
                          mask):
    if mutation_counts_out is None:
        return
    bucket = _ensure_mutation_stats_bucket(mutation_counts_out, bucket_name)
    if mutation_types is None:
        return

    for type_idx, type_name in enumerate(MUTATION_TYPE_NAMES, start=1):
        type_mask = mutation_types == type_idx
        if not type_mask.any():
            continue
        bucket[type_name] += int(mask[:, type_mask].sum().item())


def _record_accepted_mutations(mutation_counts_out, mutation_types,
                               accepted_mask):
    _record_mutation_mask(
        mutation_counts_out, 'accepted', mutation_types, accepted_mask)


def _record_worse_accepted_mutations(mutation_counts_out, mutation_types,
                                     accepted_mask):
    _record_mutation_mask(
        mutation_counts_out, 'worse_accepted', mutation_types, accepted_mask)


def _ensure_selection_stats_bucket(mutation_counts_out):
    if mutation_counts_out is None:
        return None
    bucket = mutation_counts_out.setdefault('selection', {})
    bucket.setdefault('parent_copies', 0)
    bucket.setdefault('nonbest_parent_copies', 0)
    bucket.setdefault('worse_parent_copies', 0)
    bucket.setdefault('trim_grace_forced_accepts', 0)
    bucket.setdefault('trim_grace_protected', 0)
    return bucket


def _record_trim_grace_stats(mutation_counts_out, *, forced_accepts=0,
                             protected=0):
    """Record trim-grace activity (force-accepted trims, protected bees)."""
    if mutation_counts_out is None:
        return
    bucket = _ensure_selection_stats_bucket(mutation_counts_out)
    bucket['trim_grace_forced_accepts'] += int(forced_accepts)
    bucket['trim_grace_protected'] += int(protected)


def _apply_trim_grace_to_parents(parent_idxs, trim_grace, bee_idxs):
    """Force trim-protected bees to be their own parent during selection.

    A bee whose route was just trimmed has a temporarily worse cost. Without
    protection, cost-based selection (recruiter/follower or soft) discards it
    immediately, so the trim never reaches the next generation and a
    follow-up extend never gets the chance to recover the loss. Bees with
    ``trim_grace > 0`` keep their own just-trimmed network instead.

    Returns ``(parent_idxs, n_protected)``.
    """
    protected = trim_grace > 0
    if not protected.any():
        return parent_idxs, 0
    self_idxs = bee_idxs[None].expand_as(parent_idxs)
    parent_idxs = torch.where(protected, self_idxs, parent_idxs)
    return parent_idxs, int(protected.sum().item())


def _record_selection_stats(mutation_counts_out, parent_idxs,
                            bee_objective_costs):
    """Record which population members survive the selection step."""
    if mutation_counts_out is None:
        return
    bucket = _ensure_selection_stats_bucket(mutation_counts_out)
    parent_costs = bee_objective_costs.gather(1, parent_idxs)
    best_costs = bee_objective_costs.min(dim=1, keepdim=True).values

    bucket['parent_copies'] += int(parent_idxs.numel())
    bucket['nonbest_parent_copies'] += int(
        (parent_costs > best_costs).sum().item())
    bucket['worse_parent_copies'] += int(
        (parent_costs > bee_objective_costs).sum().item())


def _sample_soft_selection_parents(bee_objective_costs, temperature,
                                   uniform_mix=0.05, elite_count=1):
    """Sample parent bee indices with a soft preference for lower cost."""
    _, n_bees = bee_objective_costs.shape
    centered = bee_objective_costs - \
        bee_objective_costs.min(dim=1, keepdim=True).values
    probs = torch.softmax(-centered / temperature, dim=1)
    if uniform_mix > 0:
        probs = (1 - uniform_mix) * probs + uniform_mix / n_bees
    parents = probs.multinomial(n_bees, replacement=True)

    elite_count = min(max(int(elite_count), 0), n_bees)
    if elite_count > 0:
        elite_idxs = bee_objective_costs.topk(
            elite_count, dim=1, largest=False).indices
        parents[:, :elite_count] = elite_idxs
    return parents


def _get_route_selection_weights(bee_networks, demand,
                                 use_demand_weighted_route_selection=False):
    """Return per-route sampling weights for choosing mutation targets.

    When demand weighting is enabled, routes that directly satisfy less demand
    get larger weights, so they are sampled more often but not deterministically.
    """
    batch_size, n_bees, n_routes, _ = bee_networks.shape
    dev = bee_networks.device
    weights = torch.ones((batch_size, n_bees, n_routes), dtype=torch.float32,
                         device=dev)
    if not use_demand_weighted_route_selection:
        return weights

    expanded_demand = demand[:, None].expand(-1, n_bees, -1, -1)
    flat_expanded_demand = expanded_demand.flatten(0, 1)
    flat_bee_networks = bee_networks.flatten(0, 1)
    direct_demand = tu.aggr_edges_over_sequences(
        flat_bee_networks,
        flat_expanded_demand[..., None],
    ).squeeze(-1)
    direct_demand = direct_demand.reshape(batch_size, n_bees, n_routes)

    # Turn "how much demand this route directly serves" into a "badness"
    # score so worse-covered routes are sampled more often.
    max_direct_demand = direct_demand.max(dim=-1, keepdim=True).values
    weights = (max_direct_demand - direct_demand).clamp_min(0)
    no_positive_weights = weights.sum(dim=-1) == 0
    weights[no_positive_weights] = 1.0
    return weights


def _choose_route_indices(bee_networks, demand, n_routes,
                          use_demand_weighted_route_selection=False):
    """Sample one route index per bee for mutation."""
    if not use_demand_weighted_route_selection:
        return torch.randint(
            high=int(n_routes.item()),
            size=bee_networks.shape[:2],
            device=bee_networks.device,
        )

    weights = _get_route_selection_weights(
        bee_networks,
        demand,
        use_demand_weighted_route_selection=True,
    )
    flat_weights = weights.flatten(0, 1)
    chosen_flat = flat_weights.multinomial(1).squeeze(-1)
    return chosen_flat.reshape(bee_networks.shape[:2])


def bee_colony(state, cost_obj, init_network, n_bees=10, passes_per_it=5,
               mod_steps_per_pass=2, shorten_prob=0.2, n_iterations=400,
               n_type1_bees=None, n_type2_bees=None, n_type4_bees=0,
               n_type5_bees=0, n_type6_bees=0, n_type7_bees=0,
               silent=False,
               force_linking_unlinked=False,
               bee_model=None, edit_model=None,
               sum_writer=None, mutation_counts_out=None,
               adjustment_degree_weight=0.0,
               adjustment_degree_gap=0.1,
               adjustment_degree_mode='current',
               adjustment_degree_objective='raw',
               adjustment_degree_target=0.2,
               ignore_type4_max_route_len=False,
               ignore_type5_max_route_len=False,
               ignore_type6_max_route_len=False,
               ignore_type7_max_route_len=False,
               type4_allow_halt=True,
               type5_allow_halt=True,
               type6_allow_halt=True,
               type7_allow_halt=True,
               use_demand_weighted_route_selection=False,
               worse_accept_temperature=0.0,
               worse_accept_decay=0.995,
               worse_accept_min_temperature=0.001,
               worse_selection_temperature=0.0,
               worse_selection_decay=0.995,
               worse_selection_min_temperature=0.001,
               worse_selection_uniform_mix=0.05,
               worse_selection_elite_count=1,
               trim_grace_period=0,
               early_stop_patience=None,
               early_stop_min_delta=0.0):
    """Implementation of the method of  Nikolic and Teodorovic (2013).
    
    state -- A RouteGenBatchState object representing the initial state.
    cost_obj -- A function that determines the cost (badness) of a network.  In
        the paper, this is wieghted total travel time.
    n_routes -- The number of routes in each candidate network, called NBL in
        the paper.
    n_bees -- The number of worker bees, called B in the paper.
    passes_per_it -- The number of forward and backward passes to perform each
        iteration, called NP in the paper.
    mod_steps_per_pass -- The number of modifications each bee considers in the
        forward pass, called NC in the paper.
    shorten_prob -- The probability that type-2 bees will shorten a route,
        called P in the paper.  In their experiments, they use 0.2.
    n_iters -- The number of iterations to perform, called IT in the paper.
    n_type1_bees -- There are 2 types of bees used in the algorithm, which
        modify the solution in different ways.  This parameter determines the
        balance between them.  The paper isn't clear how many of each they use,
        so by default we make it half-and-half.
    n_type4_bees -- neural construction bees that apply one extension/halt
        step to the selected route.
    n_type5_bees -- neural edit bees that apply one extend/trim/halt step.
    n_type6_bees -- neural trim-only bees that apply one trim/halt step.
    n_type7_bees -- neural compound bees that apply one trim-only step and
        then one construction/extension step before evaluation.
    type4_allow_halt/type5_allow_halt/type6_allow_halt/type7_allow_halt --
        whether the corresponding one-step mutation bees may return a no-op
        halt when another action is valid.
    silent -- if true, no tqdm output or printing
    bee_model -- if a torch model is provided, use it as the only bee type.
    adjustment_degree_weight -- penalty weight for changing routes too much
        relative to the original initialized network.
    adjustment_degree_gap -- gap parameter used in sequence alignment when
        computing route similarity.
    adjustment_degree_mode -- one of "current" or "paper".
    adjustment_degree_objective -- one of "raw" or "target".  "target"
        optimizes the distance between the network's adjustment degree and
        adjustment_degree_target instead of the raw adjustment itself.
    adjustment_degree_target -- target adjustment value when using the
        "target" objective.
    use_demand_weighted_route_selection -- if True, choose routes for mutation
        by sampling routes that directly satisfy less demand more often.
    worse_accept_temperature -- if > 0, also accept strictly worse mutations
        with probability exp(-delta / temperature).  Set to 0 to keep the
        original greedy acceptance.
    worse_accept_decay -- multiplicative temperature decay per BCO iteration.
    worse_accept_min_temperature -- lower temperature bound while worse
        acceptance is enabled.
    worse_selection_temperature -- if > 0, use soft population selection
        instead of the original recruiter/follower copying. Lower objective
        costs remain more likely, but worse/non-best bees can survive.
    worse_selection_decay -- multiplicative temperature decay per BCO
        iteration for soft population selection.
    worse_selection_min_temperature -- lower selection temperature bound while
        soft population selection is enabled.
    worse_selection_uniform_mix -- probability mass mixed into a uniform
        parent distribution to preserve exploration.
    worse_selection_elite_count -- number of best current bees copied into
        the first population slots before the remaining slots are sampled.
    trim_grace_period -- if > 0, enable trim-grace selection. A route-edit
        that shrinks its route (a trim) is a setup move: on its own it
        raises cost, so plain cost-based acceptance/selection discards it
        before a follow-up extend can pay off. With trim grace, a trimming
        mutation is force-accepted (one per grace window) and the bee is
        protected from cost-based selection for ``trim_grace_period``
        population-selection rounds, letting the trim survive into the next
        generation. Set to 0 to keep the original cost-only selection.
    """
    if edit_model is None and getattr(bee_model, 'supports_trim_actions', False):
        edit_model = bee_model

    if n_type1_bees is None:
        n_type1_bees = n_bees // 2
    if n_type2_bees is None:
        # assume no type-3/4/5/6/7 bees if not specified
        n_type2_bees = (n_bees - n_type1_bees - n_type4_bees -
                        n_type5_bees - n_type6_bees - n_type7_bees)
    n_type3_bees = (n_bees - n_type1_bees - n_type2_bees -
                    n_type4_bees - n_type5_bees - n_type6_bees -
                    n_type7_bees)
    if n_type3_bees < 0:
        raise ValueError(
            "Sum of n_type1/2/4/5/6/7 bees exceeds n_bees: "
            f"{n_type1_bees}+{n_type2_bees}+{n_type4_bees}+"
            f"{n_type5_bees}+{n_type6_bees}+{n_type7_bees} "
            f"> {n_bees}"
        )
    trim_bees = n_type5_bees + n_type6_bees + n_type7_bees
    if trim_bees > 0 and edit_model is None:
        raise ValueError(
            "n_type5_bees/n_type6_bees/n_type7_bees > 0 requires an edit_model that "
            "supports trim actions (set edit_model when calling bee_colony)."
        )
    if trim_bees > 0 and not getattr(edit_model, 'supports_trim_actions',
                                     False):
        raise ValueError(
            "edit_model must have supports_trim_actions=True to drive "
            "type-5/type-6/type-7 mutations."
        )
    if worse_accept_temperature < 0:
        raise ValueError("worse_accept_temperature must be >= 0")
    if worse_accept_decay <= 0:
        raise ValueError("worse_accept_decay must be > 0")
    if worse_accept_min_temperature < 0:
        raise ValueError("worse_accept_min_temperature must be >= 0")
    if worse_selection_temperature < 0:
        raise ValueError("worse_selection_temperature must be >= 0")
    if worse_selection_decay <= 0:
        raise ValueError("worse_selection_decay must be > 0")
    if worse_selection_min_temperature < 0:
        raise ValueError("worse_selection_min_temperature must be >= 0")
    if not 0 <= worse_selection_uniform_mix <= 1:
        raise ValueError("worse_selection_uniform_mix must be in [0, 1]")
    if worse_selection_elite_count < 0:
        raise ValueError("worse_selection_elite_count must be >= 0")
    if trim_grace_period < 0:
        raise ValueError("trim_grace_period must be >= 0")

    if n_type3_bees > 0:
        # instantiate a random path-combining model
        rpc_model = lrnu.get_random_path_combiner()
    else:
        rpc_model = None

    batch_size = state.batch_size

    dev = state.device
    batch_idxs = torch.arange(batch_size, device=dev)
    max_n_nodes = state.max_n_nodes

    # get all shortest paths
    shortest_paths, _ = tu.reconstruct_all_paths(state.nexts)

    demand = torch.nn.functional.pad(state.demand, (0, 1, 0, 1))
    n_routes = state.n_routes_to_plan
    best_networks = torch.full((batch_size, n_routes, max_n_nodes), -1,
                                device=dev)
    n_init_routes = init_network.shape[1]
    best_networks[:, :n_init_routes, :init_network.shape[-1]] = init_network
    reference_networks = best_networks.clone()
    use_adjustment_penalty = adjustment_degree_weight > 0
    # Unified adjustment-degree penalty: the cost module owns the adj term
    # (single source of truth). Configure it with this run's seed + params so
    # cost_obj(...) already includes the penalty; bee_colony no longer adds it
    # separately. Reset to disabled at the end so the cost_obj does not leak.
    if use_adjustment_penalty:
        cost_obj.adjustment_degree_weight = adjustment_degree_weight
        cost_obj.adjustment_degree_target = adjustment_degree_target
        cost_obj.adjustment_degree_objective = adjustment_degree_objective
        cost_obj.adjustment_degree_gap = adjustment_degree_gap
        cost_obj.adjustment_degree_mode = adjustment_degree_mode
        cost_obj.adjustment_seed = reference_networks
    route_count = max(float(n_routes.item()), 1.0)

    # expand state to networks

    # compute can-be-directly-satisfied demand matrix
    direct_sat_dmd = get_direct_sat_dmd(demand, shortest_paths, 
                                        cost_obj.symmetric_routes)

    # set up required matrices
    street_node_neighbours = (state.street_adj.isfinite() &
                              (state.street_adj > 0))
    bee_idxs = torch.arange(n_bees, device=dev)                      

    log.debug("starting BCO")

    # set up the cost function to work with batches of bees
    # "multiply" the batch for the bees
    exp_states = sum([[substate] * n_bees 
                      for substate in state.batch_to_list()], [])
    bee_states = RouteGenBatchState.batch_from_list(exp_states)
    if bee_model is not None:
        bee_states = bee_model.setup_planning(bee_states)

    metric_names = cost_obj.get_metric_names()
    
    def batched_cost_fn(bee_networks):
        bee_states.replace_routes(bee_networks.flatten(0,1))

        result = cost_obj(bee_states)

        costs = result.cost.reshape(batch_size, n_bees)
        other_metrics = result.get_metrics_tensor()
        other_metrics = other_metrics.reshape(batch_size, n_bees, -1)
        return costs, other_metrics

    # initialize bees' networks
    # batch size x n_bees x n_routes x max_n_nodes
    bee_networks = best_networks[:, None].repeat(1, n_bees, 1, 1)
    # evaluate and record the reward of the initial network
    bee_raw_costs, bee_metrics = batched_cost_fn(bee_networks)
    bee_adjustment_degrees = torch.zeros_like(bee_raw_costs)
    bee_adjustment_penalties = get_adjustment_penalties(
        bee_adjustment_degrees,
        objective=adjustment_degree_objective,
        target=adjustment_degree_target,
    )
    # adj already included in bee_raw_costs via cost_obj (unified penalty)
    bee_objective_costs = bee_raw_costs
    best_objective_costs, best_idxs = bee_objective_costs.min(1)
    best_raw_costs = bee_raw_costs[batch_idxs, best_idxs]
    best_metrics = bee_metrics[batch_idxs, best_idxs]
    best_adjustment_degrees = bee_adjustment_degrees[batch_idxs, best_idxs]
    best_adjustment_penalties = bee_adjustment_penalties[batch_idxs, best_idxs]

    if sum_writer is not None:
        # log the initial values of various metrics
        for name, vals in zip(metric_names, best_metrics.unbind(-1)):
            sum_writer.add_scalar(f'best {name}', vals.mean(), 0)
        if use_adjustment_penalty:
            sum_writer.add_scalar('best adjustment degree',
                                  best_adjustment_degrees.mean(), 0)
            sum_writer.add_scalar('best adjustment penalty',
                                  best_adjustment_penalties.mean(), 0)

    cost_history = torch.zeros((batch_size, n_iterations + 1), device=dev)
    cost_history[:, 0] = best_raw_costs
    use_worse_accept = worse_accept_temperature > 0
    use_worse_selection = worse_selection_temperature > 0
    use_trim_grace = trim_grace_period > 0
    # Early-stopping bookkeeping. BCO's best_raw_costs is monotonic by
    # construction (only the lowest cost ever seen replaces it), so a strict
    # drop > `min_delta` resets the patience counter.
    use_early_stop = (early_stop_patience is not None
                      and early_stop_patience > 0)
    iters_since_best_improved = 0
    best_cost_tracker = float(best_raw_costs.min().item())
    # Per-bee countdown: > 0 means the bee was recently trimmed and is
    # protected from cost-based selection until the counter expires.
    trim_grace = torch.zeros((batch_size, n_bees), dtype=torch.long,
                             device=dev)

    for iteration in tqdm(range(n_iterations), disable=silent):
        if use_worse_accept:
            current_worse_accept_temperature = max(
                worse_accept_min_temperature,
                worse_accept_temperature * (worse_accept_decay ** iteration),
            )
        else:
            current_worse_accept_temperature = 0.0
        if use_worse_selection:
            current_worse_selection_temperature = max(
                worse_selection_min_temperature,
                worse_selection_temperature *
                (worse_selection_decay ** iteration),
            )
        else:
            current_worse_selection_temperature = 0.0
        for pi in range(passes_per_it):
            for mi in range(mod_steps_per_pass):

                # # flatten batch and bee dimensions
                # expanded_demand = demand[:, None].expand(-1, n_bees, -1, -1)
                # flat_exp_demand = expanded_demand.flatten(0, 1)
                # flat_bee_scens = bee_networks.flatten(0, 1)
                # route_dsds = aggr_edges_over_sequences(flat_bee_scens, 
                #                                     flat_exp_demand[..., None])
                # route_dsds.squeeze_(-1)
                # # choose routes to modify
                # route_scores = 1 / route_dsds
                # route_scores[route_scores.isinf()] = 10**10
                # flat_chosen_route_idxs = route_scores.multinomial(1).squeeze(1)
                # chosen_route_idxs = flat_chosen_route_idxs.reshape(batch_size, 
                #                                                    n_bees)                

                chosen_route_idxs = _choose_route_indices(
                    bee_networks,
                    demand,
                    n_routes,
                    use_demand_weighted_route_selection=
                    use_demand_weighted_route_selection,
                )
                gather_idx = chosen_route_idxs[..., None, None]
                gather_idx = gather_idx.expand(-1, -1, -1, max_n_nodes)
                old_modified_routes = bee_networks.gather(2, gather_idx).squeeze(2)
                _record_attempted_mutations(
                    mutation_counts_out,
                    n_type1=n_type1_bees,
                    n_type2=n_type2_bees,
                    n_type3=n_type3_bees,
                    n_type4=n_type4_bees,
                    n_type5=n_type5_bees,
                    n_type6=n_type6_bees,
                    n_type7=n_type7_bees,
                )
                new_bee_networks, mutation_types = \
                    get_mutants(bee_networks, chosen_route_idxs, n_type1_bees,
                                n_type2_bees, direct_sat_dmd, shorten_prob,
                                street_node_neighbours, shortest_paths,
                                force_linking_unlinked, bee_model, rpc_model,
                                bee_states, n_type4=n_type4_bees,
                                n_type5=n_type5_bees,
                                n_type6=n_type6_bees,
                                n_type7=n_type7_bees,
                                edit_model=edit_model,
                                ignore_type4_max_route_len=
                                ignore_type4_max_route_len,
                                type4_allow_halt=type4_allow_halt,
                                type5_allow_halt=type5_allow_halt,
                                type6_allow_halt=type6_allow_halt,
                                type7_allow_halt=type7_allow_halt,
                                adj_condition_target=adjustment_degree_target,
                                adj_condition_weight=adjustment_degree_weight,
                                ignore_type5_max_route_len=
                                ignore_type5_max_route_len,
                                ignore_type6_max_route_len=
                                ignore_type6_max_route_len,
                                ignore_type7_max_route_len=
                                ignore_type7_max_route_len,
                                return_mutation_metadata=True)

                new_bee_raw_costs, new_bee_metrics = \
                    batched_cost_fn(new_bee_networks)

                if use_adjustment_penalty:
                    reference_routes = reference_networks[:, None].expand(
                        -1, n_bees, -1, -1).gather(2, gather_idx).squeeze(2)
                    old_route_adjustments = get_adjustment_degrees(
                        old_modified_routes, reference_routes,
                        cost_obj.symmetric_routes,
                        gap=adjustment_degree_gap,
                        mode=adjustment_degree_mode,
                    )
                    new_modified_routes = new_bee_networks.gather(
                        2, gather_idx).squeeze(2)
                    new_route_adjustments = get_adjustment_degrees(
                        new_modified_routes, reference_routes,
                        cost_obj.symmetric_routes,
                        gap=adjustment_degree_gap,
                        mode=adjustment_degree_mode,
                    )
                    new_bee_adjustment_degrees = bee_adjustment_degrees + \
                        (new_route_adjustments - old_route_adjustments) / route_count
                else:
                    new_bee_adjustment_degrees = bee_adjustment_degrees

                new_bee_adjustment_penalties = get_adjustment_penalties(
                    new_bee_adjustment_degrees,
                    objective=adjustment_degree_objective,
                    target=adjustment_degree_target,
                )
                # adj already included in new_bee_raw_costs via cost_obj
                new_bee_objective_costs = new_bee_raw_costs

                objective_delta = new_bee_objective_costs - bee_objective_costs
                better_idxs = objective_delta < 0
                if use_worse_accept and current_worse_accept_temperature > 0:
                    worse_candidates = objective_delta > 0
                    worse_probs = torch.zeros_like(objective_delta)
                    worse_probs[worse_candidates] = torch.exp(
                        -objective_delta[worse_candidates] /
                        current_worse_accept_temperature)
                    worse_accepted = worse_candidates & (
                        torch.rand_like(worse_probs) < worse_probs)
                    accepted_idxs = better_idxs | worse_accepted
                else:
                    worse_accepted = torch.zeros_like(better_idxs)
                    accepted_idxs = better_idxs

                # Trim-grace: a mutation that shrinks its chosen route is a
                # setup move. Force-accept it (one per grace window) so the
                # trimmed network reaches the next generation, where a
                # follow-up extend can recover and improve on it.
                forced_trim_accepts = torch.zeros_like(better_idxs)
                if use_trim_grace:
                    new_modified_routes = new_bee_networks.gather(
                        2, gather_idx).squeeze(2)
                    old_route_lens = (old_modified_routes > -1).sum(-1)
                    new_route_lens = (new_modified_routes > -1).sum(-1)
                    is_trim_mutation = new_route_lens < old_route_lens
                    forced_trim_accepts = is_trim_mutation & \
                        (trim_grace == 0) & ~accepted_idxs
                    accepted_idxs = accepted_idxs | forced_trim_accepts
                    _record_trim_grace_stats(
                        mutation_counts_out,
                        forced_accepts=int(forced_trim_accepts.sum().item()))

                _record_accepted_mutations(
                    mutation_counts_out,
                    mutation_types,
                    accepted_idxs,
                )
                _record_worse_accepted_mutations(
                    mutation_counts_out,
                    mutation_types,
                    worse_accepted,
                )
                bee_networks[accepted_idxs] = new_bee_networks[accepted_idxs]
                bee_raw_costs[accepted_idxs] = new_bee_raw_costs[accepted_idxs]
                bee_objective_costs[accepted_idxs] = \
                    new_bee_objective_costs[accepted_idxs]
                bee_adjustment_degrees[accepted_idxs] = \
                    new_bee_adjustment_degrees[accepted_idxs]
                bee_adjustment_penalties[accepted_idxs] = \
                    new_bee_adjustment_penalties[accepted_idxs]
                bee_metrics[accepted_idxs] = new_bee_metrics[accepted_idxs]

                if use_trim_grace:
                    # Arm the grace counter for any bee whose accepted
                    # mutation trimmed its route.
                    graced_now = accepted_idxs & is_trim_mutation
                    trim_grace[graced_now] = trim_grace_period

            # do "backward pass"

            # update the best solution found so far
            current_best_objective_cost, current_best_idx = \
                bee_objective_costs.min(1)
            is_improvement = current_best_objective_cost < best_objective_costs
            improvement_idx = current_best_idx[is_improvement]
            best_objective_costs[is_improvement] = \
                current_best_objective_cost[is_improvement]
            best_networks[is_improvement] = \
                bee_networks[is_improvement, improvement_idx]
            best_raw_costs[is_improvement] = \
                bee_raw_costs[is_improvement, improvement_idx]
            best_adjustment_degrees[is_improvement] = \
                bee_adjustment_degrees[is_improvement, improvement_idx]
            best_adjustment_penalties[is_improvement] = \
                bee_adjustment_penalties[is_improvement, improvement_idx]
            cost_history[:, iteration + 1] = best_raw_costs
            new_best = bee_metrics[is_improvement, improvement_idx]
            best_metrics[is_improvement] = new_best     

            if use_worse_selection and current_worse_selection_temperature > 0:
                parent_idxs = _sample_soft_selection_parents(
                    bee_objective_costs,
                    current_worse_selection_temperature,
                    uniform_mix=worse_selection_uniform_mix,
                    elite_count=worse_selection_elite_count,
                )
                if use_trim_grace:
                    parent_idxs, n_protected = _apply_trim_grace_to_parents(
                        parent_idxs, trim_grace, bee_idxs)
                    _record_trim_grace_stats(
                        mutation_counts_out, protected=n_protected)
                _record_selection_stats(
                    mutation_counts_out, parent_idxs, bee_objective_costs)

                bee_networks = bee_networks[batch_idxs[:, None], parent_idxs]
                bee_raw_costs = bee_raw_costs[batch_idxs[:, None], parent_idxs]
                bee_objective_costs = \
                    bee_objective_costs[batch_idxs[:, None], parent_idxs]
                bee_adjustment_degrees = \
                    bee_adjustment_degrees[batch_idxs[:, None], parent_idxs]
                bee_adjustment_penalties = \
                    bee_adjustment_penalties[batch_idxs[:, None], parent_idxs]
                bee_metrics = bee_metrics[batch_idxs[:, None], parent_idxs]
                if use_trim_grace:
                    trim_grace = trim_grace[batch_idxs[:, None], parent_idxs]
                    trim_grace = (trim_grace - 1).clamp_min(0)
                continue

            # decide whether each bee is a recruiter or follower
            max_bee_costs, _ = bee_objective_costs.max(dim=1)
            min_bee_costs, _ = bee_objective_costs.min(dim=1)
            spread = max_bee_costs - min_bee_costs
            # avoid division by 0
            spread[spread == 0] = 1

            qualities = (max_bee_costs[:, None] - bee_objective_costs) / \
                spread[:, None]
            min_quality, _ = qualities.min(1)
            follow_probs = (-qualities + min_quality[:, None]).exp()
            are_recruiters = follow_probs < torch.rand(n_bees, device=dev)

            if not are_recruiters.any():
                # no recruiters, so make everyone keep their own network.
                if use_trim_grace:
                    trim_grace = (trim_grace - 1).clamp_min(0)
                continue

            # decide which recruiter each follower will follow
            denom = (qualities * are_recruiters).sum(1)
            # avoid division by 0
            denom[denom == 0] = 1
            recruit_probs = qualities / denom[:, None]
            recruit_probs[~are_recruiters] = 0
            # set probs where there is no probability to 1, so multinomial
             # doesn't complain.
            no_valid_recruiters = recruit_probs.sum(-1) == 0
            recruit_probs[no_valid_recruiters] = 1
            recruiters = recruit_probs.multinomial(n_bees, replacement=True)
            recruiters[no_valid_recruiters] = bee_idxs
            recruiters[are_recruiters] = \
                bee_idxs[None].expand(batch_size, -1)[are_recruiters]
            if use_trim_grace:
                recruiters, n_protected = _apply_trim_grace_to_parents(
                    recruiters, trim_grace, bee_idxs)
                _record_trim_grace_stats(
                    mutation_counts_out, protected=n_protected)
            _record_selection_stats(
                mutation_counts_out, recruiters, bee_objective_costs)

            # update the networks and costs of the followers
            bee_networks = bee_networks[batch_idxs[:, None], recruiters]
            bee_raw_costs = bee_raw_costs[batch_idxs[:, None], recruiters]
            bee_objective_costs = \
                bee_objective_costs[batch_idxs[:, None], recruiters]
            bee_adjustment_degrees = \
                bee_adjustment_degrees[batch_idxs[:, None], recruiters]
            bee_adjustment_penalties = \
                bee_adjustment_penalties[batch_idxs[:, None], recruiters]
            bee_metrics = bee_metrics[batch_idxs[:, None], recruiters]
            if use_trim_grace:
                trim_grace = trim_grace[batch_idxs[:, None], recruiters]
                trim_grace = (trim_grace - 1).clamp_min(0)

        if sum_writer is not None:
            # log the various metrics
            for name, vals in zip(metric_names, best_metrics.unbind(-1)):
                sum_writer.add_scalar(f'best {name}', vals.mean(), iteration+1)
            if use_adjustment_penalty:
                sum_writer.add_scalar('best adjustment degree',
                                      best_adjustment_degrees.mean(),
                                      iteration + 1)
                sum_writer.add_scalar('best adjustment penalty',
                                      best_adjustment_penalties.mean(),
                                      iteration + 1)
            if use_worse_accept:
                sum_writer.add_scalar('worse accept temperature',
                                      current_worse_accept_temperature,
                                      iteration + 1)
            if use_worse_selection:
                sum_writer.add_scalar('worse selection temperature',
                                      current_worse_selection_temperature,
                                      iteration + 1)

        # Early-stopping check (post-iteration: cost_history[iteration+1] is
        # the freshly written best_raw_costs).
        new_best = float(best_raw_costs.min().item())
        if best_cost_tracker - new_best > early_stop_min_delta:
            best_cost_tracker = new_best
            iters_since_best_improved = 0
        else:
            iters_since_best_improved += 1
        if use_early_stop and iters_since_best_improved >= early_stop_patience:
            cost_history = cost_history[:, :iteration + 2].clone()
            if not silent:
                log.info(
                    f"[BCO] early stop at iter {iteration + 1}/{n_iterations}: "
                    f"no >{early_stop_min_delta:g} best improvement in "
                    f"{iters_since_best_improved} iters")
            break

    # return the best solution
    state.replace_routes(best_networks)
    # reset the unified adjustment-degree penalty so the shared cost_obj does
    # not carry this run's seed into any later evaluation.
    if use_adjustment_penalty:
        cost_obj.adjustment_seed = None
        cost_obj.adjustment_degree_weight = 0.0
    return state, cost_history


def get_mutants(bee_networks, chosen_route_idxs, n_type1, n_type2,
                direct_sat_dmd, shorten_prob, street_node_neighbours,
                shortest_paths, force_linking_unlinked, bee_model=None,
                rpc_model=None, env_state=None, n_type4=0,
                n_type5=0, n_type6=0, n_type7=0, edit_model=None,
                type4_allow_halt=True,
                type5_allow_halt=True,
                type6_allow_halt=True,
                type7_allow_halt=True,
                adj_condition_target=None, adj_condition_weight=None,
                ignore_type4_max_route_len=False,
                ignore_type5_max_route_len=False,
                ignore_type6_max_route_len=False,
                ignore_type7_max_route_len=False,
                return_mutation_metadata=False):
    bee_networks = bee_networks.clone()

    # flatten batch and bee dimensions
    gather_idx = chosen_route_idxs[..., None, None]
    max_n_nodes = bee_networks.shape[3]
    gather_idx = gather_idx.expand(-1, -1, -1, max_n_nodes)
    modified_routes = bee_networks.gather(2, gather_idx).squeeze(2)
    empty_routes = (modified_routes > -1).sum(-1) == 0

    n_bees = bee_networks.shape[1]
    n_type3 = n_bees - n_type1 - n_type2 - n_type4 - n_type5 - n_type6 - \
        n_type7
    scen_idxs = torch.randperm(n_bees, device=bee_networks.device)
    type1_idxs = scen_idxs[:n_type1]
    type2_idxs = scen_idxs[n_type1:n_type1 + n_type2]
    type3_idxs = scen_idxs[n_type1 + n_type2:n_type1 + n_type2 + n_type3]
    type4_idxs = scen_idxs[
        n_type1 + n_type2 + n_type3:
        n_type1 + n_type2 + n_type3 + n_type4]
    type5_idxs = scen_idxs[
        n_type1 + n_type2 + n_type3 + n_type4:
        n_type1 + n_type2 + n_type3 + n_type4 + n_type5]
    type7_idxs = scen_idxs[
        n_type1 + n_type2 + n_type3 + n_type4 + n_type5 + n_type6:]
    type6_idxs = scen_idxs[
        n_type1 + n_type2 + n_type3 + n_type4 + n_type5:
        n_type1 + n_type2 + n_type3 + n_type4 + n_type5 + n_type6]
    mutation_types = torch.zeros(n_bees, device=bee_networks.device,
                                 dtype=torch.long)
    mutation_types[type1_idxs] = 1
    mutation_types[type2_idxs] = 2
    mutation_types[type3_idxs] = 3
    mutation_types[type4_idxs] = 4
    mutation_types[type5_idxs] = 5
    mutation_types[type6_idxs] = 6
    mutation_types[type7_idxs] = 7

    remaining_state = None
    if bee_model is None or (empty_routes.any() and force_linking_unlinked):
        unsel_routes = tu.get_unselected_routes(bee_networks, chosen_route_idxs)
        remaining_state = env_state.clone()
        remaining_state.replace_routes(unsel_routes.flatten(0,1))

    if empty_routes.any():
        new_routes = get_new_route_variants(
            modified_routes,
            direct_sat_dmd,
            shortest_paths,
            force_linking_unlinked=force_linking_unlinked,
            remaining_state=remaining_state,
        )
        modified_routes[empty_routes] = new_routes[empty_routes]

    # modify type 1 routes
    if n_type1 == 0:
        new_type1_routes = modified_routes[:, type1_idxs]
    elif bee_model is not None:
        # run it on all bees...
        new_type1_networks = get_neural_variants(bee_model, env_state, 
                                                 bee_networks,
                                                 chosen_route_idxs)
        # ...and keep only the type 1 bee routes
        new_type1_routes = new_type1_networks[:, type1_idxs, -1]
    else:
        new_type1_routes = get_bee_1_variants(remaining_state, modified_routes,
                                              direct_sat_dmd, shortest_paths,
                                              force_linking_unlinked)
        # take on the type-1 routes
        new_type1_routes = new_type1_routes[:, type1_idxs]

    # modify type 2 routes
    new_type2_routes = get_bee_2_variants(modified_routes[:, type2_idxs], 
                                          shorten_prob, street_node_neighbours)
    assert ((new_type2_routes > -1).sum(dim=-1) > 0).all()

    # Reassemble the selected-route mutations in the original bee order.
    # Each mutation type is computed on a permuted bee subset, so concatenating
    # by type would misalign routes and copy one bee's result into another.
    new_routes = modified_routes.clone()
    new_routes[:, type1_idxs] = new_type1_routes
    new_routes[:, type2_idxs] = new_type2_routes
    if rpc_model is not None:
        # modify type 3 routes
        new_type3_networks = get_neural_variants(rpc_model, env_state,
                                                  bee_networks,
                                                  chosen_route_idxs)
        new_type3_routes = new_type3_networks[:, type3_idxs, -1]
        new_routes[:, type3_idxs] = new_type3_routes

    if bee_model is not None and n_type4 > 0:
        # modify type 4 routes: single-step GNN extension of the chosen route
        new_type4_all = get_neural_extend_variants(
            bee_model,
            env_state,
            bee_networks,
            chosen_route_idxs,
            ignore_max_route_len=ignore_type4_max_route_len,
            allow_halt=type4_allow_halt,
        )
        type4_gather = chosen_route_idxs[:, type4_idxs, None, None].expand(
            -1, -1, -1, max_n_nodes)
        new_type4_routes = new_type4_all[:, type4_idxs].gather(
            2, type4_gather).squeeze(2)
        new_routes[:, type4_idxs] = new_type4_routes

    if edit_model is not None and n_type5 > 0:
        # modify type 5 routes: single-step edit (extend / trim_start /
        # trim_end / halt) using a trim-capable model.
        new_type5_all = get_neural_edit_variants(
            edit_model,
            env_state,
            bee_networks,
            chosen_route_idxs,
            ignore_max_route_len=ignore_type5_max_route_len,
            allow_halt=type5_allow_halt,
            adj_condition_target=adj_condition_target,
            adj_condition_weight=adj_condition_weight,
        )
        type5_gather = chosen_route_idxs[:, type5_idxs, None, None].expand(
            -1, -1, -1, max_n_nodes)
        new_type5_routes = new_type5_all[:, type5_idxs].gather(
            2, type5_gather).squeeze(2)
        new_routes[:, type5_idxs] = new_type5_routes

    if edit_model is not None and n_type6 > 0:
        # modify type 6 routes: trim-only edit (trim_start / trim_end / halt)
        # using a trim-capable model. Extending is masked out.
        new_type6_all = get_neural_trim_variants(
            edit_model,
            env_state,
            bee_networks,
            chosen_route_idxs,
            ignore_max_route_len=ignore_type6_max_route_len,
            allow_halt=type6_allow_halt,
        )
        type6_gather = chosen_route_idxs[:, type6_idxs, None, None].expand(
            -1, -1, -1, max_n_nodes)
        new_type6_routes = new_type6_all[:, type6_idxs].gather(
            2, type6_gather).squeeze(2)
        new_routes[:, type6_idxs] = new_type6_routes

    if edit_model is not None and n_type7 > 0:
        # modify type 7 routes: trim-only edit, then one construction-style
        # extension/halt step, evaluated as a single compound mutation.
        extend_model = bee_model if bee_model is not None else edit_model
        new_type7_all = get_neural_trim_then_extend_variants(
            edit_model,
            extend_model,
            env_state,
            bee_networks,
            chosen_route_idxs,
            ignore_max_route_len=ignore_type7_max_route_len,
            allow_halt=type7_allow_halt,
        )
        type7_gather = chosen_route_idxs[:, type7_idxs, None, None].expand(
            -1, -1, -1, max_n_nodes)
        new_type7_routes = new_type7_all[:, type7_idxs].gather(
            2, type7_gather).squeeze(2)
        new_routes[:, type7_idxs] = new_type7_routes

    bee_networks.scatter_(2, gather_idx, new_routes[..., None, :])
    if return_mutation_metadata:
        return bee_networks, mutation_types
    return bee_networks


def get_new_route_variants(batch_bee_routes, direct_sat_dmd_mat, shortest_paths,
                           force_linking_unlinked=False, remaining_state=None):
    """Construct fresh routes for bees that selected empty route slots."""
    batch_size, n_bees, max_n_nodes = batch_bee_routes.shape
    dev = batch_bee_routes.device
    n_nodes = direct_sat_dmd_mat.shape[-1] - 1

    route_scores = direct_sat_dmd_mat[:, :-1, :-1].clamp_min(0).clone()
    diag = torch.eye(n_nodes, device=dev, dtype=bool)
    route_scores[:, diag] = 0

    per_bee_scores = route_scores[:, None].expand(-1, n_bees, -1, -1).clone()

    if force_linking_unlinked and remaining_state is not None:
        candidate_routes = shortest_paths[:, None].expand(-1, n_bees, -1, -1, -1)
        extends_if_needed = check_extensions_add_connections(
            remaining_state.has_path,
            candidate_routes.flatten(0, 1),
        )
        extends_if_needed = extends_if_needed.reshape(batch_size, n_bees, n_nodes, n_nodes)
        filtered_scores = per_bee_scores * extends_if_needed
        has_valid_extensions = filtered_scores.sum(dim=(-1, -2)) > 0
        per_bee_scores[has_valid_extensions] = filtered_scores[has_valid_extensions]

    flat_scores = per_bee_scores.reshape(batch_size * n_bees, -1)
    off_diag = (~diag).reshape(1, -1).to(flat_scores.dtype)
    no_demand = flat_scores.sum(-1) == 0
    if no_demand.any():
        flat_scores[no_demand] = off_diag.expand(no_demand.sum(), -1)

    flat_choices = flat_scores.multinomial(1).squeeze(-1)
    repeated_batch_idxs = torch.arange(batch_size, device=dev).repeat_interleave(n_bees)
    starts = torch.div(flat_choices, n_nodes, rounding_mode='floor')
    ends = flat_choices % n_nodes
    new_routes = shortest_paths[repeated_batch_idxs, starts, ends]
    new_routes = new_routes.reshape(batch_size, n_bees, -1)

    pad_size = max_n_nodes - new_routes.shape[-1]
    if pad_size > 0:
        new_routes = torch.nn.functional.pad(new_routes, (0, pad_size), value=-1)

    return new_routes


def get_neural_extend_variants(model, env_state, bee_networks, chosen_route_idxs,
                               greedy=False, ignore_max_route_len=False,
                               allow_halt=True):
    """Extend chosen routes with a single GNN step instead of rebuilding them.

    For each bee, the chosen route is set as the current in-progress route and
    the model is asked for exactly one action (a path-segment extension or halt).
    If the model halts the route stays unchanged; if it extends, the segment is
    appended.
    """
    bee_dim = bee_networks.ndim == 4
    if not bee_dim:
        bee_networks = bee_networks.unsqueeze(1)
        chosen_route_idxs = chosen_route_idxs.unsqueeze(1)

    batch_size = bee_networks.shape[0]
    n_bees = bee_networks.shape[1]
    n_routes = bee_networks.shape[2]
    max_n_nodes = bee_networks.shape[3]

    gather_idx = chosen_route_idxs[..., None, None].expand(-1, -1, -1, max_n_nodes)
    chosen_routes = bee_networks.gather(2, gather_idx).squeeze(2)
    flat_chosen = chosen_routes.flatten(0, 1)

    keep_mask = torch.ones(bee_networks.shape[:3], dtype=bool,
                           device=bee_networks.device)
    keep_mask.scatter_(2, chosen_route_idxs[..., None], False)
    flat_kept = bee_networks[keep_mask].reshape(
        batch_size * n_bees, n_routes - 1, max_n_nodes)

    # one transit-data rebuild for the replace + seed-current pair
    with env_state.defer_route_data_update():
        env_state.replace_routes(flat_kept)
        env_state.set_current_routes(flat_chosen)
    env_state = model.setup_planning(env_state)

    pre_step_routes = env_state.current_routes.clone()

    supports_trim_actions = getattr(model, 'supports_trim_actions', False)

    if ignore_max_route_len:
        original_max_route_len = env_state.extra_data.max_route_len.clone()
        env_state.extra_data.max_route_len = env_state.n_nodes.clone()
        try:
            if supports_trim_actions:
                action_kinds, action, _, _ = model.step_route_action(
                    env_state, greedy=greedy,
                    allow_extend=True,
                    allow_trim_start=False,
                    allow_trim_end=False,
                    allow_halt=allow_halt)
            else:
                action, _, _ = model.step(
                    env_state, greedy=greedy, allow_halt=allow_halt)
        finally:
            env_state.extra_data.max_route_len = original_max_route_len
    else:
        if supports_trim_actions:
            action_kinds, action, _, _ = model.step_route_action(
                env_state, greedy=greedy,
                allow_extend=True,
                allow_trim_start=False,
                allow_trim_end=False,
                allow_halt=allow_halt)
        else:
            action, _, _ = model.step(
                env_state, greedy=greedy, allow_halt=allow_halt)

    if supports_trim_actions:
        halted = action_kinds == ROUTE_ACTION_HALT
        env_state.apply_route_actions(action_kinds, action)
    else:
        halted = action[:, 0] == -1
        env_state.shortest_path_action(action)

    post_step_routes = env_state.current_routes.clone()

    new_flat_routes = pre_step_routes.clone()
    new_flat_routes[~halted] = post_step_routes[~halted]

    pad_size = max_n_nodes - new_flat_routes.shape[-1]
    if pad_size > 0:
        new_flat_routes = torch.nn.functional.pad(
            new_flat_routes, (0, pad_size), value=-1)

    new_routes = new_flat_routes.reshape(batch_size, n_bees, max_n_nodes)

    result_networks = bee_networks.clone()
    result_networks.scatter_(2, gather_idx, new_routes.unsqueeze(2))

    if not bee_dim:
        result_networks = result_networks.squeeze(1)

    return result_networks


def get_neural_edit_variants(model, env_state, bee_networks, chosen_route_idxs,
                             greedy=False, ignore_max_route_len=False,
                             allow_extend=True, allow_trim_start=True,
                             allow_trim_end=True, allow_halt=True,
                             adj_condition_target=None,
                             adj_condition_weight=None):
    """Apply one edit step (extend / trim_start / trim_end / halt) per bee.

    Like ``get_neural_extend_variants`` but the model is allowed to choose
    any route action — extend, trim_start, trim_end, or halt — in a single
    step. The bee's chosen route is fed in as the in-progress route, the
    rest of the bee's network is fed in as context, and one
    ``step_route_action`` call decides what to do. If the model halts, the
    chosen route is kept unchanged; otherwise the route after the action is
    written back into the network.
    """
    if not getattr(model, 'supports_trim_actions', False):
        raise ValueError(
            "get_neural_edit_variants requires a model with "
            "supports_trim_actions=True"
        )

    bee_dim = bee_networks.ndim == 4
    if not bee_dim:
        bee_networks = bee_networks.unsqueeze(1)
        chosen_route_idxs = chosen_route_idxs.unsqueeze(1)

    batch_size = bee_networks.shape[0]
    n_bees = bee_networks.shape[1]
    n_routes = bee_networks.shape[2]
    max_n_nodes = bee_networks.shape[3]

    gather_idx = chosen_route_idxs[..., None, None].expand(
        -1, -1, -1, max_n_nodes)
    chosen_routes = bee_networks.gather(2, gather_idx).squeeze(2)
    flat_chosen = chosen_routes.flatten(0, 1)

    keep_mask = torch.ones(bee_networks.shape[:3], dtype=bool,
                           device=bee_networks.device)
    keep_mask.scatter_(2, chosen_route_idxs[..., None], False)
    flat_kept = bee_networks[keep_mask].reshape(
        batch_size * n_bees, n_routes - 1, max_n_nodes)

    # one transit-data rebuild for the replace + seed-current pair
    with env_state.defer_route_data_update():
        env_state.replace_routes(flat_kept)
        env_state.set_current_routes(flat_chosen)
    # Adjustment-conditioned edit models expect 1-2 extra global features
    # (target [, weight]). The state-level gate feeds EVERY model that reads
    # this state, so set it only around the edit-model calls and clear it
    # afterwards (the construction bee model is built without these feats).
    n_cond = int(getattr(model, 'n_adjustment_cond_feats', 0) or 0)
    if n_cond > 0:
        if adj_condition_target is None:
            raise ValueError(
                "edit model was trained with adjustment conditioning "
                f"(n_adjustment_cond_feats={n_cond}) but no "
                "adj_condition_target was provided")
        env_state.set_adjustment_conditioning(
            float(adj_condition_target),
            weight=(float(adj_condition_weight)
                    if (n_cond >= 2 and adj_condition_weight is not None)
                    else None))
    env_state = model.setup_planning(env_state)

    pre_step_routes = env_state.current_routes.clone()

    if ignore_max_route_len:
        original_max_route_len = env_state.extra_data.max_route_len.clone()
        env_state.extra_data.max_route_len = env_state.n_nodes.clone()
        try:
            action_kinds, action, _, _ = model.step_route_action(
                env_state, greedy=greedy,
                allow_extend=allow_extend,
                allow_trim_start=allow_trim_start,
                allow_trim_end=allow_trim_end,
                allow_halt=allow_halt)
        finally:
            env_state.extra_data.max_route_len = original_max_route_len
    else:
        action_kinds, action, _, _ = model.step_route_action(
            env_state, greedy=greedy,
            allow_extend=allow_extend,
            allow_trim_start=allow_trim_start,
            allow_trim_end=allow_trim_end,
            allow_halt=allow_halt)

    halted = action_kinds == ROUTE_ACTION_HALT
    if n_cond > 0:
        # back to the sentinel: other (unconditioned) models share this state
        env_state.set_adjustment_conditioning(-1.0, weight=None)
    env_state.apply_route_actions(action_kinds, action)

    post_step_routes = env_state.current_routes.clone()

    new_flat_routes = pre_step_routes.clone()
    new_flat_routes[~halted] = post_step_routes[~halted]

    pad_size = max_n_nodes - new_flat_routes.shape[-1]
    if pad_size > 0:
        new_flat_routes = torch.nn.functional.pad(
            new_flat_routes, (0, pad_size), value=-1)

    new_routes = new_flat_routes.reshape(batch_size, n_bees, max_n_nodes)

    result_networks = bee_networks.clone()
    result_networks.scatter_(2, gather_idx, new_routes.unsqueeze(2))

    if not bee_dim:
        result_networks = result_networks.squeeze(1)

    return result_networks


def get_neural_trim_variants(model, env_state, bee_networks, chosen_route_idxs,
                             greedy=False, ignore_max_route_len=False,
                             allow_halt=True):
    """Apply one trim-only edit step per bee.

    This is the BCO type-6 mutation: the selected route is the current route,
    all other slots remain context, and the edit model can only choose
    trim_start, trim_end, or halt. If no trim action is valid, halt keeps the
    route unchanged.
    """
    return get_neural_edit_variants(
        model,
        env_state,
        bee_networks,
        chosen_route_idxs,
        greedy=greedy,
        ignore_max_route_len=ignore_max_route_len,
        allow_extend=False,
        allow_trim_start=True,
        allow_trim_end=True,
        allow_halt=allow_halt,
    )


def get_neural_trim_then_extend_variants(trim_model, extend_model, env_state,
                                         bee_networks, chosen_route_idxs,
                                         greedy=False,
                                         ignore_max_route_len=False,
                                         allow_halt=True):
    """Apply trim-only edit followed by one construction-style extension.

    This is the BCO type-7 compound mutation.  It lets a bee pass through a
    temporarily worse-looking trim before the selected route is immediately
    offered one extension/halt step, and only the combined result is scored.
    """
    if not getattr(trim_model, 'supports_trim_actions', False):
        raise ValueError(
            "get_neural_trim_then_extend_variants requires a trim_model with "
            "supports_trim_actions=True"
        )
    if extend_model is None:
        extend_model = trim_model

    trimmed_networks = get_neural_trim_variants(
        trim_model,
        env_state,
        bee_networks,
        chosen_route_idxs,
        greedy=greedy,
        ignore_max_route_len=ignore_max_route_len,
        allow_halt=allow_halt,
    )
    return get_neural_extend_variants(
        extend_model,
        env_state,
        trimmed_networks,
        chosen_route_idxs,
        greedy=greedy,
        ignore_max_route_len=ignore_max_route_len,
        allow_halt=allow_halt,
    )


def get_neural_variants(model, env_state, bee_networks, drop_route_idxs,
                        greedy=False):
    bee_dim = bee_networks.ndim == 4
    if not bee_dim:
        # there is no bee dimension, so add one
        bee_networks = bee_networks.unsqueeze(1)
        drop_route_idxs = drop_route_idxs.unsqueeze(1)

    # flatten batch and bee dimensions
    batch_size = bee_networks.shape[0]
    n_bees = bee_networks.shape[1]
    n_routes = bee_networks.shape[2]
    keep_mask = torch.ones(bee_networks.shape[:3], dtype=bool, 
                           device=bee_networks.device)
    keep_mask.scatter_(2, drop_route_idxs[..., None], False)
    flat_kept_routes = bee_networks[keep_mask]
    # this works because we remove the same # of routes from each network
    max_n_nodes = bee_networks.shape[3]
    flatbee_kept_routes = flat_kept_routes.reshape(
        batch_size * n_bees, n_routes - 1, max_n_nodes)

    # plan a new route with the model
    env_state.replace_routes(flatbee_kept_routes)
    result = model(env_state, greedy=False)
    env_state = result.state
    routes = tu.get_batch_tensor_from_routes(env_state.routes, 
                                             bee_networks.device)
    if bee_dim:
        routes = routes.reshape(batch_size, n_bees, n_routes, -1)
    pad_size = max_n_nodes - routes.shape[-1]
    routes = torch.nn.functional.pad(routes, (0, pad_size), value=-1)
    return routes


def get_bee_1_variants(remaining_state, batch_bee_routes, direct_sat_dmd_mat,
                       shortest_paths, force_linking_unlinked=False):
    """
    batch_bee_routes: a batch_size x n_bees x n_nodes tensor of routes
    direct_sat_dmd_mat: a batch_size x n_nodes x n_nodes tensor of 
        directly-satisfied demand by the shortest-path route between each pair
        of nodes.
    shortest_paths: a batch_size x n_nodes x n_nodes tensor of the shortest
        paths between each pair of nodes.
    """
    # choose which terminal to keep
    dev = batch_bee_routes.device
    keep_start_term = torch.rand(batch_bee_routes.shape[:2], device=dev) > 0.5

    # choose the new terminal
    # first, compute the demand that would be satisfied by new terminals
    route_lens = (batch_bee_routes > -1).sum(-1)
    batch_idxs = torch.arange(batch_bee_routes.shape[0]).unsqueeze(1)
    route_starts = batch_bee_routes[:, :, 0]
    dsd_from_starts = direct_sat_dmd_mat[batch_idxs, route_starts]
    route_ends = batch_bee_routes.gather(2, route_lens[..., None] - 1)
    route_ends.squeeze_(-1)
    dsd_to_ends = direct_sat_dmd_mat[batch_idxs, :, route_ends]
    # batch_size x n_bees x n_nodes
    new_route_dsds = keep_start_term[..., None] * dsd_from_starts + \
                     ~keep_start_term[..., None] * dsd_to_ends
        
    # then, choose the new terminal proportional to the demand
    # if no demand is satisfiable, set all probs to non-zero for multinomial()
    no_demand_satisfied = new_route_dsds.sum(-1) == 0
    new_route_dsds[no_demand_satisfied] = 1
    # set the terminal that is already in the route to 0, so it can't be chosen
    kept_terms = keep_start_term * route_starts + ~keep_start_term * route_ends
    new_route_dsds.scatter_(2, kept_terms[..., None], 0)

    if force_linking_unlinked:
        # if there are any unlinked node pairs, consider only new routes that 
         # link at least one of them
        sps_from_starts = shortest_paths[batch_idxs, route_starts]
        sps_to_ends = shortest_paths[batch_idxs, :, route_ends]
        candidate_routes = keep_start_term[..., None, None] * sps_from_starts + \
                           ~keep_start_term[..., None, None] * sps_to_ends
        candidate_routes = torch.nn.functional.pad(candidate_routes, 
                                                   (0, 0, 0, 1),)
        extends_if_needed = check_extensions_add_connections(
            remaining_state.has_path, candidate_routes.flatten(0,1))
        # fold the output batch dimension back into (batch, bees)
        shape = tuple(batch_bee_routes.shape[:2]) + (-1,)
        extends_if_needed = extends_if_needed.reshape(shape)
        # only consider routes that extend coverage if not full coverage
        zero_mask = ~extends_if_needed
        nonzero_will_remain = \
            ((new_route_dsds > 0) & extends_if_needed).any(2)
        zero_mask[~nonzero_will_remain] = False        
        new_route_dsds[zero_mask] = 0.0
        # new_route_dsds[~extends_if_needed] = 0.0
     
    # sample the terminals
    flat_new_terms = new_route_dsds.flatten(0,1).multinomial(1).squeeze(-1)
    new_terms = flat_new_terms.reshape(batch_bee_routes.shape[:2])

    # if all demand is zero, dummy might get chosen.  If so, leave route alone.
    dummy_node = direct_sat_dmd_mat.shape[-1] - 1
    chose_dummy = new_terms == dummy_node
    new_terms[chose_dummy] = (~keep_start_term * route_starts + \
                              keep_start_term * route_ends)[chose_dummy]

    new_starts = keep_start_term * route_starts + ~keep_start_term * new_terms
    new_ends = ~keep_start_term * route_ends + keep_start_term * new_terms
    new_routes = shortest_paths[batch_idxs, new_starts, new_ends]

    # pad the end of the new routes to match the existing ones
    n_pad_stops = batch_bee_routes.shape[-1] - new_routes.shape[-1]
    new_routes = torch.nn.functional.pad(new_routes, (0, n_pad_stops), 
                                         value=-1)
    assert ((new_routes > -1).sum(dim=-1) > 0).all()

    return new_routes


def get_bee_2_variants(batch_bee_routes, shorten_prob, are_neighbours):
    """
    batch_bee_routes: a batch_size x n_bees x n_nodes tensor of routes
    shorten_prob: a scalar probability of shortening each route
    are_neighbours: a batch_size x n_nodes x n_nodes boolean tensor of whether
        each node is a neighbour of each other node
        
    """
    bee_dim = batch_bee_routes.ndim == 3
    if bee_dim:
        # flatten the batch and bee dimensions to ease what follows
        flat_routes = batch_bee_routes.flatten(0,1)
        n_bees = batch_bee_routes.shape[1]
    else:
        flat_routes = batch_bee_routes
        n_bees = 1

    # expand and reshape are_neighbours to match flat_routes
    are_neighbours = are_neighbours[:, None].expand(-1, n_bees, -1, -1)
    are_neighbours = are_neighbours.flatten(0,1)
    # convert from boolean to float to allow use of torch.multinomial()
    neighbour_probs = are_neighbours.to(dtype=torch.float32)
    # add a padding column for scattering
    neighbour_probs = torch.nn.functional.pad(neighbour_probs, (0, 1, 0, 1))

    dev = batch_bee_routes.device
    keep_start_term = torch.rand(flat_routes.shape[0], device=dev) > 0.5
    keep_start_term.unsqueeze_(-1)

    route_lens = (flat_routes > -1).sum(-1, keepdim=True)

    # shorten chosen routes at chosen end
    shortened_at_start = flat_routes.roll(shifts=-1, dims=-1)
    shortened_at_start[:, -1] = -1
    shortened_at_end = flat_routes.scatter(1, route_lens - 1, -1)
    shortened = keep_start_term * shortened_at_start + \
        ~keep_start_term * shortened_at_end  

    # extend chosen routes at chosen ends
    n_nodes = are_neighbours.shape[-1]
    # choose new extended start nodes
    route_starts = flat_routes[:, 0]
    rs_gatherer = route_starts[:, None, None].expand(-1, n_nodes+1, 1)
    start_nbr_probs = neighbour_probs.gather(2, rs_gatherer).squeeze(-1)
    # set the probabilities of nodes already on the route to 0
    bbr_scatterer = tu.get_update_at_mask(flat_routes, flat_routes==-1, n_nodes)
    start_nbr_probs.scatter_(1, bbr_scatterer, 0)
    no_start_options = start_nbr_probs.sum(-1) == 0
    start_nbr_probs[no_start_options] = 1
    chosen_start_exts = start_nbr_probs.multinomial(1).squeeze(-1)
    chosen_start_exts[no_start_options] = -1
    extended_starts = flat_routes.roll(shifts=1, dims=-1)
    extended_starts[:, 0] = chosen_start_exts

    # choose new extended end nodes
    route_ends = flat_routes.gather(1, route_lens - 1)
    re_gatherer = route_ends[:, None].repeat(1, 1, n_nodes+1)
    re_gatherer[re_gatherer == -1] = n_nodes
    end_nbr_probs = neighbour_probs.gather(1, re_gatherer).squeeze(-2)
    end_nbr_probs.scatter_(1, bbr_scatterer, 0)
    no_end_options = end_nbr_probs.sum(-1) == 0
    end_nbr_probs[no_end_options] = 1
    chosen_end_exts = end_nbr_probs.multinomial(1)
    chosen_end_exts[no_end_options] = -1
    # pad the routes before scattering, so that full-length routes don't cause
     # an index-out-of-bounds error.
    extended_ends = torch.nn.functional.pad(flat_routes, (0, 1))
    extended_ends = extended_ends.scatter(1, route_lens, chosen_end_exts)
    extended_ends = extended_ends[..., :-1]

    extended_routes = extended_ends * keep_start_term + \
        extended_starts * ~keep_start_term

    # assemble the shortened routes
    shorten = shorten_prob > torch.rand(flat_routes.shape[0], device=dev)
    # remove the last dimension, since we don't need it for gathering anymore
    route_lens = route_lens.squeeze(-1)
    shorten &= route_lens > 2
    shorten.unsqueeze_(-1)
    shortened_part = shortened * shorten
    # assemble the extended routes
    extend_at_start = ~no_start_options[..., None] & ~keep_start_term
    extend_at_end = ~no_end_options[..., None] & keep_start_term
    extend = (extend_at_start | extend_at_end) & ~shorten
    extended_part = extended_routes * extend
    # assemble the unmodified routes (ones with no valid extension)
    keep_same = ~(extend | shorten)
    same_part = flat_routes * keep_same
    # combine the three
    out_routes = shortened_part + extended_part + same_part
    
    out_lens = (out_routes > -1).sum(-1)
    assert ((out_lens - route_lens).abs() <= 1).all()

    # fold back into batch x bees
    if bee_dim:
        out_routes = out_routes.reshape(batch_bee_routes.shape)
    return out_routes


# @hydra.main(version_base=None, config_path="../cfg", config_name="bco_mumford")
def main(cfg: DictConfig, tensors:dict):
    global DEVICE
    use_neural_bees = cfg.get('neural_bees', False)
    if use_neural_bees:
        prefix = 'neural_bco_'
    else:
        prefix = 'bco_'

    DEVICE, run_name, sum_writer, cost_obj, bee_model = \
        lrnu.process_standard_experiment_cfg(cfg, prefix, 
                                             weights_required=True)

    # read in the dataset
    test_ds = get_dataset_from_config(cfg.eval.dataset, tensors=tensors)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    force_linking_unlinked = cfg.get('force_linking_unlinked', False)
    ignore_type4_max_route_len = cfg.get('ignore_type4_max_route_len', False)
    ignore_type5_max_route_len = cfg.get('ignore_type5_max_route_len', False)
    ignore_type6_max_route_len = cfg.get('ignore_type6_max_route_len', False)
    ignore_type7_max_route_len = cfg.get('ignore_type7_max_route_len', False)
    type4_allow_halt = cfg.get('type4_allow_halt', True)
    type5_allow_halt = cfg.get('type5_allow_halt', True)
    type6_allow_halt = cfg.get('type6_allow_halt', True)
    type7_allow_halt = cfg.get('type7_allow_halt', True)
    use_demand_weighted_route_selection = \
        cfg.get('use_demand_weighted_route_selection', False)
    worse_accept_temperature = cfg.get('worse_accept_temperature', 0.0)
    worse_accept_decay = cfg.get('worse_accept_decay', 0.995)
    worse_accept_min_temperature = \
        cfg.get('worse_accept_min_temperature', 0.001)
    worse_selection_temperature = cfg.get('worse_selection_temperature', 0.0)
    worse_selection_decay = cfg.get('worse_selection_decay', 0.995)
    worse_selection_min_temperature = \
        cfg.get('worse_selection_min_temperature', 0.001)
    worse_selection_uniform_mix = cfg.get('worse_selection_uniform_mix', 0.05)
    worse_selection_elite_count = cfg.get('worse_selection_elite_count', 1)
    trim_grace_period = cfg.get('trim_grace_period', 0)

    if not use_neural_bees:
        bee_model = None
        edit_model = None
    elif bee_model is not None:
        bee_model.force_linking_unlinked = force_linking_unlinked
        bee_model.eval()
        edit_model = bee_model if getattr(
            bee_model, 'supports_trim_actions', False) else None
    else:
        edit_model = None

    nt1b = cfg.get('n_type1_bees', None)
    nt2b = cfg.get('n_type2_bees', None)
    nt4b = cfg.get('n_type4_bees', 0)
    nt5b = cfg.get('n_type5_bees', 0)
    nt6b = cfg.get('n_type6_bees', 0)
    nt7b = cfg.get('n_type7_bees', 0)
    adjustment_degree_weight = cfg.get('adjustment_degree_weight', 0.0)
    adjustment_degree_gap = cfg.get('adjustment_degree_gap', 0.1)
    adjustment_degree_mode = cfg.get('adjustment_degree_mode', 'current')
    adjustment_degree_objective = cfg.get('adjustment_degree_objective', 'raw')
    adjustment_degree_target = cfg.get('adjustment_degree_target', 0.2)
    test_output = \
        lrnu.test_method(bee_colony, test_dl, cfg.eval, cfg.init, cost_obj, 
            sum_writer=sum_writer, silent=True, n_bees=cfg.n_bees,
            n_iterations=cfg.n_iterations, n_type1_bees=nt1b, n_type2_bees=nt2b,  
            n_type4_bees=nt4b, n_type5_bees=nt5b, n_type6_bees=nt6b,
            n_type7_bees=nt7b,
            device=DEVICE, bee_model=bee_model, edit_model=edit_model,
            return_routes=True,
            force_linking_unlinked=force_linking_unlinked,
            adjustment_degree_weight=adjustment_degree_weight,
            adjustment_degree_gap=adjustment_degree_gap,
            adjustment_degree_mode=adjustment_degree_mode,
            adjustment_degree_objective=adjustment_degree_objective,
            adjustment_degree_target=adjustment_degree_target,
            ignore_type4_max_route_len=ignore_type4_max_route_len,
            ignore_type5_max_route_len=ignore_type5_max_route_len,
            ignore_type6_max_route_len=ignore_type6_max_route_len,
            ignore_type7_max_route_len=ignore_type7_max_route_len,
            type4_allow_halt=type4_allow_halt,
            type5_allow_halt=type5_allow_halt,
            type6_allow_halt=type6_allow_halt,
            type7_allow_halt=type7_allow_halt,
            use_demand_weighted_route_selection=
            use_demand_weighted_route_selection,
            worse_accept_temperature=worse_accept_temperature,
            worse_accept_decay=worse_accept_decay,
            worse_accept_min_temperature=worse_accept_min_temperature,
            worse_selection_temperature=worse_selection_temperature,
            worse_selection_decay=worse_selection_decay,
            worse_selection_min_temperature=worse_selection_min_temperature,
            worse_selection_uniform_mix=worse_selection_uniform_mix,
            worse_selection_elite_count=worse_selection_elite_count,
            trim_grace_period=trim_grace_period)
    routes = test_output[-1]
    metrics = test_output[-2]
    unserved_demand = test_output[-3]
    
    # save the final routes that were produced
    tu.dump_routes(run_name, routes)
    return metrics, unserved_demand
    

if __name__ == "__main__":
    main()
