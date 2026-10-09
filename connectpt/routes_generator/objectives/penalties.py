"""Constraint, adjustment and edit penalties, independent of policy and search."""
import torch
from .const import STATE_PENALTIES, ACTION_PENALTIES, PENALTIES

# Order of the penalty vector: cost computes the state penalties, the environment
# step appends the action penalties.


def constraint_penalty(state, result, ignore_stops_oob):
    fraction = result.n_disconnected_demand_edges / state.n_demand_edges
    penalty = fraction + 0.1 * (fraction > 0)
    if not ignore_stops_oob:
        denominator = state.n_routes_to_plan * state.min_route_len
        denominator = denominator.clone()
        denominator[denominator == 0] = 1
        fraction = result.n_stops_oob / denominator
        penalty += fraction + 0.1 * (fraction > 0)
    return penalty


def edit_step_penalty(action, active, weight):
    return ((action != 3) & active).to(torch.float32) * weight


def forced_halt_penalty(active, forced, weight):
    return active.to(torch.float32) * weight if forced else torch.zeros_like(active, dtype=torch.float32)


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
    dp = torch.zeros(batch_size, n_rows + 1, n_cols + 1, device=scores.device, dtype=scores.dtype)
    shifted_scores = scores - gap

    for diag_idx in range(2, n_rows + n_cols + 1):
        row_min = max(1, diag_idx - n_cols)
        row_max = min(n_rows, diag_idx - 1)
        row_idxs = torch.arange(row_min, row_max + 1, device=scores.device)
        col_idxs = diag_idx - row_idxs
        up = dp[:, row_idxs - 1, col_idxs]
        left = dp[:, row_idxs, col_idxs - 1]
        diag = dp[:, row_idxs - 1, col_idxs - 1] + shifted_scores[:, row_idxs - 1, col_idxs - 1]
        dp[:, row_idxs, col_idxs] = torch.maximum(torch.maximum(up, left), diag)

    return dp[:, -1, -1]


def _get_alignment_scores(candidate_routes, reference_routes, symmetric_routes, gap):
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
                    flat_candidates[differing], flat_references[differing], symmetric_routes, gap
                )
                alignment_scores = alignment_scores.clone()
                alignment_scores[differing] = diff_scores
            return alignment_scores, cand_lens, ref_lens

    pairwise_matches = flat_candidates[:, :, None] == flat_references[:, None, :]
    valid_pairwise_matches = pairwise_matches & cand_valid[:, :, None] & ref_valid[:, None, :]
    alignment_scores = needleman_wunsch(valid_pairwise_matches.to(torch.float32), gap=gap)

    if symmetric_routes:
        reversed_refs = _reverse_padded_routes(flat_references)
        reversed_valid = reversed_refs > -1
        reversed_matches = flat_candidates[:, :, None] == reversed_refs[:, None, :]
        valid_reversed_matches = reversed_matches & cand_valid[:, :, None] & reversed_valid[:, None, :]
        reversed_scores = needleman_wunsch(valid_reversed_matches.to(torch.float32), gap=gap)
        alignment_scores = torch.maximum(alignment_scores, reversed_scores)

    cand_lens = cand_valid.sum(dim=-1)
    ref_lens = ref_valid.sum(dim=-1)
    return alignment_scores, cand_lens, ref_lens


def get_current_adjustment_degrees(candidate_routes, reference_routes, symmetric_routes, gap=0.1):
    """Return the current normalized alignment-based adjustment degree."""
    alignment_scores, cand_lens, ref_lens = _get_alignment_scores(
        candidate_routes, reference_routes, symmetric_routes, gap=gap
    )
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


def get_paper_adjustment_degrees(candidate_routes, reference_routes, symmetric_routes):
    """Return the paper-style adjustment degree based on LCS F-measure."""
    lcs_lengths, cand_lens, ref_lens = _get_alignment_scores(
        candidate_routes, reference_routes, symmetric_routes, gap=0.0
    )
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


def adjustment_degree(candidate_routes, reference_routes, symmetric_routes, gap=0.1, mode="current"):
    """Return route-wise adjustment degree in [0, 1], 0 means unchanged."""
    if mode == "current":
        return get_current_adjustment_degrees(
            candidate_routes,
            reference_routes,
            symmetric_routes,
            gap=gap,
        )
    if mode == "paper":
        return get_paper_adjustment_degrees(
            candidate_routes,
            reference_routes,
            symmetric_routes,
        )
    raise ValueError(f"Unknown adjustment degree mode '{mode}'. Expected 'current' or 'paper'.")


def adjustment_penalty(adjustment_degrees, objective="raw", target=0.2):
    """Convert raw adjustment degrees into the penalty used in the objective."""
    if objective == "raw":
        return adjustment_degrees
    if objective == "target":
        if not bool(((torch.as_tensor(target) >= 0) & (torch.as_tensor(target) <= 1)).all()):
            raise ValueError(f"adjustment_degree_target must be in [0, 1], got {target}")
        target_tensor = adjustment_degrees.new_tensor(target)
        return (adjustment_degrees - target_tensor).abs()
    if objective == "cap":
        # One-sided (hinge) penalty: penalize only EXCEEDING the target, i.e.
        # treat `target` as an upper bound the routes should not pass. No
        # penalty (and no pull) while adjustment <= target.
        if not bool(((torch.as_tensor(target) >= 0) & (torch.as_tensor(target) <= 1)).all()):
            raise ValueError(f"adjustment_degree_target must be in [0, 1], got {target}")
        target_tensor = adjustment_degrees.new_tensor(target)
        return (adjustment_degrees - target_tensor).clamp(min=0.0)
    if objective == "cap_sq":
        # Quadratic one-sided penalty: max(0, adj - target)**2. Same upper-bound
        # semantics as 'cap' (no penalty below target), but the gradient
        # 2*(adj - target) GROWS with the overshoot -- gentle near the target
        # (does not swamp the main objective / keeps return variance low) and
        # firm on large excess. Unlike linear 'cap' (constant gradient = W),
        # the optimal adjustment depends on the target, so target/W become real
        # controllability levers under conditioning.
        if not bool(((torch.as_tensor(target) >= 0) & (torch.as_tensor(target) <= 1)).all()):
            raise ValueError(f"adjustment_degree_target must be in [0, 1], got {target}")
        target_tensor = adjustment_degrees.new_tensor(target)
        return (adjustment_degrees - target_tensor).clamp(min=0.0) ** 2
    raise ValueError(
        f"Unknown adjustment degree objective '{objective}'. " "Expected 'raw', 'target', 'cap', or 'cap_sq'."
    )
