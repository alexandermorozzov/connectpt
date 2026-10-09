from connectpt.routes_generator.env.features import aggr_edges_over_sequences

"""Validity masks shared by construction, edit and search."""
import torch


def check_extensions_add_connections(is_linked, extensions):
    # determine if any path on its own makes new connections
    path_makes_new_connection = aggr_edges_over_sequences(extensions, (~is_linked[..., None]))
    path_makes_new_connection = path_makes_new_connection.bool().squeeze(-1)

    # if a graph is not fully connected, then any path that doesn't make a
    # new connection is invalid.
    is_fully_connected = is_linked.all(dim=-1).all(dim=-1)
    while is_fully_connected.ndim < path_makes_new_connection.ndim:
        is_fully_connected.unsqueeze_(-1)
    has_valid_coverage = is_fully_connected | path_makes_new_connection

    return has_valid_coverage


def mask_action_family(scores, valid, allowed):
    """Mask whole batch rows when an action family is disabled."""
    if isinstance(allowed, bool):
        if allowed:
            return scores, valid
        return torch.full_like(scores, TORCH_FMIN), torch.zeros_like(valid)

    allowed = torch.as_tensor(allowed, dtype=torch.bool, device=scores.device)
    if allowed.ndim == 0:
        if bool(allowed.item()):
            return scores, valid
        return torch.full_like(scores, TORCH_FMIN), torch.zeros_like(valid)
    if allowed.ndim != 1 or allowed.shape[0] != scores.shape[0]:
        raise ValueError(
            "Per-batch action allow mask must have shape (batch_size,), "
            f"got {tuple(allowed.shape)} for batch size {scores.shape[0]}"
        )
    disallowed = ~allowed
    if not disallowed.any():
        return scores, valid

    scores = scores.clone()
    valid = valid.clone()
    scores[disallowed] = TORCH_FMIN
    valid[disallowed] = False
    return scores, valid


def set_invalid_scores_to_fmin(scores, valid_terminals_mat):
    scores_holder = torch.full_like(scores, TORCH_FMIN)
    no_valid_options = valid_terminals_mat.sum(dim=(1, 2)) == 0
    if no_valid_options.any():
        # no valid options, so make all terminal pairs equally likely
        off_diag = ~torch.eye(scores.shape[-1], device=scores.device, dtype=bool)
        where_nvo = torch.where(no_valid_options)[0]
        scores_holder[where_nvo[:, None], off_diag] = 0

    # valid_terminals_mat keeps changing, so clone so pytorch doesn't
    # complain during backprop
    vtm = valid_terminals_mat.clone()
    scores_holder[vtm] = scores[vtm]

    return scores_holder


from dataclasses import dataclass
from .const import TORCH_FMAX, TORCH_FMIN


@dataclass(frozen=True)
class ActionMask:
    extend: object = True
    trim_start: object = True
    trim_end: object = True
    halt: object = True

    @classmethod
    def from_names(cls, names):
        return cls(**{name: name in names for name in ("extend", "trim_start", "trim_end", "halt")})


def path_mask(state, path_seqs, max_act_len, force_linking_unlinked):
    path_lens = (path_seqs > -1).sum(dim=-1)
    current_route_lens = state.current_route_n_stops
    overlap = (current_route_lens > 0)[:, None, None].to(path_lens.dtype)
    added_path_lens = (path_lens - overlap).clamp_min(0)
    post_ext_lens = current_route_lens[:, None, None] + added_path_lens
    exts_are_too_long = post_ext_lens > state.max_route_len[:, None, None]
    paths_are_invalid = exts_are_too_long | ~state.valid_terms_mat
    if max_act_len is not None:
        path_too_long = path_lens > max_act_len
        paths_are_invalid = paths_are_invalid | path_too_long
    if force_linking_unlinked:
        extends_coverage_if_needed = check_extensions_add_connections(state.has_path, path_seqs)
        paths_are_invalid |= ~extends_coverage_if_needed
    return path_lens, current_route_lens, paths_are_invalid


def extension_mask(
    state,
    are_on_routes,
    extension_paths,
    raw_extension_paths,
    path_lens,
    terminal_nodes,
    new_lens,
    force_linking_unlinked,
):
    batch_idxs = state.batch_indices
    revisits = are_on_routes[batch_idxs[:, None, None], extension_paths]
    revisits = revisits.any(dim=-1)
    is_empty_path = path_lens[batch_idxs, terminal_nodes] == 0
    invalid = revisits | is_empty_path | (new_lens > state.max_route_len[:, None])
    if force_linking_unlinked:
        extends_coverage_if_needed = check_extensions_add_connections(state.has_path, raw_extension_paths)
        invalid |= ~extends_coverage_if_needed
    return invalid


def restrict_halt(scores, forced, allowed):
    if isinstance(allowed, bool):
        if not allowed:
            scores = scores.clone()
            scores[~forced] = TORCH_FMIN
        return scores
    disallowed = ~torch.as_tensor(allowed, device=scores.device, dtype=torch.bool)
    scores = scores.clone()
    scores[~forced & disallowed] = TORCH_FMIN
    return scores


def construction_halt_scores(
    state, halt_scores, current_route_lens, old_route_is_done, starting, ext_valid, force_linking_unlinked, allowed
):
    if force_linking_unlinked:
        not_fully_linked = ~(state.has_path.all(-1).all(-1))
        halt_scores[not_fully_linked] = TORCH_FMIN
    halt_scores[current_route_lens < state.min_route_len] = TORCH_FMIN
    halt_scores[old_route_is_done] = TORCH_FMAX
    no_valid_ext = ~ext_valid.any(-1).any(-1)
    halt_scores[~starting & no_valid_ext] = TORCH_FMAX
    return restrict_halt(halt_scores, old_route_is_done, allowed)


def trim_coverage_mask(state, start_scores, start_valid, end_scores, end_valid, force_linking_unlinked):
    if force_linking_unlinked:
        not_fully_linked = ~(state.has_path.all(-1).all(-1))
        start_scores[not_fully_linked] = TORCH_FMIN
        end_scores[not_fully_linked] = TORCH_FMIN
        start_valid[not_fully_linked] = False
        end_valid[not_fully_linked] = False
    return start_scores, start_valid, end_scores, end_valid


def edit_halt_scores(
    state,
    halt_scores,
    current_route_lens,
    no_route_action_yet,
    trim_start_valid,
    trim_end_valid,
    force_linking_unlinked,
    forbid_halt_when_overlong,
    allowed,
):
    if force_linking_unlinked:
        not_fully_linked = ~(state.has_path.all(-1).all(-1))
        halt_scores[not_fully_linked] = TORCH_FMIN
    halt_scores[current_route_lens < state.min_route_len] = TORCH_FMIN
    old_route_is_done = state.is_done() | no_route_action_yet
    halt_scores[old_route_is_done] = TORCH_FMAX
    if forbid_halt_when_overlong:
        overlong_routes = current_route_lens > state.max_route_len
        batch_size = state.batch_size
        can_trim = trim_start_valid.reshape(batch_size, -1).any(-1) | trim_end_valid.reshape(batch_size, -1).any(-1)
        repairable_overlong = overlong_routes & can_trim & ~old_route_is_done
        halt_scores[repairable_overlong] = TORCH_FMIN
    return restrict_halt(halt_scores, old_route_is_done, allowed), old_route_is_done
