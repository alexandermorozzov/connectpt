"""Validity masks shared by construction, edit and search."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from .features import aggr_edges_over_sequences
from .const import TORCH_FMAX, TORCH_FMIN


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


@dataclass(frozen=True)
class ActionMask:
    extend: object = True
    trim_start: object = True
    trim_end: object = True
    halt: object = True
    candidates: CandidateMasks | None = None

    @classmethod
    def from_names(cls, names):
        return cls(**{name: name in names for name in ("extend", "trim_start", "trim_end", "halt")})

    def require_candidates(self):
        if not isinstance(self.candidates, CandidateMasks):
            raise TypeError('policy requires candidate masks prepared by the environment')
        return self.candidates


@dataclass(frozen=True)
class CandidateMasks:
    """Bool tensors on the State device; paths [B,N,N], trims [B,L], halt [B]."""
    path_valid: torch.Tensor
    before_valid: torch.Tensor
    after_valid: torch.Tensor
    trim_start_raw: torch.Tensor
    trim_end_raw: torch.Tensor
    trim_start_coverage: torch.Tensor
    trim_end_coverage: torch.Tensor
    extend_valid: torch.Tensor
    trim_start_valid: torch.Tensor
    trim_end_valid: torch.Tensor
    halt_blocked: torch.Tensor
    halt_forced: torch.Tensor
    old_route_is_done: torch.Tensor

    def __post_init__(self):
        if self.path_valid.ndim != 3 or self.path_valid.shape[1] != self.path_valid.shape[2]:
            raise ValueError('path candidate masks must have shape [B,N,N]')
        if self.trim_start_raw.ndim != 2 or self.trim_start_raw.shape[0] != self.path_valid.shape[0]:
            raise ValueError('trim candidate masks must have shape [B,L]')
        for field in fields(self):
            value = getattr(self, field.name)
            if value.dtype != torch.bool or value.device != self.path_valid.device:
                raise ValueError('candidate masks must be bool tensors on one device')
            if field.name in ('path_valid', 'before_valid', 'after_valid', 'extend_valid'):
                shape = self.path_valid.shape
            elif field.name in ('halt_blocked', 'halt_forced', 'old_route_is_done'):
                shape = self.path_valid.shape[:1]
            else:
                shape = self.trim_start_raw.shape
            if value.shape != shape:
                raise ValueError(f'{field.name} must have shape {tuple(shape)}, got {tuple(value.shape)}')

    def index_select(self, indices):
        return CandidateMasks(**{field.name: getattr(self, field.name)[indices] for field in fields(self)})

    @classmethod
    def concatenate(cls, masks, device):
        return cls(**{field.name: torch.cat([getattr(mask, field.name).to(device) for mask in masks])
                      for field in fields(cls)})


def allow_candidates(valid, allowed):
    allowed = torch.as_tensor(allowed, device=valid.device, dtype=torch.bool)
    if allowed.ndim == 0:
        return valid if bool(allowed) else torch.zeros_like(valid)
    if allowed.shape != (valid.shape[0],):
        raise ValueError('action family mask must have shape [B]')
    return valid & allowed.reshape((-1,) + (1,) * (valid.ndim - 1))


def extension_validity(state, paths, path_lens, invalid_terms, force_linking_unlinked, *, before):
    lengths = state.current_route_n_stops
    last = (lengths - 1).clamp_min(0)
    terminals = state.current_routes[:, 0] if before else state.current_routes.gather(1, last[:, None]).squeeze(-1)
    if before:
        paths = paths.transpose(-3, -2)
        path_lens = path_lens.transpose(-2, -1)
    batch = state.batch_indices
    raw = paths[batch, terminals]
    extensions = raw.clone()
    extensions[extensions == terminals[:, None, None]] = -1
    occupied = torch.zeros((state.batch_size, state.max_n_nodes + 1), device=state.device, dtype=torch.bool)
    occupied[batch[:, None], state.current_routes] = True
    occupied[:, -1] = False
    new_lengths = path_lens[batch, terminals] + lengths[:, None] - 1
    invalid = extension_mask(state, occupied, extensions, raw, path_lens, terminals, new_lengths, force_linking_unlinked)
    valid = torch.zeros_like(path_lens, dtype=torch.bool)
    valid[batch, terminals] = ~invalid
    if before:
        valid = valid.transpose(-2, -1)
    return valid & ~invalid_terms


def prepare_action_mask(obs, params, role, families):
    """Compute all task validity before policy scoring, without model or RNG calls."""
    path_lens, lengths, invalid = path_mask(obs, obs.paths, params.max_act_len, params.force_linking_unlinked)
    before = extension_validity(obs, obs.paths, path_lens, invalid, params.force_linking_unlinked, before=True)
    after = extension_validity(obs, obs.paths, path_lens, invalid, params.force_linking_unlinked, before=False)
    starting = ~obs.done & (lengths == 0)
    extension = before | after
    old_done = obs.done | invalid.all(-1).all(-1) if role == 'construction' else obs.done
    continuing = ~starting & ~old_done
    valid = torch.where(continuing[:, None, None], extension, ~invalid)
    extend_valid = allow_candidates(valid, families.extend)
    start_raw = obs.trim_start[1] & (obs.trim_start[2] >= 0) & (obs.trim_start[3] >= 0)
    end_raw = obs.trim_end[1] & (obs.trim_end[2] >= 0) & (obs.trim_end[3] >= 0)
    linked = obs.has_path.all(-1).all(-1)
    coverage = linked if params.force_linking_unlinked else torch.ones_like(linked)
    start_coverage = start_raw & coverage[:, None]
    end_coverage = end_raw & coverage[:, None]
    start_valid = allow_candidates(start_coverage, families.trim_start)
    end_valid = allow_candidates(end_coverage, families.trim_end)
    blocked = (lengths < obs.min_route_len) | ~coverage
    if role == 'construction':
        extend_allowed = allow_candidates(torch.ones_like(obs.done), families.extend)
        old_done = obs.done | invalid.all(-1).all(-1) | ~extend_allowed
        forced = old_done | (~starting & ~extension.any(-1).any(-1))
    elif role == 'edit':
        no_action = ~(extend_valid.flatten(1).any(-1) | start_valid.any(-1) | end_valid.any(-1))
        old_done = obs.done | no_action
        forced = old_done
        if params.forbid_halt_when_overlong:
            can_trim = start_coverage.any(-1) | end_coverage.any(-1)
            blocked = blocked | ((lengths > obs.max_route_len) & can_trim & ~old_done)
    else:
        raise ValueError(f'unsupported action task {role}')
    blocked = blocked & ~forced
    allowed = allow_candidates(torch.ones_like(obs.done), families.halt)
    restricted = ~old_done & ~allowed
    blocked = blocked | restricted
    forced = forced & ~restricted
    candidates = CandidateMasks(~invalid, before, after, start_raw, end_raw, start_coverage, end_coverage,
                                extend_valid, start_valid, end_valid, blocked, forced, old_done)
    return replace(families, candidates=candidates)


def apply_halt_mask(scores, candidates):
    scores = scores.clone()
    scores[candidates.halt_blocked] = TORCH_FMIN
    scores[candidates.halt_forced] = TORCH_FMAX
    return scores


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
