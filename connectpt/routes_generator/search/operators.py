"""Four paper operators; neural mutations predict on owned environment state."""
from dataclasses import dataclass, replace
from typing import Any
import torch
from ..common.tensors import get_update_at_mask
from ..env.masks import ActionMask, check_extensions_add_connections
from ..env.features import aggr_edges_over_sequences
from ..policy.predict import predict


def _predict_selected_routes(model, state, networks, chosen, *, cost, binding,
                             greedy=False, max_steps=None, mask=None, ignore_max_route_len=False):
    """Predict on an owned B × bees batch; the caller's State stays untouched."""
    params = replace(binding.params, n_routes=state.n_routes_to_plan,
                     min_route_len=state.min_route_len,
                     max_route_len=state.n_nodes if ignore_max_route_len else state.max_route_len)
    reference = state.adjustment_seed if state.adjustment_seed.numel() else None
    prediction = predict(
        model, state.graph_data, state.cost_weights, cost, binding.factory, params,
        'greedy' if greedy else 'sample', seed_networks=networks.flatten(0, 1),
        reference_networks=reference, route_idx=chosen.flatten(), max_steps=max_steps, mask=mask,
    )
    routes = prediction.routes[torch.arange(chosen.numel(), device=chosen.device), chosen.flatten()]
    routes = torch.nn.functional.pad(routes, (0, networks.shape[-1] - routes.shape[-1]), value=-1)
    return routes.reshape(*networks.shape[:2], networks.shape[-1])


def get_direct_sat_dmd(stop_dmd, shortest_paths, symmetric):
    direct_sat_dmd = torch.zeros_like(stop_dmd)
    summed_demands = aggr_edges_over_sequences(shortest_paths, stop_dmd[..., None])
    if symmetric:
        summed_demands += aggr_edges_over_sequences(shortest_paths.transpose(1, 2), stop_dmd.transpose(1, 2)[..., None])
    direct_sat_dmd[:, :-1, :-1] = summed_demands.squeeze(-1)

    # assumes there's a dummy column and row
    direct_sat_dmd[:, -1] = 0
    direct_sat_dmd[:, :, -1] = 0
    return direct_sat_dmd


def _index_single_bee(bee_networks, chosen_route_idxs, bee_idx):
    bee_idx = bee_idx.reshape(1)
    return (bee_networks.index_select(1, bee_idx), chosen_route_idxs.index_select(1, bee_idx))


def _stack_bee_route_outputs(outputs, bee_networks):
    if outputs:
        return torch.stack(outputs, dim=1)
    return bee_networks.new_full((bee_networks.shape[0], 0, bee_networks.shape[-1]), -1)


def _gather_chosen_routes_from_networks(networks, chosen_route_idxs):
    max_n_nodes = networks.shape[-1]
    gather_idx = chosen_route_idxs[..., None, None].expand(-1, -1, -1, max_n_nodes)
    return networks.gather(2, gather_idx).squeeze(2)


def _run_rebuild_variants_for_bees(variant_fn, model, env_state, bee_networks, chosen_route_idxs, bee_idxs, *, cost,binding):
    """Run a rebuild-style neural variant one bee at a time."""
    outputs = []
    for bee_idx in bee_idxs.unbind(0):
        bee_slice, chosen_slice = _index_single_bee(bee_networks, chosen_route_idxs, bee_idx)
        networks = variant_fn(model, env_state, bee_slice, chosen_slice, cost=cost,binding=binding)
        outputs.append(networks[:, 0, -1])
    return _stack_bee_route_outputs(outputs, bee_networks)


def _run_selected_route_variants_for_bees(variant_fn, *variant_args, **variant_kwargs):
    """Run one-step neural/edit variants one bee at a time.

    The final three positional args are ``bee_networks``,
    ``chosen_route_idxs``, and ``bee_idxs`` preceded by any model/env args
    expected by ``variant_fn``.
    """
    *prefix_args, bee_networks, chosen_route_idxs, bee_idxs = variant_args
    outputs = []
    for bee_idx in bee_idxs.unbind(0):
        bee_slice, chosen_slice = _index_single_bee(bee_networks, chosen_route_idxs, bee_idx)
        networks = variant_fn(*prefix_args, bee_slice, chosen_slice, **variant_kwargs)
        routes = _gather_chosen_routes_from_networks(networks, chosen_slice)
        outputs.append(routes[:, 0])
    return _stack_bee_route_outputs(outputs, bee_networks)


def get_new_route_variants(
    batch_bee_routes, direct_sat_dmd_mat, shortest_paths, force_linking_unlinked=False, remaining_state=None
):
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
    starts = torch.div(flat_choices, n_nodes, rounding_mode="floor")
    ends = flat_choices % n_nodes
    new_routes = shortest_paths[repeated_batch_idxs, starts, ends]
    new_routes = new_routes.reshape(batch_size, n_bees, -1)

    pad_size = max_n_nodes - new_routes.shape[-1]
    if pad_size > 0:
        new_routes = torch.nn.functional.pad(new_routes, (0, pad_size), value=-1)

    return new_routes


def edit_route(
    model,
    env_state,
    bee_networks,
    chosen_route_idxs,
    greedy=False,
    ignore_max_route_len=False,
    allow_extend=True,
    allow_trim_start=True,
    allow_trim_end=True,
    allow_halt=True,
    adj_condition_target=None,
    adj_condition_weight=None,
    *,
    cost,
    binding,
):
    """Apply one selected-route prediction per bee, with owned environment state."""
    if not model.supports_trim_actions:
        raise ValueError("edit_route requires a model with " "supports_trim_actions=True")

    if bee_networks.ndim != 4:
        raise ValueError('neural mutations require networks [B,bees,R,L]')
    n_cond = model.n_adjustment_cond_feats
    if n_cond > 0:
        if adj_condition_target is None:
            raise ValueError(
                "edit model was trained with adjustment conditioning "
                f"(n_adjustment_cond_feats={n_cond}) but no "
                "adj_condition_target was provided"
            )
        cost = cost.with_params(
            adjustment_degree_target=float(adj_condition_target),
            adjustment_degree_weight=float(adj_condition_weight) if adj_condition_weight is not None else 0.,
        )
    new_routes = _predict_selected_routes(
        model, env_state, bee_networks, chosen_route_idxs, cost=cost, binding=binding,
        greedy=greedy, max_steps=1, ignore_max_route_len=ignore_max_route_len,
        mask=ActionMask(allow_extend, allow_trim_start, allow_trim_end, allow_halt),
    )
    gather_idx = chosen_route_idxs[..., None, None].expand(-1, -1, -1, bee_networks.shape[-1])
    result_networks = bee_networks.clone()
    result_networks.scatter_(2, gather_idx, new_routes.unsqueeze(2))

    return result_networks


def rebuild_route(model, env_state, bee_networks, drop_route_idxs, *, cost,binding):
    """Return the remaining routes followed by the predicted selected route."""
    if bee_networks.ndim != 4:
        raise ValueError('neural mutations require networks [B,bees,R,L]')
    batch_size, n_bees, n_routes, max_n_nodes = bee_networks.shape
    keep_mask = torch.ones(bee_networks.shape[:3], dtype=bool, device=bee_networks.device)
    keep_mask.scatter_(2, drop_route_idxs[..., None], False)
    flat_kept_routes = bee_networks[keep_mask]
    # this works because we remove the same # of routes from each network
    flatbee_kept_routes = flat_kept_routes.reshape(batch_size * n_bees, n_routes - 1, max_n_nodes)

    selected = _predict_selected_routes(model, env_state, bee_networks, drop_route_idxs,
                                        cost=cost, binding=binding)
    return torch.cat((flatbee_kept_routes.reshape(batch_size, n_bees, n_routes - 1, max_n_nodes),
                      selected[:, :, None]), dim=2)


def get_bee_1_variants(
    remaining_state, batch_bee_routes, direct_sat_dmd_mat, shortest_paths, force_linking_unlinked=False
):
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
    new_route_dsds = keep_start_term[..., None] * dsd_from_starts + ~keep_start_term[..., None] * dsd_to_ends

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
        candidate_routes = (
            keep_start_term[..., None, None] * sps_from_starts + ~keep_start_term[..., None, None] * sps_to_ends
        )
        candidate_routes = torch.nn.functional.pad(
            candidate_routes,
            (0, 0, 0, 1),
        )
        extends_if_needed = check_extensions_add_connections(remaining_state.has_path, candidate_routes.flatten(0, 1))
        # fold the output batch dimension back into (batch, bees)
        shape = tuple(batch_bee_routes.shape[:2]) + (-1,)
        extends_if_needed = extends_if_needed.reshape(shape)
        # only consider routes that extend coverage if not full coverage
        zero_mask = ~extends_if_needed
        nonzero_will_remain = ((new_route_dsds > 0) & extends_if_needed).any(2)
        zero_mask[~nonzero_will_remain] = False
        new_route_dsds[zero_mask] = 0.0
        # new_route_dsds[~extends_if_needed] = 0.0

    # sample the terminals
    flat_new_terms = new_route_dsds.flatten(0, 1).multinomial(1).squeeze(-1)
    new_terms = flat_new_terms.reshape(batch_bee_routes.shape[:2])

    # if all demand is zero, dummy might get chosen.  If so, leave route alone.
    dummy_node = direct_sat_dmd_mat.shape[-1] - 1
    chose_dummy = new_terms == dummy_node
    new_terms[chose_dummy] = (~keep_start_term * route_starts + keep_start_term * route_ends)[chose_dummy]

    new_starts = keep_start_term * route_starts + ~keep_start_term * new_terms
    new_ends = ~keep_start_term * route_ends + keep_start_term * new_terms
    new_routes = shortest_paths[batch_idxs, new_starts, new_ends]

    # pad the end of the new routes to match the existing ones
    n_pad_stops = batch_bee_routes.shape[-1] - new_routes.shape[-1]
    new_routes = torch.nn.functional.pad(new_routes, (0, n_pad_stops), value=-1)
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
        flat_routes = batch_bee_routes.flatten(0, 1)
        n_bees = batch_bee_routes.shape[1]
    else:
        flat_routes = batch_bee_routes
        n_bees = 1

    # expand and reshape are_neighbours to match flat_routes
    are_neighbours = are_neighbours[:, None].expand(-1, n_bees, -1, -1)
    are_neighbours = are_neighbours.flatten(0, 1)
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
    shortened = keep_start_term * shortened_at_start + ~keep_start_term * shortened_at_end

    # extend chosen routes at chosen ends
    n_nodes = are_neighbours.shape[-1]
    # choose new extended start nodes
    route_starts = flat_routes[:, 0]
    rs_gatherer = route_starts[:, None, None].expand(-1, n_nodes + 1, 1)
    start_nbr_probs = neighbour_probs.gather(2, rs_gatherer).squeeze(-1)
    # set the probabilities of nodes already on the route to 0
    bbr_scatterer = get_update_at_mask(flat_routes, flat_routes == -1, n_nodes)
    start_nbr_probs.scatter_(1, bbr_scatterer, 0)
    no_start_options = start_nbr_probs.sum(-1) == 0
    start_nbr_probs[no_start_options] = 1
    chosen_start_exts = start_nbr_probs.multinomial(1).squeeze(-1)
    chosen_start_exts[no_start_options] = -1
    extended_starts = flat_routes.roll(shifts=1, dims=-1)
    extended_starts[:, 0] = chosen_start_exts

    # choose new extended end nodes
    route_ends = flat_routes.gather(1, route_lens - 1)
    re_gatherer = route_ends[:, None].repeat(1, 1, n_nodes + 1)
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

    extended_routes = extended_ends * keep_start_term + extended_starts * ~keep_start_term

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


@dataclass
class MutationContext:
    """Everything a mutation operator may need at one mod step."""

    bee_networks: Any  # (batch, n_bees, n_routes, max_n_nodes)
    chosen_route_idxs: Any  # (batch, n_bees)
    modified_routes: Any  # (batch, n_bees, max_n_nodes), empties refilled
    env_state: Any  # bee-expanded RouteGenBatchState
    seq_state: Any  # single-bee state (sequential mode) or env_state
    remaining_state: Any  # state without the chosen routes, or None
    process_sequentially: bool
    direct_sat_dmd: Any
    shorten_prob: float
    street_node_neighbours: Any
    shortest_paths: Any
    force_linking_unlinked: bool
    adj_condition_target: Any
    adj_condition_weight: Any
    max_n_nodes: int
    cost: Any


def _gather_group_routes(all_networks, ctx: MutationContext, idxs):
    """Legacy per-type gather: pick the group's slots' chosen routes."""
    gather = ctx.chosen_route_idxs[:, idxs, None, None].expand(-1, -1, -1, ctx.max_n_nodes)
    return all_networks[:, idxs].gather(2, gather).squeeze(2)


class HeuristicOperator:
    model_role = None

    @classmethod
    def from_spec(cls, name, spec, policies, bindings):
        if spec.model_ref is not None or spec.allowed_actions or spec.ignore_max_route_len:
            raise ValueError('heuristic bees do not take policies, action masks or neural bounds')
        return cls(name=name)


class HeuristicRebuildOp(HeuristicOperator):
    """Legacy type-1 heuristic: rebuild the chosen route from shortest paths."""

    kind = "shortest_path_rebuild"
    uses_remaining_state = True

    def __init__(self, *, name):
        self.name = name

    def mutate(self, ctx: MutationContext, idxs):
        variants = get_bee_1_variants(
            ctx.remaining_state, ctx.modified_routes, ctx.direct_sat_dmd, ctx.shortest_paths, ctx.force_linking_unlinked
        )
        return variants[:, idxs]


class NeuralRebuildOp:
    """Legacy type-1 neural: full-route rebuild via the construction model."""

    kind = "neural_rebuild"
    uses_remaining_state = False
    model_role = 'construction'

    @classmethod
    def from_spec(cls, name, spec, policies, bindings):
        if spec.model_ref is None:
            raise ValueError('rebuild requires a construction policy')
        model = policies[spec.model_ref]
        if model.action_spec.role != cls.model_role or spec.allowed_actions != ['extend', 'halt']:
            raise ValueError('rebuild requires construction extend/halt actions')
        if spec.ignore_max_route_len:
            raise ValueError('full rebuild respects route bounds')
        return cls(model, name=name, binding=bindings[spec.model_ref])

    def __init__(self, model, *, name,binding):
        self.model = model
        self.binding=binding
        self.name = name

    def mutate(self, ctx: MutationContext, idxs):
        if ctx.process_sequentially:
            return _run_rebuild_variants_for_bees(
                rebuild_route,
                self.model,
                ctx.seq_state,
                ctx.bee_networks,
                ctx.chosen_route_idxs,
                idxs,
                cost=ctx.cost,
                binding=self.binding,
            )
        networks = rebuild_route(self.model, ctx.env_state, ctx.bee_networks, ctx.chosen_route_idxs,
                                 cost=ctx.cost,binding=self.binding)
        return networks[:, idxs, -1]


class ShortenOp(HeuristicOperator):
    """Legacy type-2: heuristic shorten/lengthen of the chosen route."""

    kind = "local_edit"
    uses_remaining_state = False

    def __init__(self, *, name):
        self.name = name

    def mutate(self, ctx: MutationContext, idxs):
        new_routes = get_bee_2_variants(ctx.modified_routes[:, idxs], ctx.shorten_prob, ctx.street_node_neighbours)
        assert ((new_routes > -1).sum(dim=-1) > 0).all()
        return new_routes


class EditOp:
    kind = "neural_edit"
    uses_remaining_state = False
    model_role = 'edit'

    @classmethod
    def from_spec(cls, name, spec, policies, bindings):
        if spec.model_ref is None:
            raise ValueError('edit requires an edit policy')
        model = policies[spec.model_ref]
        if (model.action_spec.role != cls.model_role or not model.supports_trim_actions
                or not spec.allowed_actions
                or not set(spec.allowed_actions) <= {'extend', 'trim_start', 'trim_end', 'halt'}):
            raise ValueError('edit bee has an incompatible policy or action mask')
        return cls(model, name=name, binding=bindings[spec.model_ref],
                   allowed_actions=spec.allowed_actions, ignore_max_route_len=spec.ignore_max_route_len)

    def __init__(self, model, *, name, allowed_actions, ignore_max_route_len,binding):
        self.model = model
        self.binding=binding
        self.name = name
        self.mask = ActionMask.from_names(allowed_actions)
        self.ignore_max_route_len = ignore_max_route_len

    def mutate(self, ctx, idxs):
        kwargs = dict(
            cost=ctx.cost,
            binding=self.binding,
            ignore_max_route_len=self.ignore_max_route_len,
            allow_extend=self.mask.extend,
            allow_trim_start=self.mask.trim_start,
            allow_trim_end=self.mask.trim_end,
            allow_halt=self.mask.halt,
            adj_condition_target=ctx.adj_condition_target,
            adj_condition_weight=ctx.adj_condition_weight,
        )
        if ctx.process_sequentially:
            return _run_selected_route_variants_for_bees(
                edit_route,
                self.model,
                ctx.seq_state,
                ctx.bee_networks,
                ctx.chosen_route_idxs,
                idxs,
                **kwargs,
            )
        networks = edit_route(self.model, ctx.env_state, ctx.bee_networks, ctx.chosen_route_idxs, **kwargs)
        return _gather_group_routes(networks, ctx, idxs)


OPERATOR_FACTORIES = {
    'shortest_path_rebuild': HeuristicRebuildOp,
    'neural_rebuild': NeuralRebuildOp,
    'local_edit': ShortenOp,
    'neural_edit': EditOp,
}
