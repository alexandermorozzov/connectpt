"""Pure observations and checkpoint-locked feature calculations."""
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from ..city.graph import STOP_KEY, DEMAND_KEY
from ..common.tensors import get_update_at_mask
from ..network.routes import get_route_leg_times, to_routes
from ..objectives.cost import EPSILON, _finite_time_diameter


PREFERENCE_KEYS = {"ATT": "demand_time_weight", "RTT": "route_time_weight", "WMC": "median_connectivity_weight"}


def preference_tensor(state, goals):
    return state.get_cost_weights_tensor(tuple(PREFERENCE_KEYS[goal] for goal in goals))


@dataclass(frozen=True)
class Feature:
    name: str
    dim: int = 1

    def compute(self, columns):
        return columns[self.name]


REGISTRY = {
    name: Feature(name)
    for name in (
        "adj.current",
        "adj.target",
        "adj.weight",
        "demand",
        "demand.disconnected_fraction",
        "demand.disconnected_log",
        "demand.served_time",
        "graph.demand_edges",
        "graph.mean_demand",
        "graph.node_x_mean",
        "graph.node_x_scaled_mean",
        "graph.node_y_mean",
        "graph.node_y_scaled_mean",
        "graph.std_demand",
        "graph.total_demand",
        "graph.weighted_time_mean",
        "graph.weighted_time_std",
        "legs.context_log",
        "legs.current_log",
        "legs.max_context",
        "legs.mean_excess",
        "legs.redundancy",
        "legs.reused",
        "legs.total_log",
        "network.has_path",
        "network.one_transfer",
        "network.route_time",
        "network.transit_time",
        "network.two_transfers",
        "node.same",
        "node.x",
        "node.x_scaled",
        "node.y",
        "node.y_scaled",
        "path.context_overlap",
        "path.drive_time",
        "path.excess_count",
        "path.excess_time",
        "path.n_stops",
        "path.new_coverage",
        "path.redundancy_delta",
        "pref.ATT",
        "pref.RTT",
        "pref.WMC",
        "route.context_overlap",
        "route.direct",
        "route.direct_time",
        "route.drive_time",
        "route.length_fraction",
        "route.n_stops",
        "route.self_repeat",
        "routes.finished_fraction",
        "routes.finished_log",
        "routes.left_fraction",
        "routes.left_log",
        "street.drive_time",
        "street.exists",
        "street.time",
        "trim.direction",
        "trim.kept_edge_overlap",
        "trim.kept_edge_unique",
        "trim.kept_excess_fraction",
        "trim.kept_node_overlap",
        "trim.kept_node_unique",
        "trim.redundancy_delta",
        "trim.remaining_fraction",
        "trim.remaining_time_fraction",
        "trim.removed_edge_overlap",
        "trim.removed_edge_unique",
        "trim.removed_excess_fraction",
        "trim.removed_excess_time",
        "trim.removed_fraction",
        "trim.removed_node_overlap",
        "trim.removed_node_unique",
        "trim.removed_self_repeat",
        "trim.removed_time_fraction",
    )
}


GLOBAL_COLUMNS = (
    "network.route_time",
    "routes.finished_log",
    "routes.left_log",
    "routes.finished_fraction",
    "routes.left_fraction",
    "demand.disconnected_log",
    "demand.disconnected_fraction",
    "route.n_stops",
    "demand.served_time",
)
GLOBAL_REDUNDANCY_COLUMNS = (
    "legs.redundancy",
    "legs.mean_excess",
    "legs.max_context",
    "route.self_repeat",
    "route.context_overlap",
)
PAIR_COLUMNS = (
    "demand",
    "route.direct",
    "route.direct_time",
    "street.exists",
    "street.time",
    "network.has_path",
    "network.transit_time",
    "network.one_transfer",
    "network.two_transfers",
    "node.same",
)
PAIR_REDUNDANCY_COLUMNS = ("legs.context_log", "legs.current_log", "legs.total_log", "legs.reused")
NODE_COLUMNS = ("node.x", "node.y", "node.x_scaled", "node.y_scaled")
EXTEND_COLUMNS = ("path.drive_time", "path.n_stops", "route.drive_time", "route.n_stops")
EXTEND_REDUNDANCY_COLUMNS = (
    "path.context_overlap",
    "path.excess_count",
    "path.excess_time",
    "path.new_coverage",
    "path.redundancy_delta",
)
TRIM_COLUMNS = (
    "trim.removed_fraction",
    "trim.remaining_fraction",
    "route.length_fraction",
    "trim.removed_time_fraction",
    "trim.remaining_time_fraction",
    "trim.direction",
    "trim.removed_node_overlap",
    "trim.removed_node_unique",
    "trim.removed_edge_overlap",
    "trim.removed_edge_unique",
    "trim.kept_node_overlap",
    "trim.kept_node_unique",
    "trim.kept_edge_overlap",
    "trim.kept_edge_unique",
)
TRIM_REDUNDANCY_COLUMNS = (
    "trim.removed_excess_fraction",
    "trim.removed_excess_time",
    "trim.removed_self_repeat",
    "trim.kept_excess_fraction",
    "trim.redundancy_delta",
)
GRAPH_COLUMNS = (
    "graph.total_demand",
    "graph.demand_edges",
    "graph.mean_demand",
    "graph.std_demand",
    "graph.weighted_time_mean",
    "graph.weighted_time_std",
    "graph.node_x_mean",
    "graph.node_y_mean",
    "graph.node_x_scaled_mean",
    "graph.node_y_scaled_mean",
)


@dataclass(frozen=True)
class FeatureParams:
    name: str
    node: tuple[str, ...]
    pair: tuple[str, ...]
    global_features: tuple[str, ...]
    extend: tuple[str, ...]
    halt: tuple[str, ...]
    trim: tuple[str, ...]
    critic: tuple[str, ...]

    @classmethod
    def from_values(cls, values):
        return cls(**{k: v if k == "name" else tuple(v) for k, v in values.items()})

    def __post_init__(self):
        for group in (self.node, self.pair, self.global_features, self.extend, self.halt, self.trim, self.critic):
            unknown = set(group) - REGISTRY.keys()
            if unknown:
                raise ValueError(f"unknown features: {sorted(unknown)}")

    @property
    def preference_goals(self):
        return tuple(name[5:] for name in self.global_features if name.startswith("pref."))

    @property
    def redundancy(self):
        return "legs.redundancy" in self.global_features

    @property
    def adjustment_features(self):
        return tuple(name for name in self.global_features if name.startswith("adj."))

    def metadata(self):
        from dataclasses import asdict

        return {
            "feature_set": self.name,
            "feature_names": asdict(self),
            "dims": dims(self),
            "objectives": list(self.preference_goals),
        }


def dims(feature_set):
    groups = {
        name: sum(REGISTRY[col].dim for col in getattr(feature_set, name))
        for name in ("node", "pair", "global_features", "extend", "halt", "trim", "critic")
    }
    groups["global"] = groups.pop("global_features")
    for name in ("extend", "halt", "trim"):
        groups[name] += groups["global"]
    groups["extend"] += 1  # learned base_path_score belongs to the policy
    return groups


def select_features(values, source_names, requested_names):
    """The YAML order, including permutations, determines the tensor columns."""
    columns = dict(zip(source_names, values.unbind(-1), strict=True))
    return torch.stack([REGISTRY[name].compute(columns) for name in requested_names], dim=-1)


@dataclass(frozen=True)
class Observation:
    feature_set: FeatureParams
    graph_data: object
    norm_node_features: Tensor
    global_features: Tensor
    pair_features: Tensor
    critic: Tensor
    paths: Tensor
    done: Tensor
    drive_times: Tensor
    n_nodes: Tensor
    batch_indices: Tensor
    batch_size: int
    max_n_nodes: int
    device: torch.device
    symmetric_routes: bool
    mean_stop_time: Tensor
    current_routes: Tensor
    current_route_n_stops: Tensor
    current_route_time: Tensor
    current_route_times_from_start: Tensor
    min_route_len: Tensor
    max_route_len: Tensor
    valid_terms_mat: Tensor
    has_path: Tensor
    context_leg_use_count: Tensor
    total_leg_use_count: Tensor
    trim_start: tuple
    trim_end: tuple

    def is_done(self):
        return self.done

    def get_shortest_path_sequences(self):
        return self.paths


def observe(state, feature_set, *, allow_below_min=False):
    if isinstance(state, Observation):
        if state.feature_set != feature_set:
            raise ValueError("observation feature set differs from policy")
        return state
    glob = global_state_features(
        state,
        include_redundancy=feature_set.redundancy,
        preference_goals=feature_set.preference_goals,
        adjustment_features=feature_set.adjustment_features,
    )
    pref_names = tuple(f"pref.{name}" for name in feature_set.preference_goals)
    global_names = (
        pref_names
        + GLOBAL_COLUMNS
        + (GLOBAL_REDUNDANCY_COLUMNS if feature_set.redundancy else ())
        + feature_set.adjustment_features
    )
    glob = select_features(glob, global_names, feature_set.global_features)
    pair = pair_features(state, use_redundancy=feature_set.redundancy, preference_goals=feature_set.preference_goals)
    if feature_set.pair[-1] != "street.drive_time":
        raise ValueError("street.drive_time must be the last pair feature for separate normalization")
    pair_names = PAIR_COLUMNS + (PAIR_REDUNDANCY_COLUMNS if feature_set.redundancy else ()) + pref_names
    pair = select_features(pair, pair_names, feature_set.pair[:-1])
    critic = critic_inputs(
        state.graph_data,
        state.cost_weights,
        input_dim=len(GRAPH_COLUMNS) + glob.shape[-1],
        holliday_features=feature_set.name == "construction_holliday",
        weight_input_keys=tuple(key for goal, key in PREFERENCE_KEYS.items() if goal in feature_set.preference_goals),
    )
    critic[..., -glob.shape[-1] :] = glob
    critic = select_features(critic, GRAPH_COLUMNS + feature_set.global_features, feature_set.critic)
    from ..city.shortest_paths import reconstruct_all_paths

    paths = state.extra_data.shortest_path_sequences
    if paths.numel() == 0:
        paths, _ = reconstruct_all_paths(state.nexts)
    trims = tuple(
        trim_candidate_features(
            state,
            glob,
            torch.float32,
            direction,
            use_redundancy=feature_set.redundancy,
            allow_below_min=allow_below_min,
        )
        for direction in (True, False)
    )
    trim_names = (
        TRIM_COLUMNS + (TRIM_REDUNDANCY_COLUMNS if feature_set.redundancy else ()) + feature_set.global_features
    )
    trims = tuple(
        (select_features(values[0], trim_names, feature_set.trim + feature_set.global_features), *values[1:])
        for values in trims
    )
    fields = (
        "graph_data",
        "norm_node_features",
        "drive_times",
        "n_nodes",
        "batch_indices",
        "batch_size",
        "max_n_nodes",
        "device",
        "symmetric_routes",
        "mean_stop_time",
        "current_routes",
        "current_route_n_stops",
        "current_route_time",
        "current_route_times_from_start",
        "min_route_len",
        "max_route_len",
        "valid_terms_mat",
        "has_path",
        "context_leg_use_count",
        "total_leg_use_count",
    )
    values = {name: getattr(state, name) for name in fields}
    return Observation(
        feature_set,
        global_features=glob,
        pair_features=pair,
        critic=critic,
        paths=paths,
        done=state.is_done(),
        trim_start=trims[0],
        trim_end=trims[1],
        **values,
    )


def global_state_features(state, include_redundancy=None, *, preference_goals, adjustment_features=()):
    """Global columns in historical checkpoint order, without mutation."""
    if isinstance(state, Observation):
        return state.global_features
    cost_weights = preference_tensor(state, preference_goals)
    diameter = _finite_time_diameter(state.drive_times)
    mean_route_time = state.total_route_time / (state.n_routes_to_plan * diameter)

    so_far = state.n_finished_routes
    left = state.n_routes_left_to_plan
    both = torch.stack((so_far, left), dim=-1)
    n_routes_log_feats = (both + 1).log()
    # use fractions so it's independent of n_routes_to_plan
    n_routes_frac_feats = both / (so_far + left)[:, None]

    n_disconnected_demand_edges = state.get_n_disconnected_demand_edges()
    # as with n_routes feats, use both log and fractional
    log_uncovered = (n_disconnected_demand_edges + 1).log()
    frac_uncovered = n_disconnected_demand_edges / state.n_demand_edges
    uncovered_feats = torch.stack((log_uncovered, frac_uncovered), dim=-1)
    curr_route_n_stops = state.current_route_n_stops[:, None]

    served_demand = (state.has_path * state.demand).sum(dim=(1, 2))
    tt = state.transit_times.clone()
    tt[~state.has_path] = 0
    total_demand_time = (state.demand * tt).sum(dim=(1, 2))
    mean_demand_time = total_demand_time / (served_demand + EPSILON)
    mean_demand_time_frac = mean_demand_time / diameter

    global_features = torch.cat(
        (
            cost_weights,
            mean_route_time[:, None],
            n_routes_log_feats,
            n_routes_frac_feats,
            uncovered_feats,
            curr_route_n_stops,
            mean_demand_time_frac[:, None],
        ),
        dim=-1,
    )

    if include_redundancy is None:
        include_redundancy = state._redundancy_features_enabled
    if include_redundancy:
        redundancy_features = redundancy_global_features(state)
        global_features = torch.cat((global_features, redundancy_features), dim=-1)

    if "adj.target" in adjustment_features:
        feats = [state.adjustment_target.reshape(-1).clamp(min=0.0)]
        if "adj.weight" in adjustment_features:
            feats.append(torch.log1p(state.adjustment_weight.reshape(-1).clamp(min=0.0)) / math.log1p(8.0))
        if "adj.current" in adjustment_features:
            feats.append(current_adjustment(state))
        global_features = torch.cat((global_features, torch.stack(feats, dim=-1)), dim=-1)
    return global_features


def redundancy_global_features(state):
    """Return normalized network and active-route redundancy summaries."""
    context_counts = state.context_leg_use_count.to(dtype=torch.float32)
    current_counts = state.current_leg_use_count.to(dtype=torch.float32)
    total_counts = context_counts + current_counts
    excess_counts = (total_counts - 1).clamp_min(0)

    total_traversals = total_counts.sum(dim=(1, 2))
    network_redundancy = excess_counts.sum(dim=(1, 2)) / total_traversals.clamp_min(1)

    n_used_legs = (total_counts > 0).sum(dim=(1, 2))
    mean_excess_leg_use = excess_counts.sum(dim=(1, 2)) / n_used_legs.clamp_min(1)
    max_context_leg_use = context_counts.amax(dim=(1, 2))

    current_traversals = current_counts.sum(dim=(1, 2))
    current_route_state_repeat = (current_counts - 1).clamp_min(0).sum(dim=(1, 2)) / current_traversals.clamp_min(1)
    current_route_context_overlap = (current_counts * (context_counts > 0)).sum(
        dim=(1, 2)
    ) / current_traversals.clamp_min(1)

    return torch.stack(
        (
            network_redundancy,
            mean_excess_leg_use,
            max_context_leg_use,
            current_route_state_repeat,
            current_route_context_overlap,
        ),
        dim=-1,
    )


def current_adjustment(state):
    """Per-graph adjustment degree of the CURRENT network vs the stored
    seed routes ([batch] in [0,1]). Used as a closed-loop conditioning
    feature."""
    from ..objectives.penalties import adjustment_degree

    if state.route_slot_context.numel() > 0:
        cur = state.route_slot_context.clone()
        for bi, route_idx in enumerate(state.active_route_idx.tolist()):
            if route_idx < 0:
                continue
            cur[bi, route_idx] = -1
            route = state.current_routes[bi]
            copy_len = min(cur.shape[-1], route.shape[-1])
            cur[bi, route_idx, :copy_len] = route[:copy_len]
    else:
        cur = to_routes(state.routes).to(state.device)
    seed = state.adjustment_seed.to(state.device)
    if seed.dim() == 2:  # unbatched [n_routes, L]
        seed = seed.unsqueeze(0)
    nr = min(cur.shape[1], seed.shape[1])
    ll = min(cur.shape[-1], seed.shape[-1])
    adj = adjustment_degree(
        cur[:, :nr, :ll],
        seed[:, :nr, :ll].long(),
        state.symmetric_routes,
        gap=state._adjustment_gap,
        mode=state._adjustment_mode,
    )
    # adjustment_degree returns per-route [batch, n_routes]; the
    # feature is the per-graph mean adjustment.
    return adj.reshape(adj.shape[0], -1).mean(dim=1).float()


def added_redundancy_features(state, candidate_paths, dtype):
    """Describe how each candidate shortest path changes leg reuse."""
    paths = candidate_paths.to(device=state.device, dtype=torch.long)
    from_nodes = paths[..., :-1]
    to_nodes = paths[..., 1:]
    valid = (from_nodes >= 0) & (to_nodes >= 0)
    safe_from = from_nodes.clamp(min=0)
    safe_to = to_nodes.clamp(min=0)

    batch_idxs = torch.arange(state.batch_size, device=state.device)
    for _ in range(from_nodes.ndim - 1):
        batch_idxs = batch_idxs.unsqueeze(-1)
    batch_idxs = batch_idxs.expand_as(from_nodes)

    context_counts = state.context_leg_use_count
    total_counts = state.total_leg_use_count
    path_context_counts = context_counts[batch_idxs, safe_from, safe_to]
    path_total_counts = total_counts[batch_idxs, safe_from, safe_to]
    adds_context_overlap = valid & (path_context_counts > 0)
    adds_excess = valid & (path_total_counts > 0)
    adds_new_coverage = valid & (path_total_counts == 0)

    valid_counts = valid.sum(dim=-1)
    denom = valid_counts.clamp_min(1).to(dtype=dtype)
    added_context_overlap_fraction = adds_context_overlap.to(dtype=dtype).sum(dim=-1) / denom
    added_excess_count = adds_excess.to(dtype=dtype).sum(dim=-1)
    added_new_coverage_fraction = adds_new_coverage.to(dtype=dtype).sum(dim=-1) / denom

    leg_times = state.drive_times[batch_idxs, safe_from, safe_to]
    mean_stop_time = state.mean_stop_time
    for _ in range(leg_times.ndim - 1):
        mean_stop_time = mean_stop_time.unsqueeze(-1)
    leg_times = (leg_times + mean_stop_time) * valid
    path_times = leg_times.sum(dim=-1)
    excess_times = (leg_times * adds_excess).sum(dim=-1)
    added_excess_time_fraction = excess_times / path_times.clamp_min(EPSILON)

    mirror_factor = 2 if state.symmetric_routes else 1
    all_total_counts = total_counts.to(dtype=dtype)
    old_traversals = all_total_counts.sum(dim=(1, 2)) / mirror_factor
    old_excess = (all_total_counts - 1).clamp_min(0).sum(dim=(1, 2)) / mirror_factor
    old_redundancy = old_excess / old_traversals.clamp_min(1)
    for _ in range(valid_counts.ndim - 1):
        old_traversals = old_traversals.unsqueeze(-1)
        old_excess = old_excess.unsqueeze(-1)
        old_redundancy = old_redundancy.unsqueeze(-1)
    new_redundancy = (old_excess + added_excess_count) / (old_traversals + valid_counts).clamp_min(1)
    redundancy_delta_if_extend = new_redundancy - old_redundancy

    return torch.stack(
        (
            added_context_overlap_fraction,
            added_excess_count,
            added_excess_time_fraction,
            added_new_coverage_fraction,
            redundancy_delta_if_extend,
        ),
        dim=-1,
    )


def trim_candidate_features(state, global_features, dtype, trim_start, *, use_redundancy, allow_below_min):
    if isinstance(state, Observation):
        values = state.trim_start if trim_start else state.trim_end
        return values
    routes = state.current_routes.to(dtype=torch.long)
    batch_size, route_capacity = routes.shape
    route_lens = state.current_route_n_stops.to(dtype=torch.long)
    positions = torch.arange(route_capacity, device=state.device)
    pos = positions[None].expand(batch_size, -1)

    min_lens = state.min_route_len.to(device=state.device, dtype=torch.long)
    # Absolute trim floor: normally min_route_len, but if below-min trims
    # are allowed a route may be shortened down to 2 stops (still a path).
    trim_floor = torch.full_like(min_lens, 2) if allow_below_min else min_lens
    max_lens = state.max_route_len.to(device=state.device, dtype=dtype).clamp_min(1)
    route_lens_f = route_lens.to(dtype=dtype).clamp_min(1)
    route_time = state.current_route_time.to(dtype=dtype).clamp_min(EPSILON)
    times_from_start = state.current_route_times_from_start.to(dtype=dtype)

    if trim_start:
        from_nodes = routes[:, :1].expand(-1, route_capacity)
        to_nodes = routes
        removed_len = pos
        remaining_len = route_lens[:, None] - pos
        removed_time = times_from_start
        remaining_time = route_time[:, None] - removed_time
        direction_flag = torch.zeros_like(route_lens_f)[:, None].expand_as(times_from_start)
        valid = (pos >= 1) & (pos < route_lens[:, None]) & (remaining_len >= trim_floor[:, None])
    else:
        last_pos = (route_lens - 1).clamp_min(0)
        old_end = routes.gather(1, last_pos[:, None])
        from_nodes = routes
        to_nodes = old_end.expand(-1, route_capacity)
        remaining_len = pos + 1
        removed_len = route_lens[:, None] - remaining_len
        remaining_time = times_from_start
        removed_time = route_time[:, None] - remaining_time
        direction_flag = torch.ones_like(route_lens_f)[:, None].expand_as(times_from_start)
        valid = (pos < (route_lens - 1)[:, None]) & (remaining_len >= trim_floor[:, None])

    valid = valid & (route_lens[:, None] > trim_floor[:, None])
    route_len_frac = (route_lens.to(dtype=dtype)[:, None] / max_lens[:, None]).expand_as(pos)
    base_features = torch.stack(
        (
            removed_len.to(dtype=dtype) / route_lens_f[:, None],
            remaining_len.to(dtype=dtype) / max_lens[:, None],
            route_len_frac,
            removed_time / route_time[:, None],
            remaining_time / route_time[:, None],
            direction_flag.to(dtype=dtype),
        ),
        dim=-1,
    )
    overlap_features = trim_overlap_features(state, positions, trim_start, dtype)
    if use_redundancy:
        redundancy_features = trim_redundancy_features(state, positions, trim_start, dtype)
    else:
        redundancy_features = torch.zeros(
            (batch_size, route_capacity, 0),
            device=state.device,
            dtype=dtype,
        )
    expanded_global = global_features[:, None].expand(-1, route_capacity, -1)
    features = torch.cat((base_features, overlap_features, redundancy_features, expanded_global), dim=-1)
    return features, valid, from_nodes, to_nodes


def trim_overlap_features(state, positions, trim_start, dtype):
    routes = state.current_routes.to(dtype=torch.long)
    batch_size, route_capacity = routes.shape
    route_lens = state.current_route_n_stops.to(dtype=torch.long)
    node_pos = positions[None, None]
    cand_pos = positions[None, :, None]
    route_valid = routes >= 0

    other_node_masks, other_edge_masks = other_route_overlap_masks(state)
    safe_routes = routes.clamp(min=0)
    covered_route_nodes = other_node_masks.gather(1, safe_routes) & route_valid

    if trim_start:
        removed_node_pos = node_pos < cand_pos
        kept_node_pos = (node_pos >= cand_pos) & (node_pos < route_lens[:, None, None])
    else:
        removed_node_pos = (node_pos > cand_pos) & (node_pos < route_lens[:, None, None])
        kept_node_pos = node_pos <= cand_pos
    removed_node_pos = removed_node_pos & route_valid[:, None]
    kept_node_pos = kept_node_pos & route_valid[:, None]

    removed_node_overlap = masked_candidate_fraction(covered_route_nodes, removed_node_pos, dtype)
    kept_node_overlap = masked_candidate_fraction(covered_route_nodes, kept_node_pos, dtype)

    if route_capacity > 1:
        edge_pos = positions[:-1][None, None]
        from_nodes = routes[..., :-1]
        to_nodes = routes[..., 1:]
        valid_edges = (from_nodes >= 0) & (to_nodes >= 0)
        edge_batch_idxs = torch.arange(batch_size, device=state.device)[:, None]
        safe_from = from_nodes.clamp(min=0)
        safe_to = to_nodes.clamp(min=0)
        covered_route_edges = other_edge_masks[edge_batch_idxs, safe_from, safe_to] & valid_edges

        if trim_start:
            removed_edge_pos = edge_pos < cand_pos
            kept_edge_pos = (edge_pos >= cand_pos) & (edge_pos < (route_lens - 1)[:, None, None])
        else:
            removed_edge_pos = (edge_pos >= cand_pos) & (edge_pos < (route_lens - 1)[:, None, None])
            kept_edge_pos = edge_pos < cand_pos
        removed_edge_pos = removed_edge_pos & valid_edges[:, None]
        kept_edge_pos = kept_edge_pos & valid_edges[:, None]
        removed_edge_overlap = masked_candidate_fraction(covered_route_edges, removed_edge_pos, dtype)
        kept_edge_overlap = masked_candidate_fraction(covered_route_edges, kept_edge_pos, dtype)
    else:
        empty = torch.zeros((batch_size, route_capacity), device=state.device, dtype=dtype)
        removed_edge_overlap = empty
        kept_edge_overlap = empty

    one = torch.ones((), device=state.device, dtype=dtype)
    return torch.stack(
        (
            removed_node_overlap,
            one - removed_node_overlap,
            removed_edge_overlap,
            one - removed_edge_overlap,
            kept_node_overlap,
            one - kept_node_overlap,
            kept_edge_overlap,
            one - kept_edge_overlap,
        ),
        dim=-1,
    )


def masked_candidate_fraction(covered_items, candidate_mask, dtype):
    covered = covered_items[:, None] & candidate_mask
    counts = candidate_mask.sum(dim=-1)
    denom = counts.clamp_min(1).to(dtype=dtype)
    frac = covered.to(dtype=dtype).sum(dim=-1) / denom
    return torch.where(
        counts > 0,
        frac,
        torch.zeros_like(frac),
    )


def trim_redundancy_features(state, positions, trim_start, dtype):
    """Describe which repeated legs each possible trim action removes."""
    routes = state.current_routes.to(dtype=torch.long)
    batch_size, route_capacity = routes.shape
    route_lens = state.current_route_n_stops.to(dtype=torch.long)
    edge_positions = positions[:-1]
    edge_pos = edge_positions[None, None]
    cand_pos = positions[None, :, None]

    from_nodes = routes[..., :-1]
    to_nodes = routes[..., 1:]
    valid_edges = (from_nodes >= 0) & (to_nodes >= 0)
    safe_from = from_nodes.clamp(min=0)
    safe_to = to_nodes.clamp(min=0)
    edge_batch_idxs = torch.arange(batch_size, device=state.device)[:, None]

    total_edge_counts = state.total_leg_use_count[edge_batch_idxs, safe_from, safe_to]
    current_edge_counts = state.current_leg_use_count[edge_batch_idxs, safe_from, safe_to]
    excess_edges = valid_edges & (total_edge_counts > 1)
    self_repeat_edges = valid_edges & (current_edge_counts > 1)

    if trim_start:
        removed_edge_pos = edge_pos < cand_pos
        kept_edge_pos = (edge_pos >= cand_pos) & (edge_pos < (route_lens - 1)[:, None, None])
    else:
        removed_edge_pos = (edge_pos >= cand_pos) & (edge_pos < (route_lens - 1)[:, None, None])
        kept_edge_pos = edge_pos < cand_pos
    removed_edge_pos = removed_edge_pos & valid_edges[:, None]
    kept_edge_pos = kept_edge_pos & valid_edges[:, None]

    removed_excess_fraction = masked_candidate_fraction(excess_edges, removed_edge_pos, dtype)
    removed_self_repeat_fraction = masked_candidate_fraction(self_repeat_edges, removed_edge_pos, dtype)
    kept_excess_fraction = masked_candidate_fraction(excess_edges, kept_edge_pos, dtype)

    leg_times = get_route_leg_times(state.current_routes, state.drive_times, state.mean_stop_time)
    redundant_leg_times = leg_times * excess_edges
    removed_excess_time = (redundant_leg_times[:, None] * removed_edge_pos.to(dtype=leg_times.dtype)).sum(dim=-1)
    removed_excess_time_fraction = removed_excess_time / leg_times.sum(dim=-1)[:, None].clamp_min(EPSILON)

    mirror_factor = 2 if state.symmetric_routes else 1
    total_counts = state.total_leg_use_count.to(dtype=dtype)
    old_traversals = total_counts.sum(dim=(1, 2)) / mirror_factor
    old_excess = (total_counts - 1).clamp_min(0).sum(dim=(1, 2)) / mirror_factor
    old_redundancy = old_excess / old_traversals.clamp_min(1)

    if state.symmetric_routes:
        compare_from = torch.minimum(safe_from, safe_to)
        compare_to = torch.maximum(safe_from, safe_to)
    else:
        compare_from = safe_from
        compare_to = safe_to
    same_edge = (compare_from[:, :, None] == compare_from[:, None]) & (compare_to[:, :, None] == compare_to[:, None])
    earlier_edge = edge_positions[None, :, None] > edge_positions[None, None]
    prior_removed_same = (removed_edge_pos[:, :, :, None] & same_edge[:, None] & earlier_edge[:, None]).sum(dim=-1)
    count_before_removal = total_edge_counts[:, None] - prior_removed_same
    removed_excess = (removed_edge_pos & (count_before_removal > 1)).sum(dim=-1).to(dtype=dtype)
    removed_count = removed_edge_pos.sum(dim=-1).to(dtype=dtype)
    new_redundancy = (old_excess[:, None] - removed_excess) / (old_traversals[:, None] - removed_count).clamp_min(1)
    redundancy_delta_if_trim = old_redundancy[:, None] - new_redundancy

    return torch.stack(
        (
            removed_excess_fraction,
            removed_excess_time_fraction,
            removed_self_repeat_fraction,
            kept_excess_fraction,
            redundancy_delta_if_trim,
        ),
        dim=-1,
    )


def other_route_overlap_masks(state):
    if hasattr(state.extra_data, "context_node_covered_mask") and hasattr(
        state.extra_data, "context_edge_covered_mask"
    ):
        return (
            state.context_node_covered_mask.to(device=state.device, dtype=torch.bool),
            state.context_edge_covered_mask.to(device=state.device, dtype=torch.bool),
        )
    node_masks = torch.zeros(
        (state.batch_size, state.max_n_nodes),
        dtype=torch.bool,
        device=state.device,
    )
    edge_masks = torch.zeros(
        (state.batch_size, state.max_n_nodes, state.max_n_nodes),
        dtype=torch.bool,
        device=state.device,
    )

    for bi, batch_routes in enumerate(state._finished_routes):
        for route in batch_routes:
            route = route.to(device=state.device, dtype=torch.long)
            route = route[route > -1]
            if route.numel() == 0:
                continue
            node_masks[bi, route] = True
            if route.numel() < 2:
                continue
            from_nodes = route[:-1]
            to_nodes = route[1:]
            edge_masks[bi, from_nodes, to_nodes] = True
            if state.symmetric_routes:
                edge_masks[bi, to_nodes, from_nodes] = True

    return node_masks, edge_masks


def pair_features(state, *, use_redundancy, preference_goals):
    if isinstance(state, Observation):
        return state.pair_features
    # edge feats: demand, drive time, has route, route time, has street,
    # street time, has 1-transfer path, has 2-transfer path, has any path,
    # shortest transit path time, is same node
    edge_feature_parts = [state.demand]
    # has a direct route
    has_direct_route = state.route_mat.isfinite()
    edge_feature_parts.append(has_direct_route)
    # time on direct route
    finite_direct_times = get_update_at_mask(state.route_mat, ~has_direct_route)
    edge_feature_parts.append(finite_direct_times)
    # has street
    has_street = state.graph_data.street_adj.isfinite()
    edge_feature_parts.append(has_street)
    # street time
    street_times = get_update_at_mask(state.graph_data.street_adj, ~has_street)
    edge_feature_parts.append(street_times)
    # has 1-transfer path, 2-transfer path, any path, shortest transit
    # compute route shortest path lengths
    edge_feature_parts.append(state.has_path)
    # times of existing paths
    transit_times = state.transit_times.clone()
    transit_times[~state.has_path] = 0
    edge_feature_parts.append(transit_times)
    is_direct_path = state.directly_connected
    flt_is_direct_path = is_direct_path.to(torch.float32)
    flt_upto_1trnsfr_path = flt_is_direct_path.bmm(flt_is_direct_path)
    flt_upto_2trnsfr_path = flt_upto_1trnsfr_path.bmm(flt_is_direct_path)
    upto_1trnsfr_path = flt_upto_1trnsfr_path.bool()
    is_1trnsfr_path = upto_1trnsfr_path ^ is_direct_path
    edge_feature_parts.append(is_1trnsfr_path)
    upto_2trnsfr_path = flt_upto_2trnsfr_path.bool()
    is_2trnsfr_path = upto_2trnsfr_path ^ upto_1trnsfr_path
    edge_feature_parts.append(is_2trnsfr_path)

    # is same node
    eye = torch.eye(state.max_n_nodes, device=state.device)
    eye = eye.expand(state.batch_size, -1, -1)
    edge_feature_parts.append(eye)
    if use_redundancy:
        context_counts = state.context_leg_use_count.to(torch.float32)
        current_counts = state.current_leg_use_count.to(torch.float32)
        total_counts = state.total_leg_use_count.to(torch.float32)
        edge_feature_parts.extend(
            (
                torch.log1p(context_counts),
                torch.log1p(current_counts),
                torch.log1p(total_counts),
                total_counts > 1,
            )
        )
    edge_features = torch.stack(edge_feature_parts, dim=-1)

    # add planes with cost weights (named dict -> vector via the model's
    # declared key spec)
    cost_weight_planes = preference_tensor(state, preference_goals)[:, None, None, :]
    cost_weight_planes = cost_weight_planes.expand(-1, state.max_n_nodes, state.max_n_nodes, -1)
    edge_features = torch.cat((edge_features, cost_weight_planes), dim=-1)
    return edge_features


def critic_inputs(graph_data, cost_weights, *, input_dim, holliday_features, weight_input_keys):
    dev = graph_data[STOP_KEY].x.device
    input_data = torch.zeros(graph_data.num_graphs, input_dim, dtype=torch.float, device=dev)
    input_data[:, 0] = graph_data.demand.sum(dim=(1, 2))

    for bi, dl in enumerate(graph_data.to_data_list()):
        dmd_graph = dl[DEMAND_KEY]
        input_data[bi, 1] = dmd_graph.num_edges
        input_data[bi, 2] = dmd_graph.edge_attr[:, 0].mean()
        if dmd_graph.num_edges > 1:
            input_data[bi, 3] = dmd_graph.edge_attr[:, 0].std()
        else:
            input_data[bi, 3] = 0

        dmd_weighted_times = dmd_graph.edge_attr[:, 0] * dmd_graph.edge_attr[:, 1]
        input_data[bi, 4] = dmd_weighted_times.mean()
        if holliday_features:
            input_data[bi, 5] = dmd_weighted_times.mean()
        elif dmd_graph.num_edges > 1:
            input_data[bi, 5] = dmd_weighted_times.std()
        else:
            input_data[bi, 5] = 0
        x_dim = dl[STOP_KEY].x.shape[1]
        input_data[bi, 6 : 6 + x_dim] = dl[STOP_KEY].x.mean(dim=0)

    # Named dict -> fixed slots by the explicit key spec (never dict
    # iteration order). A key absent from the dict leaves its slot at
    # zero.
    for ii, key in enumerate(weight_input_keys):
        if key not in cost_weights:
            continue
        cw = cost_weights[key]
        data_idx = -(1 + ii)
        if torch.is_tensor(cw):
            input_data[:, data_idx] = cw.to(dev)
        else:
            input_data[:, data_idx] = cw

    return input_data


def aggr_edges_over_sequences(seqs_tensor, edge_features, agg_mode="sum", sparse=False, low_memory_mode=False):
    """
    Assumes a batch dim exists.  Note that in "concat" mode, a mask is returned
        along with the feature sequence.  The mask is True where the sequence
        is invalid.  In general, the valid elements of the sequence will *not*
        be contiguous!

    sparse: if True, aggregate edges sparsely.  Otherwise, do dense
        aggregation.
    low_memory_mode: if True, loop over batch elements in some cases instead of
        dealing with them all at once.

    """
    dev = edge_features.device
    max_seq_len = seqs_tensor.shape[-1]
    # augment the feature matrix with a fringe of zeros so indexing to -1 won't
    # affect the sum.
    # aug_features = torch.torch.nn.functional.pad(edge_features, (0, 0, 0, 1, 0, 1,))
    aug_shape = list(edge_features.shape)
    aug_shape[-3] += 1
    aug_shape[-2] += 1
    aug_features = torch.zeros(aug_shape, device=dev)
    aug_features[:, :-1, :-1] = edge_features

    # It's possible to do this without a loop, but it takes huge amounts of
    # memory, so don't.
    aggs_shape = seqs_tensor.shape[:-1] + (edge_features.shape[-1],)
    if agg_mode == "concat":
        # add a sequence dimension
        aggs_shape = aggs_shape[:-1] + (0, aggs_shape[-1])

    aggs = torch.zeros(aggs_shape, dtype=torch.float32, device=dev)
    if agg_mode == "mean":
        edge_counts = torch.zeros(aggs.shape[:3], dtype=int, device=dev)
    if agg_mode == "concat":
        mask = torch.zeros(aggs_shape[:-1], dtype=bool, device=dev)

    batch_size = seqs_tensor.shape[0]
    batch_idxs = torch.arange(batch_size, device=dev)
    # expand the batch indexes to match any extra "batch" dims in seqs_tensor
    n_extra_batch_dims = seqs_tensor.ndim - 2
    for _ in range(n_extra_batch_dims):
        # add a new dimension after the main batch dimension
        batch_idxs = batch_idxs[:, None]
    batch_idxs = batch_idxs.expand(seqs_tensor.shape[:-1])

    batch_idxs = batch_idxs[..., None]
    for ii in range(max_seq_len - 1):
        from_idxs = seqs_tensor[..., ii][..., None]
        if sparse:
            to_idxs = seqs_tensor[..., ii + 1][..., None]
        else:
            to_idxs = seqs_tensor[..., ii + 1 :]

        if low_memory_mode and agg_mode in ["mean", "sum"]:
            for bi in range(aug_features.shape[0]):
                out_edges = aug_features[bi, from_idxs[bi], to_idxs[bi]]
                aggs[bi] += out_edges.sum(dim=-2)

        else:
            out_edges = aug_features[batch_idxs, from_idxs, to_idxs]
            if agg_mode == "concat":
                aggs = torch.cat((aggs, out_edges), dim=-2)
                mask = torch.cat((mask, to_idxs == -1), dim=-1)
            else:
                aggs += out_edges.sum(dim=-2)

        if agg_mode == "mean":
            edge_counts += (to_idxs > -1).sum(dim=-1)

    if agg_mode == "mean":
        # avoid division by 0 by making 0s into 1s
        denoms = edge_counts + (edge_counts == 0)
        aggs /= denoms[..., None]

    if agg_mode == "concat":
        return aggs, mask

    return aggs


def finite_time_features(state, times):
    """Replace unreachable travel times with a finite graph penalty."""
    penalty = 2.0 * _finite_time_diameter(state.drive_times)
    penalty = penalty.to(device=times.device, dtype=times.dtype)
    while penalty.ndim < times.ndim:
        penalty = penalty.unsqueeze(-1)
    return torch.where(torch.isfinite(times), times, penalty)


def halt_features(obs, route_time, route_len):
    route = torch.stack((route_time, route_len), dim=1)
    route = select_features(route, ("route.drive_time", "route.n_stops"), obs.feature_set.halt)
    return torch.cat((route, obs.global_features), dim=1)


def extend_features(
    state, base_path_scores, new_path_lens, new_drive_times, prev_len=None, prev_drive_time=None, candidate_paths=None
):
    if prev_len is None:
        prev_len = torch.zeros_like(new_path_lens)
    else:
        # add a dimension to match new_path_lens
        prev_len = prev_len[:, None]
        prev_len = prev_len.expand(-1, new_path_lens.shape[-1])
    if prev_drive_time is None:
        prev_drive_time = torch.zeros_like(new_drive_times)
    else:
        # add a dimension to match new_drive_times
        prev_drive_time = prev_drive_time[:, None]
        prev_drive_time = prev_drive_time.expand(-1, new_drive_times.shape[-1])
    new_drive_times = finite_time_features(state, new_drive_times)
    prev_drive_time = finite_time_features(state, prev_drive_time)

    update_in = torch.stack((base_path_scores, new_drive_times, new_path_lens, prev_drive_time, prev_len), dim=-1)

    if state.feature_set.redundancy:
        if candidate_paths is None:
            raise ValueError("Redundancy-aware path scoring requires candidate paths")
        redundancy_feats = added_redundancy_features(state, candidate_paths, update_in.dtype)
        update_in = torch.cat((update_in, redundancy_feats), dim=-1)

    source = EXTEND_COLUMNS + (EXTEND_REDUNDANCY_COLUMNS if state.feature_set.redundancy else ())
    raw_features = select_features(update_in[..., 1:], source, state.feature_set.extend)
    update_in = torch.cat((update_in[..., :1], raw_features), dim=-1)

    global_feat = global_state_features(
        state, include_redundancy=state.feature_set.redundancy, preference_goals=state.feature_set.preference_goals
    )
    update_in = torch.nn.functional.pad(update_in, (0, global_feat.shape[-1]))
    for _ in range(update_in.ndim - global_feat.ndim):
        global_feat = global_feat.unsqueeze(1)

    update_in[..., -global_feat.shape[-1] :] = global_feat

    return update_in
