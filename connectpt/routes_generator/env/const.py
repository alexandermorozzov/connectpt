"""Constants owned by this package."""
from enum import IntEnum


class RouteActionKind(IntEnum):
    EXTEND = 0
    TRIM_START = 1
    TRIM_END = 2
    HALT = 3


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

TORCH_FMAX = 10**10

TORCH_FMIN = -TORCH_FMAX

FEATURE_NAMES = frozenset({
    'adj.current',
    'adj.target',
    'adj.weight',
    'demand',
    'demand.disconnected_fraction',
    'demand.disconnected_log',
    'demand.served_time',
    'graph.demand_edges',
    'graph.mean_demand',
    'graph.node_x_mean',
    'graph.node_x_scaled_mean',
    'graph.node_y_mean',
    'graph.node_y_scaled_mean',
    'graph.std_demand',
    'graph.total_demand',
    'graph.weighted_time_mean',
    'graph.weighted_time_std',
    'legs.context_log',
    'legs.current_log',
    'legs.max_context',
    'legs.mean_excess',
    'legs.redundancy',
    'legs.reused',
    'legs.total_log',
    'network.has_path',
    'network.one_transfer',
    'network.route_time',
    'network.transit_time',
    'network.two_transfers',
    'node.same',
    'node.x',
    'node.x_scaled',
    'node.y',
    'node.y_scaled',
    'path.context_overlap',
    'path.drive_time',
    'path.excess_count',
    'path.excess_time',
    'path.n_stops',
    'path.new_coverage',
    'path.redundancy_delta',
    'pref.ATT',
    'pref.RTT',
    'pref.WMC',
    'route.context_overlap',
    'route.direct',
    'route.direct_time',
    'route.drive_time',
    'route.length_fraction',
    'route.n_stops',
    'route.self_repeat',
    'routes.finished_fraction',
    'routes.finished_log',
    'routes.left_fraction',
    'routes.left_log',
    'street.drive_time',
    'street.exists',
    'street.time',
    'trim.direction',
    'trim.kept_edge_overlap',
    'trim.kept_edge_unique',
    'trim.kept_excess_fraction',
    'trim.kept_node_overlap',
    'trim.kept_node_unique',
    'trim.redundancy_delta',
    'trim.remaining_fraction',
    'trim.remaining_time_fraction',
    'trim.removed_edge_overlap',
    'trim.removed_edge_unique',
    'trim.removed_excess_fraction',
    'trim.removed_excess_time',
    'trim.removed_fraction',
    'trim.removed_node_overlap',
    'trim.removed_node_unique',
    'trim.removed_self_repeat',
    'trim.removed_time_fraction',
})
