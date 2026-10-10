"""Batched route construction: extend or halt, no action penalties."""

from .actions import ActionSpec, RouteAction
from .base import Env
from .masks import ActionMask
from .state import RouteGenBatchState
from ..city.graph import to_batch
from ..network.routes import to_routes
from ..objectives.const import ACTION_PENALTIES


class ConstructionEnv(Env):
    action_spec = ActionSpec("construction")

    def reset(self, graphs, preferences, seed_networks=None, *, normalize=True, route_idx=None, reference_networks=None):
        self.state = RouteGenBatchState(
            to_batch(graphs),
            self.cost.constants,
            self.params.n_routes,
            self.params.min_route_len,
            self.params.max_route_len,
            preferences=preferences,
        )
        if seed_networks is not None:
            routes = to_routes(seed_networks).to(self.state.device)
            if route_idx is not None:
                self.state.set_route_slot_context(routes, route_idx)
                routes = self.state.route_slot_context.clone()
                routes[self.state.batch_indices, self.state.active_route_idx] = -1
            self.state.add_new_routes(routes)
        elif route_idx is not None:
            raise ValueError("a selected route requires a seed network")
        if reference_networks is not None:
            self.state.set_adjustment_seed(reference_networks)
        return self.bind(self.state, normalize=normalize)

    def updated_network(self):
        return self.state.adjustment_routes()

    def reset_completed(self, done, preferences=None):
        """Restart completed construction networks and refresh their evaluation."""
        self.state.reset_dones()
        if preferences is not None:
            merged = {key:value.clone() for key,value in self.state.cost_weights.items()}
            for key,value in preferences.items():
                merged[key][done] = value
            self.state.set_cost_weights(merged)
        self.result = self.cost(self.state)

    def action_mask(self, state):
        return self.mask_override if self.mask_override is not None else ActionMask(extend=True, trim_start=False, trim_end=False, halt=True)

    def forced_halt(self):
        return False

    def _apply(self, action: RouteAction):
        self.state.shortest_path_action(action.indices)

    def action_penalties(self, action: RouteAction, active, forced):
        return self.result.goals.new_zeros(active.shape[0], len(ACTION_PENALTIES))
