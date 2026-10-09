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

    def reset(self, graphs, preferences, seed_networks=None, *, normalize=True):
        self.state = RouteGenBatchState(
            to_batch(graphs),
            self.cost.constants,
            self.params.n_routes,
            self.params.min_route_len,
            self.params.max_route_len,
            preferences=preferences,
        )
        if seed_networks is not None:
            self.state.add_new_routes(to_routes(seed_networks).to(self.state.device))
        return self.bind(self.state, normalize=normalize)

    def action_mask(self, state):
        return self.mask_override if self.mask_override is not None else ActionMask(extend=True, trim_start=False, trim_end=False, halt=True)

    def forced_halt(self):
        return False

    def _apply(self, action: RouteAction):
        self.state.shortest_path_action(action.indices)

    def action_penalties(self, action: RouteAction, active, forced):
        return self.result.goals.new_zeros(active.shape[0], len(ACTION_PENALTIES))
