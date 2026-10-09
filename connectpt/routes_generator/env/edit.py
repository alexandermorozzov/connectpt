"""Single-route editing with network slots and explicit episode limits."""
import torch

from .actions import ActionSpec, RouteAction, ROUTE_ACTION_TRIM_START, ROUTE_ACTION_TRIM_END
from .base import Env
from .masks import ActionMask
from .state import RouteGenBatchState
from ..city.graph import to_batch
from ..network.routes import to_routes
from ..objectives.penalties import ACTION_PENALTIES, edit_step_penalty, forced_halt_penalty


class EditEnv(Env):
    action_spec = ActionSpec("edit")

    def reset(self, graphs, preferences, seed_networks=None, *, route_idx=0, normalize=True, reference_networks=None):
        if seed_networks is None:
            raise ValueError("edit environment requires seed networks")
        routes = to_routes(seed_networks)
        self.network = routes.clone()
        self.route_idx = route_idx
        self.state = RouteGenBatchState(
            to_batch(graphs, edit_features=True),
            self.cost.constants,
            routes.shape[1],
            self.params.min_route_len,
            self.params.max_route_len,
            preferences=preferences,
        )
        routes = routes.to(self.state.device)
        self.state.set_adjustment_seed(routes if reference_networks is None else reference_networks)
        self.state.set_route_slot_context(routes, route_idx)
        context = routes.clone()
        context[:, route_idx] = -1
        self.state.add_new_routes(context, invalid_directly_connected=not bool((context >= 0).any()))
        self.context_route_counts = self.state.n_finished_routes.clone()
        self.state.set_current_routes(routes[:, route_idx])
        if self.feature_set.adjustment_features:
            self.state.set_adjustment_conditioning(
                self.cost.adjustment_degree_target,
                self.cost.penalty_weights["adjustment"],
                routes,
                gap=self.cost.adjustment_degree_gap,
                mode=self.cost.adjustment_degree_mode,
            )
        return self.bind(self.state, normalize=normalize)

    def updated_network(self):
        """Write the planned route back into its original slot."""
        width = max(self.network.shape[-1], self.state.current_routes.shape[-1])
        network = to_routes(self.network, width=width).to(self.state.device)
        for bi, routes in enumerate(self.state.routes):
            route = routes[-1] if len(routes) > int(self.context_route_counts[bi]) else self.state.current_routes[bi]
            route = route[route >= 0]
            if not route.numel():
                route = self.network[bi, self.route_idx]
                route = route[route >= 0]
            network[bi, self.route_idx] = -1
            network[bi, self.route_idx, : route.numel()] = route
        return network

    def bind(self, state, *, normalize=True, single_route=False):
        self.trim_counts = torch.zeros(state.batch_size, device=state.device, dtype=torch.long)
        return super().bind(state, normalize=normalize, single_route=single_route)

    def forced_halt(self):
        limit = self.params.max_route_edit_steps
        return limit is not None and self.steps >= limit

    def action_mask(self, state):
        limit = self.params.max_trim_actions_per_route
        trim = True if limit is None else self.trim_counts < limit
        halt = not (self.params.force_nonhalt_first_step and self.steps == 0)
        if self.forced_halt():
            return ActionMask(False, False, False, True)
        if self.mask_override is not None:
            return self.mask_override
        return ActionMask(True, trim, trim, halt)

    def _apply(self, action: RouteAction):
        active = ~self.is_done()
        trims = ((action.kind == ROUTE_ACTION_TRIM_START) | (action.kind == ROUTE_ACTION_TRIM_END)) & active
        self.trim_counts += trims.long()
        self.state.apply_route_actions(action.kind, action.indices)

    def action_penalties(self, action: RouteAction, active, forced):
        values = {
            "edit_step": edit_step_penalty(action.kind, active, 1.0),
            "forced_halt": forced_halt_penalty(active, forced, 1.0),
        }
        return torch.stack([values[name] for name in ACTION_PENALTIES], dim=-1)
