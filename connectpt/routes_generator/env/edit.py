"""Single-route editing with network slots and explicit episode limits."""
import torch

from .actions import ActionSpec, RouteAction, ROUTE_ACTION_HALT, ROUTE_ACTION_TRIM_START, ROUTE_ACTION_TRIM_END
from .construction import ConstructionEnv
from .masks import ActionMask
from .state import RouteGenBatchState
from ..city.graph import to_batch
from ..network.routes import to_routes
from ..objectives.penalties import edit_step_penalty, forced_halt_penalty


class EditEnv(ConstructionEnv):
    action_spec = ActionSpec("edit")

    def reset(self, graphs, preferences, seed_networks=None, *, route_idx=0):
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
        self.state.set_adjustment_seed(routes)
        self.state.set_route_slot_context(routes, route_idx)
        context = routes.clone()
        context[:, route_idx] = -1
        self.state.add_new_routes(context)
        self.state.set_current_routes(routes[:, route_idx])
        if self.feature_set.adjustment_features:
            self.state.set_adjustment_conditioning(
                self.cost.adjustment_degree_target,
                self.cost.penalty_weights["adjustment"],
                routes,
                gap=self.cost.adjustment_degree_gap,
                mode=self.cost.adjustment_degree_mode,
            )
        return self.bind(self.state)

    def bind(self, state, *, normalize=True):
        self.trim_counts = torch.zeros(state.batch_size, device=state.device, dtype=torch.long)
        return super().bind(state, normalize=normalize)

    def forced_halt(self):
        limit = self.params.max_route_edit_steps
        return limit is not None and self.steps >= limit

    def action_mask(self, state):
        limit = self.params.max_trim_actions_per_route
        trim = True if limit is None else self.trim_counts < limit
        halt = not (self.params.force_nonhalt_first_step and self.steps == 0)
        if self.forced_halt():
            return ActionMask(False, False, False, True)
        return ActionMask(True, trim, trim, halt)

    def _apply(self, action: RouteAction):
        forced = self.forced_halt()
        if forced:
            action = RouteAction(torch.full_like(action.kind, ROUTE_ACTION_HALT), torch.full_like(action.indices, -1))
        active = ~self.state.is_done()
        trims = ((action.kind == ROUTE_ACTION_TRIM_START) | (action.kind == ROUTE_ACTION_TRIM_END)) & active
        self.trim_counts += trims.long()
        self.state.apply_route_actions(action.kind, action.indices)

    def step(self, action: RouteAction):
        active = ~self.state.is_done()
        forced = self.forced_halt()
        if forced:
            action = RouteAction(torch.full_like(action.kind, ROUTE_ACTION_HALT), torch.full_like(action.indices, -1))
        obs, mask, (goals, penalties), done, info = super().step(action)
        # The two action penalties are episodic costs, not potential differences.
        names = ("constraints", "adjustment", "edit_step", "forced_halt")
        for name, values in (
            ("edit_step", edit_step_penalty(action.kind, active, 1.0)),
            ("forced_halt", forced_halt_penalty(active, forced, 1.0)),
        ):
            penalties[:, names.index(name)] -= values.to(penalties.dtype)
        info["forced_halt"] = forced
        return obs, mask, (goals, penalties), done, info
