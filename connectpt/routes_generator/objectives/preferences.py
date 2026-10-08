import torch
from torch import Tensor

EPSILON = 1e-6
COST_WEIGHT_KEY_ORDER = (
    "demand_time_weight",
    "route_time_weight",
    "median_connectivity_weight",
)

# NOTE on ordering: the state exposes cost weights as a NAMED dict
# (state.cost_weights) — exactly how the cost math consumes them (lookup by
# key at each use site). A positional vector exists only where a neural net
# needs one, and is assembled in a single place (get_cost_weights_tensor)
# from an EXPLICIT key tuple supplied by the caller: the cost/reward math
# passes COST_WEIGHT_KEY_ORDER; models pass the checkpoint-locked spec they
# declare themselves (models.DEFAULT_PREFERENCE_GOALS / preference_goals).

# Short, human-readable names for the three cost components, index-aligned
# with COST_WEIGHT_KEY_ORDER. Used to enable / disable individual components.
COST_COMPONENT_NAMES = ("demand", "route", "connectivity")


class PreferenceSampler:
    def _enabled_mask_row(self, device=None, dtype=torch.float32):
        return torch.tensor([1.0 if on else 0.0 for on in self._enabled_components], device=device, dtype=dtype)

    def apply_enabled_mask(self, weights, normalize=True):
        """Zero the weights of disabled cost components along the last axis
        and (optionally) renormalize the survivors to sum to 1.

        ``weights`` is a ``[..., 3]`` tensor. This is a no-op when every
        component is enabled, so default runs are unaffected. Rows whose
        enabled weights are all zero fall back to a uniform distribution
        over the enabled components instead of producing a degenerate
        all-zero (pure-constraint) cost."""
        if self._all_components_enabled:
            return weights
        mask = self._enabled_mask_row(weights.device, weights.dtype)
        weights = weights * mask
        if normalize:
            total = weights.sum(dim=-1, keepdim=True)
            uniform = mask / mask.sum()
            weights = torch.where(total > EPSILON, weights / total.clamp_min(EPSILON), uniform)
        return weights

    def _mask_weight_dict(self, wdict, normalize=True):
        """Apply ``apply_enabled_mask`` to a dict of per-component weight
        tensors keyed by ``COST_WEIGHT_KEY_ORDER``."""
        if self._all_components_enabled:
            return wdict
        keys = list(COST_WEIGHT_KEY_ORDER)
        stacked = torch.stack([wdict[k] for k in keys], dim=-1)
        stacked = self.apply_enabled_mask(stacked, normalize=normalize)
        out = dict(wdict)
        for i, k in enumerate(keys):
            out[k] = stacked[..., i]
        return out

    def sample_variable_weights(self, batch_size, device=None):
        if not self.variable_weights:
            dtw = torch.full((batch_size,), self.demand_time_weight, device=device)
            rtw = torch.full((batch_size,), self.route_time_weight, device=device)
            mcw = torch.full((batch_size,), self.median_connectivity_weight, device=device)
        else:
            random_number = torch.rand(batch_size, device=device)
            dtw = torch.zeros(batch_size, device=device)
            rtw = torch.zeros(batch_size, device=device)
            mcw = torch.zeros(batch_size, device=device)

            # Set weights for extreme cases
            is_pp = random_number < self.pp_fraction
            dtw[is_pp] = 1.0  # Passenger-perspective: demand_time_weight = 1
            is_op = (self.pp_fraction <= random_number) & (random_number < self.pp_fraction + self.op_fraction)
            rtw[is_op] = 1.0  # Operator-perspective: route_time_weight = 1
            is_mcw = (self.pp_fraction + self.op_fraction <= random_number) & (
                random_number < self.pp_fraction + self.op_fraction + self.mcw_fraction
            )
            mcw[is_mcw] = 1.0  # Connectivity-perspective: median_connectivity_weight = 1

            # For intermediate cases, generate random weights that sum to 1
            extremes_fraction = self.pp_fraction + self.op_fraction + self.mcw_fraction
            is_intermediate = random_number >= extremes_fraction
            n_intermediate = is_intermediate.sum()

            if n_intermediate > 0:
                # Two objectives use Holliday's uniform alpha. Other component
                # counts retain the existing normalized-uniform distribution.
                enabled_idx = [i for i, on in enumerate(self._enabled_components) if on]
                cols = [dtw, rtw, mcw]
                if len(enabled_idx) == 2:
                    alpha = torch.rand(int(n_intermediate), device=device)
                    weights = torch.stack((alpha, 1 - alpha), dim=-1)
                else:
                    weights = torch.rand(int(n_intermediate), len(enabled_idx), device=device)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                for _j, _ci in enumerate(enabled_idx):
                    cols[_ci][is_intermediate] = weights[:, _j]

        return self._mask_weight_dict(
            {
                "demand_time_weight": dtw,
                "route_time_weight": rtw,
                "median_connectivity_weight": mcw,
            }
        )

    def get_weights(self, device=None):
        dtm = self.demand_time_weight
        if type(dtm) is not Tensor:
            dtm = torch.tensor([dtm], device=device)
        rtm = self.route_time_weight
        if type(rtm) is not Tensor:
            rtm = torch.tensor([rtm], device=device)
        mcw = self.median_connectivity_weight
        if type(mcw) is not Tensor:
            mcw = torch.tensor([mcw], device=device)

        return self._mask_weight_dict(
            {
                "demand_time_weight": dtm,
                "route_time_weight": rtm,
                "median_connectivity_weight": mcw,
            }
        )

    def get_preference_weights(self, state, normalize=False):
        if self._all_components_enabled:
            return state.get_cost_weights_tensor(COST_WEIGHT_KEY_ORDER, normalize=normalize)

        # A disabled component must not carry preference mass — mask the
        # raw weights, then renormalize the survivors when requested.
        weights = state.get_cost_weights_tensor(COST_WEIGHT_KEY_ORDER, normalize=False)
        return self.apply_enabled_mask(weights, normalize=normalize)

    def sample(self, batch_size, device=None):
        """Sample only active goals, in configured objective order."""
        from .criteria import GOAL_NAMES

        named = self.sample_variable_weights(batch_size, device)
        return torch.stack([named[COST_WEIGHT_KEY_ORDER[GOAL_NAMES.index(name)]] for name in self.objectives], dim=-1)


def scalarize(goals, preferences, penalties, penalty_weights):
    """Weighted goals followed by ordered penalties; preserve float arithmetic."""
    value = (goals * preferences).sum(dim=-1)
    for index in range(penalties.shape[-1]):
        value = value + penalties[:, index] * penalty_weights[index]
    return value
