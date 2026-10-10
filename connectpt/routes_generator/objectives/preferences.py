import torch
from torch import Tensor
from dataclasses import dataclass
from .const import GOAL_NAMES, COST_WEIGHT_KEY_ORDER, PREFERENCE_KEYS

from ..common.const import EPSILON

# NOTE on ordering: the state exposes cost weights as a NAMED dict
# (state.cost_weights) — exactly how the cost math consumes them (lookup by
# key at each use site). A positional vector exists only where a neural net
# needs one, and is assembled in a single place (get_cost_weights_tensor)
# from an EXPLICIT key tuple supplied by the caller: the cost/reward math
# passes COST_WEIGHT_KEY_ORDER; models pass the checkpoint-locked spec they
# declare themselves (models.DEFAULT_PREFERENCE_GOALS / preference_goals).

# Short, human-readable names for the three cost components, index-aligned
# with COST_WEIGHT_KEY_ORDER. Used to enable / disable individual components.


def batch_value(value, batch_size, device):
    """Broadcast a named scalar weight using the historical batch convention."""
    if type(value) is not Tensor:
        value = torch.tensor([value], device=device)
    else:
        value = value.to(device)
        if value.ndim == 0:
            value = value[None]
    if value.shape[0] == 1 and batch_size > 1:
        value = value.expand(batch_size)
    if value.shape[0] > batch_size:
        value = value[:batch_size]
    return value


def named_preferences(state, goals):
    """Use goal IDs in configuration order, independent of model input slots."""
    weights = torch.stack([batch_value(state.cost_weights[PREFERENCE_KEYS.get(name, name)],
                                      state.batch_size, state.device) for name in goals], -1)
    if set(goals) == set(GOAL_NAMES):
        return weights
    total = weights.sum(-1, keepdim=True)
    return torch.where(total > EPSILON, weights / total.clamp_min(EPSILON), torch.full_like(weights, 1 / len(goals)))


@dataclass(frozen=True)
class PreferenceSamplerParams:
    objectives: tuple[str,...]
    demand_time_weight: float
    route_time_weight: float
    median_connectivity_weight: float
    variable_weights: bool
    pp_fraction: float
    op_fraction: float
    mcw_fraction: float

    def __post_init__(self):
        if not self.objectives or len(set(self.objectives))!=len(self.objectives):
            raise ValueError('sampler objectives must be nonempty and unique')
        fractions=(self.pp_fraction,self.op_fraction,self.mcw_fraction)
        if any(not 0<=value<=1 for value in fractions) or sum(fractions)>1:
            raise ValueError('fractions of extreme samples must sum to <= 1')


def mask_component_weights(weights,enabled,*,normalize=True):
    if all(enabled):
        return weights
    mask=torch.tensor([1.0 if on else 0.0 for on in enabled],device=weights.device,dtype=weights.dtype)
    weights=weights*mask
    if normalize:
        total=weights.sum(dim=-1,keepdim=True)
        uniform=mask/mask.sum()
        weights=torch.where(total>EPSILON,weights/total.clamp_min(EPSILON),uniform)
    return weights


def mask_component_dict(weights,enabled,*,normalize=True):
    if all(enabled):
        return weights
    stacked=torch.stack([weights[key] for key in COST_WEIGHT_KEY_ORDER],dim=-1)
    stacked=mask_component_weights(stacked,enabled,normalize=normalize)
    result=dict(weights)
    for i,key in enumerate(COST_WEIGHT_KEY_ORDER):
        result[key]=stacked[...,i]
    return result


def fixed_preferences(objectives,demand_time_weight,route_time_weight,median_connectivity_weight,*,device=None):
    values=(demand_time_weight,route_time_weight,median_connectivity_weight)
    weights={key:value if type(value) is Tensor else torch.tensor([value],device=device)
             for key,value in zip(COST_WEIGHT_KEY_ORDER,values)}
    return mask_component_dict(weights,tuple(name in objectives for name in GOAL_NAMES))


def component_preferences(state,enabled,*,normalize=False):
    if all(enabled):
        return state.get_cost_weights_tensor(COST_WEIGHT_KEY_ORDER,normalize=normalize)
    weights=state.get_cost_weights_tensor(COST_WEIGHT_KEY_ORDER,normalize=False)
    return mask_component_weights(weights,enabled,normalize=normalize)


def state_preferences(named):
    """Map goal IDs to State keys and retain disabled checkpoint slots."""
    prototype=next(iter(named.values()))
    result={key:torch.zeros_like(prototype) for key in COST_WEIGHT_KEY_ORDER}
    for name,value in named.items():
        result[PREFERENCE_KEYS[name] if name in PREFERENCE_KEYS else name]=value
    return result


class PreferenceSampler:
    """Legacy preference sampling adapter; retain checkpoint slots and RNG."""

    def __init__(self, params):
        if not isinstance(params,PreferenceSamplerParams):
            raise TypeError('PreferenceSampler requires PreferenceSamplerParams')
        self.params=params
        self.objectives=tuple(params.objectives)
        self.enabled=tuple(name in self.objectives for name in GOAL_NAMES)

    def sample_variable_weights(self, batch_size, device=None):
        if not self.params.variable_weights:
            dtw = torch.full((batch_size,), self.params.demand_time_weight, device=device)
            rtw = torch.full((batch_size,), self.params.route_time_weight, device=device)
            mcw = torch.full((batch_size,), self.params.median_connectivity_weight, device=device)
        else:
            random_number = torch.rand(batch_size, device=device)
            dtw = torch.zeros(batch_size, device=device)
            rtw = torch.zeros(batch_size, device=device)
            mcw = torch.zeros(batch_size, device=device)

            # Set weights for extreme cases
            is_pp = random_number < self.params.pp_fraction
            dtw[is_pp] = 1.0  # Passenger-perspective: demand_time_weight = 1
            is_op = (self.params.pp_fraction <= random_number) & (random_number < self.params.pp_fraction + self.params.op_fraction)
            rtw[is_op] = 1.0  # Operator-perspective: route_time_weight = 1
            is_mcw = (self.params.pp_fraction + self.params.op_fraction <= random_number) & (
                random_number < self.params.pp_fraction + self.params.op_fraction + self.params.mcw_fraction
            )
            mcw[is_mcw] = 1.0  # Connectivity-perspective: median_connectivity_weight = 1

            # For intermediate cases, generate random weights that sum to 1
            extremes_fraction = self.params.pp_fraction + self.params.op_fraction + self.params.mcw_fraction
            is_intermediate = random_number >= extremes_fraction
            n_intermediate = is_intermediate.sum()

            if n_intermediate > 0:
                # Two objectives use Holliday's uniform alpha. Other component
                # counts retain the existing normalized-uniform distribution.
                enabled_idx = [i for i, on in enumerate(self.enabled) if on]
                cols = [dtw, rtw, mcw]
                if len(enabled_idx) == 2:
                    alpha = torch.rand(int(n_intermediate), device=device)
                    weights = torch.stack((alpha, 1 - alpha), dim=-1)
                else:
                    weights = torch.rand(int(n_intermediate), len(enabled_idx), device=device)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                for _j, _ci in enumerate(enabled_idx):
                    cols[_ci][is_intermediate] = weights[:, _j]

        return mask_component_dict(
            {
                "demand_time_weight": dtw,
                "route_time_weight": rtw,
                "median_connectivity_weight": mcw,
            },self.enabled
        )

    def sample_named(self,batch_size,device=None):
        """Return goal_id -> Tensor[B], in configured goal order."""
        if all(name in GOAL_NAMES for name in self.objectives):
            legacy=self.sample_variable_weights(batch_size,device)
            return {name:legacy[PREFERENCE_KEYS[name]] for name in self.objectives}
        weights=self.sample(batch_size,device)
        return {name:weights[:,i] for i,name in enumerate(self.objectives)}

    def sample(self, batch_size, device=None):
        """Sample only active goals, in configured objective order."""
        if any(name not in GOAL_NAMES for name in self.objectives):
            weights = torch.rand(batch_size, len(self.objectives), device=device)
            return weights / weights.sum(-1, keepdim=True)
        named = self.sample_variable_weights(batch_size, device)
        return torch.stack([named[COST_WEIGHT_KEY_ORDER[GOAL_NAMES.index(name)]] for name in self.objectives], dim=-1)


def scalarize(goals, preferences, penalties, penalty_weights):
    """Weighted goals followed by ordered penalties; preserve float arithmetic."""
    value = (goals * preferences).sum(dim=-1)
    for index in range(penalties.shape[-1]):
        value = value + penalties[:, index] * penalty_weights[index]
    return value
