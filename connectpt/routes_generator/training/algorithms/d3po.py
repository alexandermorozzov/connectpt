"""Goal-wise clipped PPO with full categorical KL diversity."""
import torch

from ...objectives.preferences import COST_WEIGHT_KEY_ORDER, scalarize
from .base import Algorithm, Diversity


def normalize_preferences(weights):
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def preference_dict(weights):
    return {key: weights[:, i] for i, key in enumerate(COST_WEIGHT_KEY_ORDER)}


def sample_neighbor_preferences(preferences, sigma, enabled_mask):
    # Keep the original three named slots, including disabled goals, for RNG.
    neighbor = (preferences + torch.randn_like(preferences) * sigma).clamp_min(1e-6)
    if not all(enabled_mask):
        neighbor = neighbor * torch.tensor(enabled_mask, device=neighbor.device, dtype=neighbor.dtype)
    return normalize_preferences(neighbor)


def categorical_kl(log_probs, neighbor_log_probs):
    supported = torch.isfinite(log_probs)
    safe_log_probs = torch.where(supported, log_probs, 0.0)
    safe_neighbor = torch.where(supported, neighbor_log_probs, 0.0)
    return (log_probs.exp() * (safe_log_probs - safe_neighbor)).sum(-1)


class D3PO(Algorithm):
    scalar_values = False

    @property
    def value_dim(self):
        return len(self.cost.objectives)

    def potential(self, result):
        # D2 preserves the current convention: each head includes penalties.
        penalties = scalarize(
            torch.zeros_like(result.goals), result.preferences, result.penalties, result.penalty_weights
        )
        return result.goals + penalties[:, None]

    def surrogate(self, ratios, clipped, advantages, preferences):
        clipped_goals = torch.minimum(ratios[:, None] * advantages, clipped[:, None] * advantages)
        return (preferences * clipped_goals).sum(-1)

    def diversity(self, policy, env, state, output, mask):
        if self.params.diversity_weight == 0:
            zero = output.logp.new_zeros(())
            return Diversity(zero, zero, zero)
        preferences = self.cost.get_preference_weights(state, normalize=True)
        neighbor = sample_neighbor_preferences(
            preferences, self.params.preference_noise_sigma, self.cost.enabled_component_mask
        )
        target = self.params.diversity_alpha * (preferences - neighbor).abs().sum(-1)
        saved = {key: value.clone() for key, value in state.cost_weights.items()}
        state.set_cost_weights(preference_dict(neighbor))
        try:
            obs, _ = env.observe(state)
            neighbor_output = policy.evaluate(obs, mask, output.action)
        finally:
            state.set_cost_weights(saved)
        kl = categorical_kl(output.log_probs, neighbor_output.log_probs)
        return Diversity((kl - target).pow(2).mean(), kl.mean(), target.mean())
