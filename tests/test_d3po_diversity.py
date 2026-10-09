import torch
import pytest

from connectpt.routes_generator.training.algorithms import d3po as learning
from connectpt.routes_generator.policy.models import (
    TORCH_FMIN,
    _joint_route_action_log_probs,
)


@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("temperature", [0.7, 1.6])
def test_joint_distribution_includes_conditional_continue_and_forced_halt(serial, temperature):
    scores = torch.tensor([[1.0, -1.0, 9.0], [1.0, -1.0, 9.0]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [False, False, False]])
    halt = torch.tensor([[-0.4], [TORCH_FMIN]], requires_grad=True)
    logp = _joint_route_action_log_probs(scores, valid, halt, serial, temperature, torch.zeros(2, dtype=torch.bool))
    if serial:
        decision = torch.distributions.Categorical(logits=torch.tensor([0.4, -0.4]) / temperature)
        route = torch.distributions.Categorical(logits=torch.tensor([1.0, -1.0]) / temperature)
        expected = torch.cat((decision.probs[:1] * route.probs, torch.zeros(1), decision.probs[1:]))
    else:
        expected = torch.distributions.Categorical(
            logits=torch.tensor([1.0, -1.0, -torch.inf, -0.4]) / temperature
        ).probs
    assert torch.allclose(logp[0].exp(), expected.to(torch.float64), atol=1e-7)
    assert torch.equal(logp[1].exp(), torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
    assert torch.allclose(logp.exp().sum(-1), torch.ones(2, dtype=torch.float64), atol=1e-7)
    logp[0, 0].backward()
    assert scores.grad.isfinite().all() and halt.grad.isfinite().all()
    assert scores.grad[0, 2] == 0
    assert torch.equal(scores.grad[1], torch.zeros(3))


def test_full_kl_matches_categorical_oracle_with_masked_and_rare_actions():
    p_logits = torch.tensor([[0.0, -30.0, -torch.inf], [0.1, -0.3, -torch.inf]], requires_grad=True)
    q_logits = torch.tensor([[0.0, 0.0, -torch.inf], [-0.8, 0.2, -torch.inf]], requires_grad=True)
    p, q = [torch.distributions.Categorical(logits=x) for x in (p_logits, q_logits)]
    kl = learning.categorical_kl(p.logits, q.logits)
    expected = torch.distributions.kl_divergence(p, q)
    assert torch.allclose(kl, expected, atol=1e-7)
    kl.sum().backward()
    assert p_logits.grad.isfinite().all() and q_logits.grad.isfinite().all()
    assert torch.equal(p_logits.grad[:, 2], torch.zeros(2))


def test_diversity_squares_full_kl_and_is_independent_of_replayed_action(monkeypatch):
    # With target = exact KL, paper Eq. 2 is zero for either sampled action.
    # Squaring the one-action k3 estimate instead gives a large nonzero loss.
    p = torch.tensor([[0.99, 0.01]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    preferences = torch.tensor([[0.2, 0.8, 0.0]], dtype=torch.float64)
    neighbor = torch.tensor([[0.8, 0.2, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(learning, "sample_neighbor_preferences", lambda *args, **kwargs: neighbor)

    class State:
        cost_weights = learning.preference_dict(preferences)

        def set_cost_weights(self, weights):
            self.cost_weights = weights

    from types import SimpleNamespace

    env = SimpleNamespace(observe=lambda state: (state, None))

    class Model:
        feature_set = None
        action_spec = SimpleNamespace(role="construction")

        def evaluate(self, state, mask, action):
            probs = p if state.cost_weights["demand_time_weight"][0] < 0.5 else q
            return SimpleNamespace(log_probs=probs.log())

    cost = SimpleNamespace(
        get_preference_weights=lambda state, **kwargs: preferences, enabled_component_mask=(True, True, False)
    )
    params = SimpleNamespace(diversity_weight=1.0, preference_noise_sigma=0.15, diversity_alpha=0.0)
    algorithm = learning.D3PO(params, cost)
    state = State()
    original = {k: v.clone() for k, v in state.cost_weights.items()}
    exact_kl = torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=p), torch.distributions.Categorical(probs=q)
    )
    alpha = float(exact_kl.detach() / (preferences - neighbor).abs().sum())
    for action in [0, 1]:
        params.diversity_alpha = alpha
        output = SimpleNamespace(log_probs=p.log(), action=torch.tensor([[action]]))
        loss = algorithm.diversity(Model(), env, state, output, None).loss
        assert loss.item() < 1e-20
        assert all(torch.equal(state.cost_weights[k], v) for k, v in original.items())
    params.diversity_alpha = 0.0
    loss = algorithm.diversity(Model(), env, state, output, None).loss
    assert torch.allclose(loss, exact_kl.square().mean())
    loss.backward()
    assert p.grad.isfinite().all() and q.grad.isfinite().all()
    assert p.grad.abs().sum() > 0 and q.grad.abs().sum() > 0
