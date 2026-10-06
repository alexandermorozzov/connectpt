import torch
import pytest

from connectpt.routes_generator import improvement_learning as learning
from connectpt.routes_generator.models import (
    TORCH_FMIN, _joint_route_action_log_probs,
)


@pytest.mark.parametrize('serial', [False, True])
@pytest.mark.parametrize('temperature', [.7, 1.6])
def test_joint_distribution_includes_conditional_continue_and_forced_halt(serial, temperature):
    scores = torch.tensor([[1., -1., 9.], [1., -1., 9.]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [False, False, False]])
    halt = torch.tensor([[-.4], [TORCH_FMIN]], requires_grad=True)
    logp = _joint_route_action_log_probs(
        scores, valid, halt, serial, temperature, torch.zeros(2, dtype=torch.bool))
    if serial:
        decision = torch.distributions.Categorical(logits=torch.tensor([.4, -.4]) / temperature)
        route = torch.distributions.Categorical(logits=torch.tensor([1., -1.]) / temperature)
        expected = torch.cat((decision.probs[:1] * route.probs, torch.zeros(1), decision.probs[1:]))
    else:
        expected = torch.distributions.Categorical(
            logits=torch.tensor([1., -1., -torch.inf, -.4]) / temperature).probs
    assert torch.allclose(logp[0].exp(), expected.to(torch.float64), atol=1e-7)
    assert torch.equal(logp[1].exp(), torch.tensor([0., 0., 0., 1.], dtype=torch.float64))
    assert torch.allclose(logp.exp().sum(-1), torch.ones(2, dtype=torch.float64), atol=1e-7)
    logp[0, 0].backward()
    assert scores.grad.isfinite().all() and halt.grad.isfinite().all()
    assert scores.grad[0, 2] == 0
    assert torch.equal(scores.grad[1], torch.zeros(3))


def test_full_kl_matches_categorical_oracle_with_masked_and_rare_actions():
    p_logits = torch.tensor([[0., -30., -torch.inf], [.1, -.3, -torch.inf]], requires_grad=True)
    q_logits = torch.tensor([[0., 0., -torch.inf], [-.8, .2, -torch.inf]], requires_grad=True)
    p, q = [torch.distributions.Categorical(logits=x) for x in (p_logits, q_logits)]
    kl = learning._categorical_kl(p.logits, q.logits)
    expected = torch.distributions.kl_divergence(p, q)
    assert torch.allclose(kl, expected, atol=1e-7)
    kl.sum().backward()
    assert p_logits.grad.isfinite().all() and q_logits.grad.isfinite().all()
    assert torch.equal(p_logits.grad[:, 2], torch.zeros(2))


def test_diversity_squares_full_kl_and_is_independent_of_replayed_action(monkeypatch):
    # With target = exact KL, paper Eq. 2 is zero for either sampled action.
    # Squaring the one-action k3 estimate instead gives a large nonzero loss.
    p = torch.tensor([[.99, .01]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[.5, .5]], dtype=torch.float64, requires_grad=True)
    preferences = torch.tensor([[.2, .8, 0.]], dtype=torch.float64)
    neighbor = torch.tensor([[.8, .2, 0.]], dtype=torch.float64)
    monkeypatch.setattr(learning, '_sample_neighbor_preferences', lambda *args, **kwargs: neighbor)

    class State:
        cost_weights = learning._preference_dict_from_tensor(preferences)

        def set_cost_weights(self, weights):
            self.cost_weights = weights

    class Model:
        def step(self, state, actions, return_distribution=False):
            probs = p if state.cost_weights['demand_time_weight'][0] < .5 else q
            logp = probs.log()
            return actions, logp.gather(1, actions).squeeze(1), torch.zeros(1), logp

    state = State()
    original = {k: v.clone() for k, v in state.cost_weights.items()}
    exact_kl = torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=p), torch.distributions.Categorical(probs=q))
    alpha = float(exact_kl.detach() / (preferences - neighbor).abs().sum())
    for action in [0, 1]:
        loss = learning._estimate_d3po_diversity_loss(
            Model(), state, torch.tensor([[action]]), None, None, p[:, action].log(),
            preferences, False, .15, alpha)
        assert loss.item() < 1e-20
        assert all(torch.equal(state.cost_weights[k], v) for k, v in original.items())
    loss = learning._estimate_d3po_diversity_loss(
        Model(), state, torch.tensor([[1]]), None, None, p[:, 1].log(),
        preferences, False, .15, 0.)
    assert torch.allclose(loss, exact_kl.square().mean())
    loss.backward()
    assert p.grad.isfinite().all() and q.grad.isfinite().all()
    assert p.grad.abs().sum() > 0 and q.grad.abs().sum() > 0
