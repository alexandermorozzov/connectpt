"""Regression tests for non-obvious reward leaks in the LC edit pipeline.

The PPO edit trainer credits each route-editing step with a differential
reward ``r_t = (prev_cost - new_cost) * reward_scale`` and (optionally) reshapes
trim rewards via ``zero_trim_reward`` / ``positive_only_trim_reward``.  Several
subtle invariants must hold so the agent cannot *farm* reward decoupled from
real network improvement:

* **plain** (both flags off), at ``gamma = 1``: step rewards telescope, so the
  collected reward of an episode equals ``reward_scale * (start - final) cost``
  for *any* action sequence.  No farm in any direction.
* **zero_trim_reward**: a trim earns 0 immediately and the cost baseline is
  *not* advanced, so the trim's effect is deferred to the next step.  The
  episode total still telescopes (no farm), but per-step credit shifts onto the
  following extend (a credit-assignment quirk, tested here).
* **positive_only_trim_reward**: a harmful trim is clamped to 0 (hidden) while
  the baseline still advances -> "break a route for free, get paid to undo it".
  This is a genuine leak and is asserted here so we never re-enable it silently.
* **discounting** (``gamma < 1``): telescoping breaks.  "Worsen-then-fix" is
  *penalized* (the worsening comes first, at full weight); only the unnatural
  "improve-then-undo" shape leaks a small ``(1 - gamma) * delta``.

These tests exercise the real shaping / diagnostic / return functions, not a
re-implementation, so they fail if that bookkeeping ever regresses.
"""

import pytest
import torch

from connectpt.routes_generator.transit_time_estimator import (
    ROUTE_ACTION_EXTEND,
    ROUTE_ACTION_HALT,
    ROUTE_ACTION_TRIM_END,
    ROUTE_ACTION_TRIM_START,
)
from connectpt.routes_generator.improvement_learning import (
    _compute_ppo_returns_and_advantages,
    _get_reward_delta_diagnostics,
    _is_trim_action,
    _positive_only_trim_action_rewards,
    _update_reward_baseline_cost,
    _zero_trim_action_rewards,
)

EXT = ROUTE_ACTION_EXTEND
TS = ROUTE_ACTION_TRIM_START
TE = ROUTE_ACTION_TRIM_END
HALT = ROUTE_ACTION_HALT

APPROX = dict(abs=1e-6)


# --------------------------------------------------------------------------- #
# Episode replay harness — mirrors the trainer's per-step reward bookkeeping   #
# (improvement_learning.py rollout loop) using the REAL shaping functions.     #
# --------------------------------------------------------------------------- #
def replay_episode(steps, mode="plain", reward_scale=1.0,
                   edit_step_penalty=0.0):
    """Replay one route-context episode for a scripted cost trajectory.

    ``steps`` is ``[(kind, new_cost), ...]`` where ``steps[0]`` only supplies
    the starting (baseline) cost; remaining entries are the applied actions and
    the resulting cost.  Returns ``(rewards[T, 1], start_cost, final_cost)``.
    """
    zero_trim = mode == "zero_trim"
    positive_only = mode == "positive_only"
    start_cost = float(steps[0][1])
    prev = torch.tensor([start_cost], dtype=torch.float32)
    active = torch.tensor([True])
    rewards = []
    final_cost = start_cost
    for kind, new in steps[1:]:
        new_t = torch.tensor([float(new)], dtype=torch.float32)
        kinds = torch.tensor([int(kind)], dtype=torch.long)
        step_r = (prev - new_t) * reward_scale          # plain differential
        if positive_only:
            step_r = _positive_only_trim_action_rewards(step_r, kinds, active)
        elif zero_trim:
            step_r = _zero_trim_action_rewards(step_r, kinds, active)
        if edit_step_penalty and int(kind) != HALT:
            step_r = step_r - edit_step_penalty
        prev = _update_reward_baseline_cost(
            prev, new_t, kinds, active,
            zero_trim_reward=zero_trim,
            positive_only_trim_reward=positive_only)
        rewards.append(step_r)
        final_cost = float(new)
    return torch.stack(rewards), start_cost, final_cost


def residual(rewards, start_cost, final_cost, reward_scale=1.0):
    """Leak detector: collected reward minus honest cost improvement."""
    active_masks = torch.ones_like(rewards, dtype=torch.bool)
    diag = _get_reward_delta_diagnostics(
        rewards, active_masks,
        torch.tensor([start_cost]), torch.tensor([final_cost]), reward_scale)
    return diag["residual"]


def discounted_return0(rewards, gamma):
    """Discounted return from episode start with zero value baselines."""
    horizon = rewards.shape[0]
    values = torch.zeros_like(rewards)
    final_values = torch.zeros(rewards.shape[1], dtype=torch.float32)
    dones = torch.zeros(horizon, rewards.shape[1], dtype=torch.bool)
    dones[-1] = True
    returns, _ = _compute_ppo_returns_and_advantages(
        rewards, values, dones, final_values, gamma,
        use_gae=False, gae_lambda=1.0)
    return float(returns[0, 0].item())


# --------------------------------------------------------------------------- #
# Scenario library (fake action sequences). max_trim_actions_per_route = 1,    #
# so each episode contains at most one trim.                                   #
# --------------------------------------------------------------------------- #
GENUINE_DEDUP = [(None, 1.00), (TS, 0.85), (HALT, 0.85)]
GENUINE_EXTEND = [(None, 1.00), (EXT, 0.80), (HALT, 0.80)]
# user's scenario: build a bad route (extend worsens), then trim it back
BUILD_BAD_THEN_TRIM = [(None, 1.00), (EXT, 1.50), (TS, 1.00), (HALT, 1.00)]
# positive_only farm direction: harmful TRIM (clamped to 0), then extend back
HARMFUL_TRIM_THEN_EXTEND = [(None, 1.00), (TS, 1.30), (EXT, 1.00), (HALT, 1.00)]
HARMFUL_TRIM_THEN_HALT = [(None, 1.00), (TS, 1.30), (HALT, 1.30)]
# discount-farmable shape: improve first, undo later
IMPROVE_THEN_UNDO = [(None, 1.00), (TS, 0.50), (EXT, 1.00), (HALT, 1.00)]
MULTI_IMPROVE_UNDO = [(None, 1.00), (EXT, 0.60), (EXT, 1.00),
                      (EXT, 0.60), (EXT, 1.00), (HALT, 1.00)]

ALL_SCENARIOS = [
    GENUINE_DEDUP, GENUINE_EXTEND, BUILD_BAD_THEN_TRIM,
    HARMFUL_TRIM_THEN_EXTEND, HARMFUL_TRIM_THEN_HALT,
    IMPROVE_THEN_UNDO, MULTI_IMPROVE_UNDO,
]


# --------------------------------------------------------------------------- #
# 1. Unit tests for the pure shaping helpers                                   #
# --------------------------------------------------------------------------- #
def test_is_trim_action_classifies_kinds():
    kinds = torch.tensor([EXT, TS, TE, HALT])
    assert _is_trim_action(kinds).tolist() == [False, True, True, False]


def test_zero_trim_zeros_only_trim_rewards():
    rewards = torch.tensor([0.5, 0.5, -0.5, 0.5])
    kinds = torch.tensor([EXT, TS, TE, HALT])
    active = torch.tensor([True, True, True, True])
    out = _zero_trim_action_rewards(rewards, kinds, active)
    assert out.tolist() == [0.5, 0.0, 0.0, 0.5]


def test_positive_only_hides_harmful_trim_keeps_good_trim_and_extends():
    # extend(-0.5 kept), trim(-0.5 -> 0 clamped), trim(+0.5 kept), halt(-0.5 kept)
    rewards = torch.tensor([-0.5, -0.5, 0.5, -0.5])
    kinds = torch.tensor([EXT, TS, TE, HALT])
    active = torch.tensor([True, True, True, True])
    out = _positive_only_trim_action_rewards(rewards, kinds, active)
    assert out.tolist() == [-0.5, 0.0, 0.5, -0.5]


def test_inactive_entries_are_untouched_by_shaping():
    rewards = torch.tensor([-0.5, -0.5])
    kinds = torch.tensor([TS, TS])
    active = torch.tensor([True, False])
    assert _zero_trim_action_rewards(rewards, kinds, active).tolist() == [0.0, -0.5]
    assert _positive_only_trim_action_rewards(
        rewards, kinds, active).tolist() == [0.0, -0.5]


@pytest.mark.parametrize("kind,expect_advance", [(EXT, True), (HALT, True),
                                                 (TS, False), (TE, False)])
def test_zero_trim_baseline_frozen_on_trim_only(kind, expect_advance):
    prev = torch.tensor([1.0])
    new = torch.tensor([0.5])
    active = torch.tensor([True])
    kinds = torch.tensor([int(kind)])
    out = _update_reward_baseline_cost(prev, new, kinds, active,
                                       zero_trim_reward=True)
    assert float(out.item()) == (0.5 if expect_advance else 1.0)


@pytest.mark.parametrize("kind", [EXT, TS, TE, HALT])
def test_plain_and_positive_only_always_advance_baseline(kind):
    prev = torch.tensor([1.0])
    new = torch.tensor([0.5])
    active = torch.tensor([True])
    kinds = torch.tensor([int(kind)])
    # plain
    assert float(_update_reward_baseline_cost(
        prev, new, kinds, active).item()) == 0.5
    # positive_only advances even on trims
    assert float(_update_reward_baseline_cost(
        prev, new, kinds, active,
        zero_trim_reward=True, positive_only_trim_reward=True).item()) == 0.5


# --------------------------------------------------------------------------- #
# 2. Plain reward telescopes -> no farm in ANY direction (gamma = 1)           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("steps", ALL_SCENARIOS)
def test_plain_reward_has_no_leak(steps):
    rewards, start, final = replay_episode(steps, mode="plain")
    assert residual(rewards, start, final) == pytest.approx(0.0, **APPROX)
    # discounted return at gamma=1 equals exact net improvement
    assert discounted_return0(rewards, 1.0) == pytest.approx(start - final, **APPROX)


def test_plain_build_bad_then_trim_is_not_farmable():
    # The user's worry: extend a bad route, then trim it back for reward.
    rewards, start, final = replay_episode(BUILD_BAD_THEN_TRIM, mode="plain")
    assert residual(rewards, start, final) == pytest.approx(0.0, **APPROX)
    # net state unchanged -> zero return, not positive.
    assert discounted_return0(rewards, 1.0) == pytest.approx(0.0, **APPROX)


# --------------------------------------------------------------------------- #
# 3. zero_trim: episode total still honest, but credit is deferred            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("steps", ALL_SCENARIOS)
def test_zero_trim_total_is_honest(steps):
    rewards, start, final = replay_episode(steps, mode="zero_trim")
    assert residual(rewards, start, final) == pytest.approx(0.0, **APPROX)


def test_zero_trim_defers_credit_from_trim_to_next_action():
    # good trim 1.0->0.85 then a BAD extend 0.85->0.88.
    steps = [(None, 1.00), (TS, 0.85), (EXT, 0.88), (HALT, 0.88)]
    rewards, start, final = replay_episode(steps, mode="zero_trim")
    trim_r, ext_r, _ = rewards[:, 0].tolist()
    assert trim_r == pytest.approx(0.0, **APPROX)        # trim earns nothing...
    assert ext_r == pytest.approx(0.12, **APPROX)        # ...credit lands on the extend
    # plain assigns the credit honestly instead
    plain_rewards, _, _ = replay_episode(steps, mode="plain")
    p_trim, p_ext, _ = plain_rewards[:, 0].tolist()
    assert p_trim == pytest.approx(0.15, **APPROX)
    assert p_ext == pytest.approx(-0.03, **APPROX)


def test_zero_trim_harmful_trim_then_halt_is_penalized():
    rewards, start, final = replay_episode(HARMFUL_TRIM_THEN_HALT, mode="zero_trim")
    # honest: cost rose 1.0 -> 1.3, so the episode must net negative.
    assert discounted_return0(rewards, 1.0) == pytest.approx(start - final, **APPROX)
    assert discounted_return0(rewards, 1.0) < 0.0


# --------------------------------------------------------------------------- #
# 4. positive_only_trim_reward LEAKS — assert and quantify the farm           #
# --------------------------------------------------------------------------- #
def test_positive_only_farms_harmful_trim_then_extend():
    rewards, start, final = replay_episode(HARMFUL_TRIM_THEN_EXTEND,
                                           mode="positive_only")
    # net state unchanged (1.0 -> 1.0) yet reward is strictly positive.
    res = residual(rewards, start, final)
    assert res > 1e-6
    assert res == pytest.approx(0.30, **APPROX)          # exactly the hidden -0.30 trim
    assert discounted_return0(rewards, 1.0) == pytest.approx(0.30, **APPROX)


def test_positive_only_pays_for_making_things_worse():
    # harmful trim then halt: cost ends WORSE but reward is 0, not negative.
    rewards, start, final = replay_episode(HARMFUL_TRIM_THEN_HALT,
                                           mode="positive_only")
    assert float(rewards.sum().item()) == pytest.approx(0.0, **APPROX)
    assert residual(rewards, start, final) == pytest.approx(0.30, **APPROX)


def test_positive_only_does_not_farm_extend_bad_then_trim():
    # The clamp only hides harmful TRIMS; a harmful extend keeps its penalty,
    # so the user's "build bad then trim" direction is NOT farmable here either.
    rewards, start, final = replay_episode(BUILD_BAD_THEN_TRIM,
                                           mode="positive_only")
    assert residual(rewards, start, final) == pytest.approx(0.0, **APPROX)


# --------------------------------------------------------------------------- #
# 5. Discounting (gamma < 1): telescoping breaks in a specific way            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gamma", [1.0, 0.95, 0.90])
def test_genuine_single_improvement_is_gamma_invariant(gamma):
    rewards, start, final = replay_episode(GENUINE_DEDUP, mode="plain")
    # one improving step then halt -> return == net at any gamma.
    assert discounted_return0(rewards, gamma) == pytest.approx(start - final, **APPROX)


def test_discount_penalizes_build_bad_then_fix():
    # worsen-first, fix-later: discounted return is NEGATIVE for net-zero state.
    rewards, start, final = replay_episode(BUILD_BAD_THEN_TRIM, mode="plain")
    assert discounted_return0(rewards, 1.0) == pytest.approx(0.0, **APPROX)
    assert discounted_return0(rewards, 0.90) < -1e-6


def test_discount_only_leaks_improve_then_undo():
    # improve-first, undo-later is the single farmable shape, and only at gamma<1.
    rewards, start, final = replay_episode(IMPROVE_THEN_UNDO, mode="plain")
    assert discounted_return0(rewards, 1.0) == pytest.approx(0.0, **APPROX)
    leak = discounted_return0(rewards, 0.90)
    assert leak > 1e-6
    assert leak == pytest.approx(0.05, **APPROX)         # (1 - gamma) * 0.5


def test_discount_multi_cycle_leak_grows_with_cycles_but_stays_bounded():
    rewards, _, _ = replay_episode(MULTI_IMPROVE_UNDO, mode="plain")
    g1 = discounted_return0(rewards, 1.0)
    g09 = discounted_return0(rewards, 0.90)
    assert g1 == pytest.approx(0.0, **APPROX)             # net-zero at gamma=1
    assert g09 > 1e-6                                     # leaks under discount
    # bounded: cannot exceed the total improvement transacted (0.4 * 2 cycles)
    assert g09 < 0.8


# --------------------------------------------------------------------------- #
# 6. Intentional residuals (edit penalty) are not mistaken for leaks          #
# --------------------------------------------------------------------------- #
def test_edit_step_penalty_residual_is_negative_and_accounted():
    # GENUINE_DEDUP has one non-halt action (the trim); penalty applies once.
    rewards, start, final = replay_episode(
        GENUINE_DEDUP, mode="plain", edit_step_penalty=0.05)
    # residual is exactly -(n_nonhalt * penalty); negative => not a farm.
    assert residual(rewards, start, final) == pytest.approx(-0.05, **APPROX)
