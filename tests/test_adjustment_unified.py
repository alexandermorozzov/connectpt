"""Tests for the unified adjustment-degree penalty in MyCostModule.

The cost module now owns the adjustment-degree term (single source of truth,
shared by BCO / metaheuristics / training). It is gated: with weight 0 or no
seed reference the term is skipped, so default behaviour is unchanged.
"""
import torch
from types import SimpleNamespace

from connectpt.routes_generator.transit_time_estimator import MyCostModule
from connectpt.routes_generator.bee_colony import get_adjustment_penalties
from connectpt.routes_generator import torch_utils as tu


def _stub_state(routes_batched):
    """Minimal state stub exposing only ``.routes`` (all _adjustment_penalty uses)."""
    return SimpleNamespace(routes=routes_batched)


def test_adjustment_penalties_objectives():
    deg = torch.tensor([0.0, 0.2, 0.5])
    assert torch.allclose(get_adjustment_penalties(deg, "raw"), deg)
    assert torch.allclose(get_adjustment_penalties(deg, "target", target=0.2),
                          (deg - 0.2).abs())
    assert torch.allclose(get_adjustment_penalties(deg, "cap", target=0.2),
                          (deg - 0.2).clamp(min=0.0))


def test_penalty_gated_off_by_default():
    m = MyCostModule()  # weight 0.0, seed None
    st = _stub_state([[[0, 1, 2, 3], [4, 5, 6, 7]]])
    assert float(m._adjustment_penalty(st, torch.zeros(1))) == 0.0
    # weight set but no seed -> still off
    m.adjustment_degree_weight = 5.0
    assert float(m._adjustment_penalty(st, torch.zeros(1))) == 0.0


def test_penalty_zero_when_routes_equal_seed():
    routes = [[[0, 1, 2, 3], [4, 5, 6, 7]]]
    m = MyCostModule(adjustment_degree_weight=10.0,
                     adjustment_degree_objective="raw",
                     adjustment_degree_mode="paper")
    m.adjustment_seed = tu.get_batch_tensor_from_routes(routes)
    pen = m._adjustment_penalty(_stub_state(routes), torch.zeros(1))
    assert float(pen) == 0.0  # identical network -> adjustment degree 0


def test_penalty_positive_when_routes_differ():
    cur = [[[0, 1, 2, 3], [4, 5, 6, 7]]]
    seed = [[[8, 9, 10, 11], [4, 5, 6, 7]]]   # route 0 entirely changed
    m = MyCostModule(adjustment_degree_weight=10.0,
                     adjustment_degree_objective="raw",
                     adjustment_degree_mode="paper")
    m.adjustment_seed = tu.get_batch_tensor_from_routes(seed)
    pen = m._adjustment_penalty(_stub_state(cur), torch.zeros(1))
    assert float(pen) > 0.0


def test_cap_objective_no_penalty_below_target():
    cur = [[[0, 1, 2, 3], [4, 5, 6, 7]]]
    seed = [[[8, 9, 10, 11], [4, 5, 6, 7]]]
    # cap with target 1.0 -> nothing exceeds it -> zero penalty even though adj>0
    m = MyCostModule(adjustment_degree_weight=10.0,
                     adjustment_degree_objective="cap",
                     adjustment_degree_target=1.0,
                     adjustment_degree_mode="paper")
    m.adjustment_seed = tu.get_batch_tensor_from_routes(seed)
    pen = m._adjustment_penalty(_stub_state(cur), torch.zeros(1))
    assert float(pen) == 0.0
