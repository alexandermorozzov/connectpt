"""Pin experiments._common.pad_routes_to / route_2d to the behaviours they
replaced (training silent pad/clip vs MACSA validate-and-raise)."""
import sys
from pathlib import Path

import pytest
import torch

ROUTE_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "route_generator"
if str(ROUTE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROUTE_EXAMPLES))

from experiments._common import pad_routes_to, route_2d  # noqa: E402


def test_route_2d_squeezes_batch_dim():
    x = torch.zeros(1, 3, 5)
    assert route_2d(x).shape == (3, 5)
    y = torch.zeros(3, 5)
    assert route_2d(y).shape == (3, 5)


def test_pad_non_strict_pads_and_clips_both_dims():
    # too few routes / too short -> padded with -1
    out = pad_routes_to(torch.zeros(2, 3), n_routes=4, max_route_len=5)
    assert out.shape == (4, 5)
    assert (out[2:] == -1).all() and (out[:, 3:] == -1).all()
    # too many routes / too wide -> clipped
    wide = torch.arange(6 * 8).reshape(6, 8)
    out = pad_routes_to(wide, n_routes=3, max_route_len=5)
    assert out.shape == (3, 5)


def test_pad_strict_validates_route_count_and_width():
    good = torch.zeros(4, 3)
    out = pad_routes_to(good, n_routes=4, max_route_len=5, strict=True)
    assert out.shape == (4, 5)  # short width padded
    with pytest.raises(ValueError):
        pad_routes_to(torch.zeros(3, 3), n_routes=4, max_route_len=5, strict=True)
    with pytest.raises(ValueError):
        pad_routes_to(torch.zeros(4, 9), n_routes=4, max_route_len=5, strict=True)


def test_modules_still_import_helpers():
    import experiments.macsa as m
    import experiments.training_lc as t
    assert callable(m._macsa_pad_routes) and callable(m._macsa_2d)
    assert callable(t._pad_routes_to)
