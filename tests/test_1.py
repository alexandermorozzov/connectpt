from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "connectpt" / "routes_generator" / "torch_utils.py"
MODULE_SPEC = spec_from_file_location("torch_utils_under_test", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
torch_utils = module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(torch_utils)


def test_floyd_warshall_returns_shortest_paths_for_single_graph():
    edge_costs = torch.tensor([
        [0.0, 1.0, float("inf")],
        [float("inf"), 0.0, 2.0],
        [3.0, float("inf"), 0.0],
    ])

    nexts, dists = torch_utils.floyd_warshall(edge_costs)

    expected_dists = torch.tensor([
        [0.0, 1.0, 3.0],
        [5.0, 0.0, 2.0],
        [3.0, 4.0, 0.0],
    ])

    assert torch.equal(dists.squeeze(0), expected_dists)
    assert nexts.squeeze(0)[0, 2].item() == 1
    assert torch_utils.reconstruct_path(0, 2, nexts.squeeze(0)) == [0, 1, 2]


def test_floyd_warshall_supports_batched_graphs():
    edge_costs = torch.tensor([
        [
            [0.0, 1.0, float("inf")],
            [float("inf"), 0.0, 2.0],
            [3.0, float("inf"), 0.0],
        ],
        [
            [0.0, 2.0, 10.0],
            [float("inf"), 0.0, 2.0],
            [1.0, float("inf"), 0.0],
        ],
    ])

    nexts, dists = torch_utils.floyd_warshall(edge_costs)

    expected_dists = torch.tensor([
        [
            [0.0, 1.0, 3.0],
            [5.0, 0.0, 2.0],
            [3.0, 4.0, 0.0],
        ],
        [
            [0.0, 2.0, 4.0],
            [3.0, 0.0, 2.0],
            [1.0, 3.0, 0.0],
        ],
    ])

    assert torch.equal(dists, expected_dists)
    assert nexts[0, 0, 2].item() == 1
    assert nexts[1, 0, 2].item() == 1

