import pytest
import torch

from connectpt.routes_generator.initialization import prepare_init_network


def test_prepare_init_network_broadcasts_single_batch():
    init_network = torch.tensor([[[0, 1, -1], [2, 3, -1]]], dtype=torch.long)

    prepared = prepare_init_network(
        init_network,
        batch_size=3,
        n_routes=3,
        device=torch.device("cpu"),
    )

    assert prepared.shape == (3, 2, 3)
    assert torch.equal(prepared[0], init_network[0])
    assert torch.equal(prepared[1], init_network[0])
    assert torch.equal(prepared[2], init_network[0])


def test_prepare_init_network_rejects_too_many_routes():
    init_network = torch.tensor([[[0, 1, -1], [2, 3, -1]]], dtype=torch.long)

    with pytest.raises(ValueError, match="too many routes"):
        prepare_init_network(
            init_network,
            batch_size=1,
            n_routes=1,
            device=torch.device("cpu"),
        )


def test_prepare_init_network_rejects_batch_mismatch():
    init_network = torch.tensor(
        [
            [[0, 1, -1]],
            [[2, 3, -1]],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="batch size"):
        prepare_init_network(
            init_network,
            batch_size=3,
            n_routes=2,
            device=torch.device("cpu"),
        )
