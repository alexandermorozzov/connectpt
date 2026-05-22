from pathlib import Path

import torch
from omegaconf import OmegaConf

from connectpt.routes_generator.citygraph_dataset import (
    get_dataset_from_config,
    load_macsa_scenario,
    load_macsa_scenarios,
    load_macsa_tensors,
)


MACSA_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "MACSA_data"
MANDL_8_DIR = MACSA_ROOT / "mandl_8"


def test_load_macsa_tensors_reads_mandl_8_files():
    tensors = load_macsa_tensors(MANDL_8_DIR)

    assert tensors["node_locs"].shape == (15, 2)
    assert tensors["street_adj"].shape == (15, 15)
    assert tensors["demand"].shape == (15, 15)
    assert tensors["street_adj"][0, 1].item() == 8 * 60
    assert torch.isinf(tensors["street_adj"][0, 2])
    assert tensors["demand"][0, 1].item() == 400


def test_load_macsa_scenario_preserves_one_directional_routes():
    scenario = load_macsa_scenario(MANDL_8_DIR)
    routes = scenario["routes"]

    assert scenario["name"] == "mandl_8"
    assert routes.shape == (1, 8, 15)
    assert routes[0, 0, :6].tolist() == [3, 5, 2, 5, 14, 8]
    assert routes[0, 7, :3].tolist() == [0, 1, 2]
    assert routes[0, 7, 3:].eq(-1).all()


def test_load_macsa_scenarios_discovers_available_subdirectories():
    scenarios = load_macsa_scenarios(MACSA_ROOT)

    names = [scenario["name"] for scenario in scenarios]
    assert "mandl_8" in names


def test_get_dataset_from_config_supports_macsa_type():
    cfg = OmegaConf.create({
        "type": "macsa",
        "path": str(MANDL_8_DIR),
    })

    dataset = get_dataset_from_config(cfg)
    graph = dataset[0]

    assert graph.demand.shape == (15, 15)
    assert graph.street_adj.shape == (15, 15)
    assert graph.fixed_routes.shape == (0, 15)
