"""BeeColonySearchRun config-driven sweep (M010 stage 2).

The run loads its instance from ``cfg.data`` and improves it across the
``cfg.sweep`` alpha x adj_target grid, one seeded search per point, scoring each
into a rich ``SearchArtifact{table, routes, instance}``. Here the per-point
engine + the instance loader are faked so the test exercises the orchestration
(grid iteration, per-point knobs, metric selection, table assembly) without
running BCO or loading real data.
"""
import types
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
LIB_CFG = REPO / "connectpt" / "routes_generator" / "cfg"

from connectpt.routes_generator.data.sources import Instance
from connectpt.routes_generator.search.runs import BeeColonySearchRun


def _routes():
    R = torch.full((1, 2, 4), -1, dtype=torch.long)
    R[0, 0, :3] = torch.tensor([0, 1, 2])
    R[0, 1, :2] = torch.tensor([2, 3])
    return R


def _fake_instance():
    return Instance(label="Fake", tensors={"dummy": 1}, init_routes=_routes(),
                    spec={"n_routes": 2, "min_route_len": 2, "max_route_len": 4})


def test_beecolonysearchrun_sweep_drives_grid_and_builds_table(monkeypatch, tmp_path):
    with initialize_config_dir(config_dir=str(LIB_CFG), version_base=None):
        cfg = compose(config_name="experiments/seeded/our_nbco_mumford0")
    # attach the sweep + metric selection to the run cfg (stage 3 will ship
    # these as real YAML; here we drive run() directly).
    OmegaConf.set_struct(cfg, False)
    cfg.sweep = OmegaConf.create({"alpha": [0.0, 1.0], "adj_target": 0.3,
                                  "n_iterations": 5})
    cfg.metrics = ["RTT", "WMC"]

    seen = []

    class FakeRunner:
        def plan_summary(self):
            return {"counts": {}, "n_bees": 10}

        def run_seeded(self, init, tensors, *, eval_dims, n_iterations, alpha,
                       adj_target, adj_weight=None, sequential=None,
                       sum_writer=None):
            seen.append((alpha, adj_target, n_iterations, adj_weight))
            metrics = {"RTT": torch.tensor([1.0 + alpha]), "ATT": 2.0,
                       "median_connectivity_weighted": torch.tensor([0.5]),
                       "cost": 3.0}
            return _routes(), None, metrics

    run = BeeColonySearchRun(cfg)
    monkeypatch.setattr(BeeColonySearchRun, "setup", lambda self: None)
    monkeypatch.setattr(BeeColonySearchRun, "_load_instance",
                        lambda self: _fake_instance())
    run.runner = FakeRunner()
    run.context = types.SimpleNamespace(output_dir=tmp_path)

    art = run.run()

    # the YAML grid drove one seeded search per (alpha, adj_target), with the
    # sweep's n_iterations threaded through (no adj_weight override here).
    assert seen == [(0.0, 0.3, 5, None), (1.0, 0.3, 5, None)]

    # rich artifact: table + routes map + the loaded instance.
    assert art.instance.label == "Fake"
    assert list(art.table["alpha"]) == [0.0, 1.0]
    assert list(art.table["adj_target"]) == [0.3, 0.3]
    assert list(art.table["n_iterations"]) == [5, 5]
    # metric selection: only the requested metrics survive (ATT dropped).
    assert "RTT" in art.table.columns and "WMC" in art.table.columns
    assert "ATT" not in art.table.columns
    # initial network + one route set per swept point.
    assert "Initial" in art.routes
    assert sum(k != "Initial" for k in art.routes) == 2
    assert art.metadata["sweep"] is True and art.metadata["n_points"] == 2


def test_sweep_threads_adj_weight_override(monkeypatch, tmp_path):
    """cfg.sweep.adj_weight (0 = adjustment OFF, the E2 5-model ablation) is
    passed through to each seeded search."""
    with initialize_config_dir(config_dir=str(LIB_CFG), version_base=None):
        cfg = compose(config_name="experiments/seeded/our_nbco_mumford0")
    OmegaConf.set_struct(cfg, False)
    cfg.sweep = OmegaConf.create({"alpha": [0.0], "adj_target": 0.2,
                                  "adj_weight": 0.0, "n_iterations": 3})

    seen = []

    class FakeRunner:
        def plan_summary(self):
            return {"counts": {}}

        def run_seeded(self, init, tensors, *, eval_dims, n_iterations, alpha,
                       adj_target, adj_weight=None, sequential=None,
                       sum_writer=None):
            seen.append(adj_weight)
            return _routes(), None, {"RTT": torch.tensor([1.0]), "cost": 2.0}

    run = BeeColonySearchRun(cfg)
    monkeypatch.setattr(BeeColonySearchRun, "setup", lambda self: None)
    monkeypatch.setattr(BeeColonySearchRun, "_load_instance",
                        lambda self: _fake_instance())
    run.runner = FakeRunner()
    run.context = types.SimpleNamespace(output_dir=tmp_path)

    run.run()
    assert seen == [0.0]
