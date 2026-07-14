"""The eval/ registry is really used, and kept honest.

Every data source reads its route-count / length spec from ``cfg/eval/<name>.yaml``
(benchmarks by city via ``benchmark_spec``; EKB / MACSA by dataset name via
``eval_spec`` -- see ``data/sources.py``). For the case studies those registry
values are now AUTHORITATIVE (the source reads them at load time), so these tests
guard them against silent drift from the seed data they describe.
"""
import pytest
from omegaconf import OmegaConf

from connectpt.routes_generator.data.loaders import (
    EKB_DATA_DIR, EVAL_DIR, benchmark_spec, ekb_spec, eval_spec,
    macsa_eval_bounds)


def test_benchmark_source_spec_comes_from_eval_yaml():
    # BenchmarkDataSource.load() builds spec from benchmark_spec(city); assert the
    # reader returns exactly the YAML values (eval/ is the runtime source).
    raw = OmegaConf.load(EVAL_DIR / "mumford1.yaml")
    spec = benchmark_spec("Mumford1")
    assert (spec["n_routes"], spec["min_route_len"], spec["max_route_len"]) == (
        int(raw.n_routes), int(raw.min_route_len), int(raw.max_route_len))


def test_ekb_registry_matches_seed_data():
    if not EKB_DATA_DIR.exists():
        pytest.skip("EKB data absent")
    derived = ekb_spec()
    assert eval_spec("ekb") == {
        k: derived[k] for k in ("n_routes", "min_route_len", "max_route_len")}


def test_macsa_registry_matches_scenario_data():
    from connectpt.routes_generator.core.paths import MACSA_DATA_DIR
    scenario_dir = MACSA_DATA_DIR / "mandl_8"
    if not scenario_dir.exists():
        pytest.skip("MACSA scenario data absent")
    from connectpt.routes_generator.citygraph_dataset import load_macsa_scenario
    n_routes, min_len, max_len = macsa_eval_bounds(load_macsa_scenario(scenario_dir))
    assert eval_spec("macsa_mandl_8") == {
        "n_routes": n_routes, "min_route_len": min_len, "max_route_len": max_len}
