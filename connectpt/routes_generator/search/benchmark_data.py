"""BenchmarkDataModule -- the benchmark city the search runs against.

Loads a standard benchmark instance (Mandl / Mumford0-3) straight from the
``datasets/benchmark/<city>*.txt`` files via the connectpt loader -- no eval_lib
dependency. ``setup()`` builds the single-graph dataset the bee-colony runner
feeds through ``test_method``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.paths import BENCHMARK_DIR


@dataclass
class BenchmarkDataModule:
    city: str
    n_routes: int
    min_route_len: int
    max_route_len: int
    dataset: list = field(default_factory=list, init=False)

    def setup(self) -> "BenchmarkDataModule":
        from omegaconf import OmegaConf
        from ..citygraph_dataset import get_dataset_from_config

        ds_cfg = OmegaConf.create(
            {"type": "mumford", "path": str(BENCHMARK_DIR), "city": self.city}
        )
        self.dataset = get_dataset_from_config(ds_cfg)
        return self
