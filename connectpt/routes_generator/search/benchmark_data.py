"""BenchmarkDataModule -- benchmark graphs/routes the search runs against.

A thin config holder today: it records the benchmark location and route-length
bounds. ``setup()`` is where benchmark loading is wired when the search runner
executes the full BCO; the dry-run path does not load data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.paths import DATASETS_DIR


@dataclass
class BenchmarkDataModule:
    benchmark_dirname: str
    min_route_len: int
    max_route_len: int
    graphs: list = field(default_factory=list, init=False)
    seed_routes: object = field(default=None, init=False)

    @property
    def benchmark_dir(self):
        return DATASETS_DIR / self.benchmark_dirname

    def setup(self) -> "BenchmarkDataModule":
        from ..improvement_learning import load_raw_graphs_and_lc_routes
        self.graphs, self.seed_routes = load_raw_graphs_and_lc_routes(
            self.benchmark_dir / "raw_graphs_subset.pkl", self.benchmark_dir
        )
        return self
