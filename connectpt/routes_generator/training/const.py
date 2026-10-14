"""Constants owned by this package."""

from enum import StrEnum


class DatasetKind(StrEnum):
    CONSTRUCTION = "construction"
    EDIT = "edit"


class DataSourceKind(StrEnum):
    FILE = "file"
    DYNAMIC = "dynamic"
    BENCHMARK = "benchmark"


class PreferenceDistribution(StrEnum):
    SAMPLER = "sampler"
    SIMPLEX = "simplex"


DATASET_SCHEMA = 1
DATASET_MANIFEST = "manifest.json"
DATASET_CONTENT = "dataset.pt"
