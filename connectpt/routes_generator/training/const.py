"""Constants owned by this package."""

from enum import StrEnum


class DatasetKind(StrEnum):
    CONSTRUCTION = "construction"
    EDIT = "edit"


DATASET_SCHEMA = 1
DATASET_MANIFEST = "manifest.json"
DATASET_CONTENT = "dataset.pt"
