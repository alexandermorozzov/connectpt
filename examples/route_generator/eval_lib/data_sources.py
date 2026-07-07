"""Deprecated shim: data sources moved to ``connectpt.routes_generator.data``.

Kept as a thin re-export so the notebook / eval_lib helpers keep importing
``eval_lib.data_sources`` until the notebook is migrated onto the library run
classes (M010 stage 7). No logic here -- the single implementation lives in the
library.
"""
from connectpt.routes_generator.data.sources import (  # noqa: F401
    BenchmarkDataSource,
    DataSource,
    EKBDataSource,
    Instance,
    MACSADataSource,
    create_data_source,
)
