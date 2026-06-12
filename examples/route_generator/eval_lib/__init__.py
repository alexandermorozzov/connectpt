"""eval_lib -- notebook-support package for paper_combined.ipynb.

Holds the helper functions, config builders, runners, plotting code and the
unified-objective parameters (:mod:`eval_lib.params`) the notebook uses. The
notebook imports everything via ``from eval_lib import *``.
"""
from .context import *  # noqa: F401,F403
from .params import *  # noqa: F401,F403
from .plots import *  # noqa: F401,F403
from .helpers import *  # noqa: F401,F403
from .baselines import *  # noqa: F401,F403
from .route_copies import *  # noqa: F401,F403
from .paper import *  # noqa: F401,F403
from .ekb import *  # noqa: F401,F403
from .results_io import *  # noqa: F401,F403
# private helpers the notebook still calls directly (skipped by ``*``):
from .baselines import _run_baseline  # noqa: F401
