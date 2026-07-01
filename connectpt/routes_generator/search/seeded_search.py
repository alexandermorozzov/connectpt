"""Seeded bee-colony search -- the library-owned execution contract.

paper_combined seeds BCO from a concrete initial network (LC construction +
realistic-tier corruption for the benchmarks, or the paper's routes for
MACSA / EKB) and improves it -- it does NOT build routes from scratch. This
entry point encodes exactly that: run ``bee_colony`` through ``test_method`` with
``init_cfg={"method": "tensor"}`` + the provided ``init_routes`` as the starting
network.

It takes the dataloader, cost and (optional) loaded models the caller built and
owns only the run itself: translate the search cfg into bee_colony kwargs (via
:func:`build_bee_colony_kwargs`) and drive the executor. Behaviour is identical
to the inline ``test_method(bee_colony, ...)`` call ``eval_lib.run_bco`` used, so
the seeded golden outputs are reproduced bit-for-bit.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from ..bee_colony import bee_colony
from ..utils import test_method
from .bco_invocation import build_bee_colony_kwargs

# The seeded BCO always starts from the provided routes tensor (not a
# from-scratch heuristic like "john"); this is the paper_combined contract.
_SEEDED_INIT_CFG = {"method": "tensor"}


def run_seeded_bee_colony(dataloader, eval_cfg, cost_obj, init_routes, *,
                          search_cfg, bee_model=None, edit_model=None,
                          mutation_counts_out=None, device=None,
                          silent=False, return_histories=False,
                          iteration_callback=None):
    """Run a seeded bee-colony search and return the raw ``test_method`` output.

    ``search_cfg`` is the composed BCO config (bee counts + schedule + adjustment
    block) -- translated to bee_colony kwargs by :func:`build_bee_colony_kwargs`.
    ``init_routes`` is the starting network the search improves. The return value
    is exactly what ``test_method`` returns (with ``return_routes=True`` and, when
    ``return_histories`` is set, the trailing cost-history list).
    """
    bco_kwargs = build_bee_colony_kwargs(
        search_cfg, bee_model=bee_model, edit_model=edit_model,
        mutation_counts_out=mutation_counts_out)
    return test_method(
        bee_colony,
        dataloader,
        eval_cfg,
        OmegaConf.create(_SEEDED_INIT_CFG),
        cost_obj,
        silent=silent,
        device=device,
        return_routes=True,
        return_histories=return_histories,
        routes_tensor=init_routes,
        iteration_callback=iteration_callback,
        **bco_kwargs,
    )
