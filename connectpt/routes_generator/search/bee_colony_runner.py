"""BeeColonyRunner -- drive the existing bee_colony from a translated plan.

The runner holds the cost, the loaded models, the policy adapters and the
``BeeColonyPlan`` (per-type bee counts). ``run_suite`` is the integration point
that calls the existing ``bee_colony(...)`` with those counts + models; it is
invoked only on a real (non-dry) run, which needs a loaded benchmark state.

This layer must NOT import the training application.
"""
from __future__ import annotations


class BeeColonyRunner:
    def __init__(self, cfg, *, cost_obj, models: dict, policies: dict, plan,
                 data_module):
        self.cfg = cfg
        self.cost_obj = cost_obj
        self.models = models
        self.policies = policies
        self.plan = plan
        self.data = data_module

    def plan_summary(self) -> dict:
        """The translated per-type counts + which models are needed."""
        return {
            "counts": dict(self.plan.counts),
            "needs_construction": self.plan.needs_construction,
            "needs_edit": self.plan.needs_edit,
            "n_bees": int(self.cfg.get("n_bees", sum(self.plan.counts.values()))),
        }

    def run_suite(self):
        """Run the full bee-colony search (real run only).

        Wires the translated bee counts + loaded models into the existing
        bee_colony executor. Requires ``self.data.setup()`` to have loaded the
        benchmark state.
        """
        from ..bee_colony import bee_colony  # local import: heavy, real-run only

        if not self.data.graphs:
            self.data.setup()
        raise NotImplementedError(
            "Full bee_colony execution wiring is finalized alongside the notebook "
            "search cells (C10); the dry-run path validates models/policies/plan. "
            f"bee_colony={bee_colony.__name__}, plan={self.plan_summary()}"
        )
