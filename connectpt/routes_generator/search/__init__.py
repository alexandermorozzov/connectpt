"""Search application: flexible bee-colony search over route models.

Decouples *what a bee does* (operator) from *which model it uses* (policy
adapter) from *which actions it may take* (allowed_actions), so one neural
operator works across construction / edit models. Must NOT import the training
layer (the two are separate applications joined only by checkpoints + configs).
"""

from .bee_specs import BeeSpec, parse_bee_specs
from .search_policies import (
    SearchPolicyAdapter,
    ConstructionSearchPolicy,
    EditSearchPolicy,
    build_policies,
)
from .bee_colony_runner import BeeColonyRunner
from .runs import BeeColonySearchRun, SearchArtifact


__all__ = [
    "BeeSpec",
    "parse_bee_specs",
    "SearchPolicyAdapter",
    "ConstructionSearchPolicy",
    "EditSearchPolicy",
    "build_policies",
    "BeeColonyRunner",
    "BeeColonySearchRun",
    "SearchArtifact",
]
