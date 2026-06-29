"""RouteModelFactory builds the right classes; capabilities validate actions."""
import pytest

from connectpt.routes_generator.model_factory import RouteModelFactory
from connectpt.routes_generator import model_capabilities as caps


def test_factory_builds_edit_model():
    model = RouteModelFactory.build_edit_model_by_name("edit_trim_paper")
    assert type(model).__name__ == "TrimPathCombiningRouteGenerator"
    assert caps.role_for_model(model) == "edit"
    assert caps.capabilities_for_model(model) == {"extend", "trim_start", "trim_end", "halt"}


def test_factory_builds_construction_model():
    model = RouteModelFactory.build_construction_model_by_name("construction")
    assert type(model).__name__ == "PathCombiningRouteGenerator"
    assert caps.role_for_model(model) == "construction"
    assert caps.capabilities_for_model(model) == {"extend", "halt"}


def test_factory_role_mismatch_raises():
    # edit config built as a construction model must fail loudly
    with pytest.raises(TypeError):
        RouteModelFactory.build_construction_model_by_name("edit_trim_paper")


def test_capabilities_reject_unsupported_action():
    # construction does not support trim
    with pytest.raises(ValueError, match="does not support"):
        caps.validate_actions("construction", ["extend", "trim_start"])
    # edit supports the full set
    caps.validate_actions("edit", ["extend", "trim_start", "trim_end", "halt"])


def test_capabilities_reject_unknown_action():
    with pytest.raises(ValueError, match="unknown action"):
        caps.validate_actions("edit", ["teleport"])
