"""Gate (M018 task F): the paper CLI scripts must NOT duplicate experiment logic.

A CLI runner is allowed to orchestrate ONLY through the same thin library surface
the notebooks use -- ``core.load_experiment`` / ``load_suite`` +
``paper_runs.run_experiment`` / ``run_batch`` / ``paper_dir`` -- plus CLI plumbing
(``_paper_cli``: logging + figure IO). Importing the search / bees / metrics /
sweep / report internals, or inlining a sweep/scoring loop, is duplication and
fails here. Each script must also drive the SAME config its notebook drives, so
"CLI and notebook share one logic" is checked by construction, not by convention.
"""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
NOTEBOOKS = REPO / "examples" / "route_generator"

# module -> the ONLY names a CLI script may import from connectpt (the surface
# the notebooks also call). Anything else = reaching into internals = duplication.
ALLOWED = {
    "connectpt.routes_generator.core": {"load_experiment", "load_suite"},
    "connectpt.routes_generator.paper_experiments.paper_runs":
        {"run_experiment", "run_batch", "paper_dir"},
}

CLI_SCRIPTS = ["run_ekb_case_study.py", "run_figure5_pareto.py",
               "run_table3.py", "run_table4.py"]

# script -> (the entrypoint it MUST call, the notebook that runs the same config)
EXPECT = {
    "run_ekb_case_study.py": ("run_experiment", "04_case_study.ipynb"),
    "run_figure5_pareto.py": ("run_batch", "03_nbco_experiments.ipynb"),
    "run_table3.py": ("run_batch", "03_nbco_experiments.ipynb"),
    "run_table4.py": ("run_experiment", "03_nbco_experiments.ipynb"),
}

# identifiers that only appear if a script re-implements experiment internals.
FORBIDDEN = [
    "BeeColonySearchRun", "run_sweep_table", "run_seeded", "full_metric_row",
    "select_metrics", "ExecutablePlan", "parse_bee_specs", "CostFactory",
    "SummaryWriter", "save_paper_table", "save_paper_routes", "run_bee_colony",
]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _connectpt_imports(tree: ast.Module) -> dict:
    """{module: {imported names}} for every connectpt import in the tree."""
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("connectpt"):
            out.setdefault(node.module, set()).update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("connectpt"):
                    out.setdefault(a.name, set())
    return out


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_imports_only_shared_surface(script):
    imports = _connectpt_imports(_tree(SCRIPTS / script))
    for module, names in imports.items():
        assert module in ALLOWED, (
            f"{script}: imports {module} -- not the notebook-shared surface")
        extra = names - ALLOWED[module]
        assert not extra, (
            f"{script}: imports {sorted(extra)} from {module} (not allowed -- "
            "experiment logic must stay in the library)")


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_calls_shared_entrypoint(script):
    entry, _ = EXPECT[script]
    called = {n.func.id for n in ast.walk(_tree(SCRIPTS / script))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert entry in called, f"{script}: must orchestrate via {entry}(...)"


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_has_no_internal_identifiers(script):
    text = (SCRIPTS / script).read_text(encoding="utf-8")
    hits = [tok for tok in FORBIDDEN if tok in text]
    assert not hits, f"{script}: contains internal identifiers {hits} (duplication)"


def test_paper_cli_plumbing_has_no_experiment_logic():
    # the shared CLI helper is pure logging + figure IO: nothing from connectpt.
    assert _connectpt_imports(_tree(SCRIPTS / "_paper_cli.py")) == {}


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_runs_same_config_as_notebook(script):
    _, notebook = EXPECT[script]
    tree = _tree(SCRIPTS / script)
    config = next(
        n.value.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
        and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
    nb_text = (NOTEBOOKS / notebook).read_text(encoding="utf-8")
    assert config in nb_text, (
        f"{script}: CONFIG={config!r} is not run by {notebook} -- CLI and "
        "notebook must drive the same experiment config (one shared logic)")
