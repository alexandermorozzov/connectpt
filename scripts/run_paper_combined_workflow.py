"""End-to-end paper workflow: train -> search -> evaluate -> report.

This is the ONLY entry point allowed to import training + search + evaluation +
reports together; the library layers themselves stay separate (training never
imports search and vice versa). Each stage is config-driven and optional.

    python scripts/run_paper_combined_workflow.py --dry-run
    python scripts/run_paper_combined_workflow.py --train --search --evaluate --report

By default every stage runs in --dry-run (build + validate wiring) so the glue
can be exercised without datasets / long compute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"


def _compose(name, overrides=None):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        return compose(config_name=name, overrides=overrides or [])


def run_workflow(*, train=False, search=False, evaluate=False, report=False,
                 dry_run=True) -> dict:
    # Imported here (not at module top) so importing this script does not couple
    # the layers; the workflow is the single outer glue point.
    from connectpt.routes_generator.training import EditTrainingRun
    from connectpt.routes_generator.search import BeeColonySearchRun
    from connectpt.routes_generator.evaluation import ModelEvaluationRun
    from connectpt.routes_generator.reports import ReportRun

    out: dict = {}
    if train:
        out["train"] = EditTrainingRun(_compose("training/edit")).run(dry_run=dry_run)
    if search:
        out["search"] = BeeColonySearchRun(
            _compose("search/bco_flexible_bees")).run(dry_run=dry_run)
    if evaluate:
        out["evaluate"] = ModelEvaluationRun(
            _compose("evaluation/edit_eval")).run(dry_run=dry_run)
    if report:
        out["report"] = ReportRun(_compose("evaluation/edit_eval")).run(dry_run=dry_run)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    # default: exercise the whole pipeline in dry-run
    stages = dict(train=args.train, search=args.search,
                  evaluate=args.evaluate, report=args.report)
    if not any(stages.values()):
        stages = {k: True for k in stages}
        args.dry_run = True

    results = run_workflow(dry_run=args.dry_run, **stages)
    for stage, artifact in results.items():
        print(f"[{stage}] {artifact.metadata}")


if __name__ == "__main__":
    main()
