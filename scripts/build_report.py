"""Build report tables from saved artifacts.

    python scripts/build_report.py --output-dir artifacts/runs/edit_eval_v1 --name edit_eval_v1

Reports read persisted artifacts only -- they do not import training/ or search/.
"""
from __future__ import annotations

import argparse

from connectpt.routes_generator.reports import load_evaluation_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    result = load_evaluation_result(args.output_dir, args.name)
    print("summary:")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
