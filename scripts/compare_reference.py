"""Compare a JSON metric dictionary with a Paper 1 table row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dfl_hvac.comparison import compare_metrics
from dfl_hvac.reference_results import TABLE_1, TABLE_2_HOT_YEAR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--method", choices=["ito", "dfl"], required=True)
    parser.add_argument("--split", choices=["train", "validation", "test", "hot_year"], required=True)
    args = parser.parse_args()

    observed = json.loads(args.metrics.read_text(encoding="utf-8"))
    if args.split == "hot_year":
        reference = TABLE_2_HOT_YEAR[args.method]
    else:
        reference = TABLE_1[args.method][args.split]

    print(json.dumps(compare_metrics(observed, reference), indent=2))


if __name__ == "__main__":
    main()
