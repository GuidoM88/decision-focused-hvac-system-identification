"""Build Paper 1 representative-day artifacts from an EPW weather file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dfl_hvac.epw import epw_daily_temperature_profiles
from dfl_hvac.provenance import describe_external_file
from dfl_hvac.scenarios import build_hot_year, build_representative_day_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weather", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/weather_scenarios.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=100)
    args = parser.parse_args()

    profiles, labels = epw_daily_temperature_profiles(args.weather)
    split = build_representative_day_split(
        profiles,
        n_clusters=10,
        n_restarts=args.restarts,
        seed=args.seed,
    )
    hot_year = build_hot_year(profiles, split.labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        daily_profiles_c=profiles,
        day_labels=labels,
        cluster_labels=split.labels,
        training_indices=split.medoid_indices,
        training_order=split.training_order,
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
        hot_year_profiles_c=hot_year,
    )

    source = describe_external_file(args.weather)
    metadata = {
        "weather": source.path,
        "weather_sha256": source.sha256,
        "days": int(profiles.shape[0]),
        "seed": args.seed,
        "restarts": args.restarts,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
