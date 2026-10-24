"""Convert a simulator CSV into one-step RC training transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dfl_hvac.data_adapter import EnergyPlusColumnMap, historical_transitions_from_frame
from dfl_hvac.provenance import describe_external_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--indoor", nargs="+", required=True)
    parser.add_argument("--ambient", required=True)
    parser.add_argument("--heating", nargs="+", required=True)
    parser.add_argument("--cooling", nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/historical_transitions.npz"),
    )
    args = parser.parse_args()

    if not (len(args.indoor) == len(args.heating) == len(args.cooling)):
        raise ValueError("indoor, heating and cooling mappings must have equal lengths")

    frame = pd.read_csv(args.csv)
    transitions = historical_transitions_from_frame(
        frame,
        EnergyPlusColumnMap(
            indoor_temperature=tuple(args.indoor),
            ambient_temperature=args.ambient,
            heating_power=tuple(args.heating),
            cooling_power=tuple(args.cooling),
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        indoor_temperature_c=transitions.indoor_temperature_c,
        next_indoor_temperature_c=transitions.next_indoor_temperature_c,
        ambient_temperature_c=transitions.ambient_temperature_c,
        heating_power_kw=transitions.heating_power_kw,
        cooling_power_kw=transitions.cooling_power_kw,
    )

    source = describe_external_file(args.csv)
    metadata = {
        "source_csv": source.path,
        "source_sha256": source.sha256,
        "samples": int(transitions.indoor_temperature_c.shape[0]),
        "zones": int(transitions.indoor_temperature_c.shape[1]),
        "output": str(args.output),
        "columns": {
            "indoor": args.indoor,
            "ambient": args.ambient,
            "heating": args.heating,
            "cooling": args.cooling,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
