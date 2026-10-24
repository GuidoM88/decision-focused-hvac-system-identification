"""Paired ablation: static, weather-conditioned, and recursive ReLU bounds.

Uses the same model and feasible domain for each pair. Timeouts remain in the
output. Solution equivalence is assessed only for certified optimal solves.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from dfl_hvac.neural import ReLUDynamics
from dfl_hvac.neural_control import NeuralScenario, solve_neural_hvac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("reports/public_data"))
    parser.add_argument("--output", type=Path, default=Path("reports/bounds"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--time-limit", type=float, default=10.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for width in args.widths:
            model_path = args.models / f"nn{width}_seed{seed}.json"
            network = ReLUDynamics.from_dict(json.loads(model_path.read_text()))
            for horizon in args.horizons:
                for mean in (5.0, 20.0, 32.0):
                    hours = np.arange(horizon)
                    s = NeuralScenario(
                        mean + 3 * np.sin(2 * np.pi * hours / 24),
                        np.full(5, 21.0),
                        np.where((hours >= 6) & (hours < 19), 0.6, 0.3),
                        heating_capacity_kw=3.0,
                        cooling_capacity_kw=3.0,
                        discomfort_weight=1.0,
                        temperature_bounds_c=(10.0, 35.0),
                        name=f"{mean:g}C_{horizon}h",
                    )
                    modes = list(
                        np.random.default_rng(
                            seed * 100 + width * 10 + horizon + int(mean)
                        ).permutation(["static", "weather", "reachable"])
                    )
                    for mode in modes:
                        started = time.perf_counter()
                        row = {
                            "seed": seed,
                            "width": width,
                            "horizon": horizon,
                            "mean_c": mean,
                            "mode": mode,
                        }
                        try:
                            result = solve_neural_hvac(
                                network,
                                s,
                                bound_mode=mode,
                                time_limit=args.time_limit,
                                relative_gap=1e-6,
                                require_optimal=False,
                            )
                            row.update(
                                {
                                    k: v
                                    for k, v in asdict(result).items()
                                    if not isinstance(v, np.ndarray)
                                }
                            )
                            row["certified"] = (
                                result.status == "optimal" or result.relative_gap <= 1e-6
                            )
                        except (RuntimeError, ValueError) as exc:
                            row.update(status="failed", error=str(exc), certified=False)
                        row["wall_seconds"] = time.perf_counter() - started
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(args.output / "runs.csv", index=False)
                        print(json.dumps(row), flush=True)
    frame = pd.DataFrame(rows)
    paired = []
    for key, group in frame.groupby(["seed", "width", "horizon", "mean_c"]):
        modes = group.set_index("mode")
        a, b = modes.loc["weather"], modes.loc["reachable"]
        both = bool(a.certified and b.certified)
        gap = abs(a.get("objective", np.nan) - b.get("objective", np.nan))
        paired.append(
            dict(zip(["seed", "width", "horizon", "mean_c"], key))
            | {
                "both_certified": both,
                "objective_difference": gap,
                "binary_reduction": a.get("binary_variables", np.nan)
                - b.get("binary_variables", np.nan),
                "width_reduction_fraction": 1
                - b.get("preactivation_width", np.nan) / a.get("preactivation_width", np.nan),
                "time_ratio_reachable_over_weather": b.wall_seconds / a.wall_seconds,
            }
        )
        if both and gap > 1e-3 * max(1, abs(a.objective)):
            raise AssertionError(f"objective equivalence failed: {key}, gap={gap}")
    pd.DataFrame(paired).to_csv(args.output / "paired.csv", index=False)
    metadata = {
        "scope": "optimization-only ablation on publicly trained networks",
        "temperature_domain": [10, 35],
        "hvac_capacity_kw": 3,
        "seeds": args.seeds,
        "widths": args.widths,
        "horizons": args.horizons,
        "time_limit": args.time_limit,
        "means_c": [5, 20, 32],
        "same_domain_and_objective": True,
        "mode_order": "deterministically randomized within each paired instance",
        "runtime_note": "one timing per instance, no universal speedup claim",
        "equivalence_tolerance": "absolute objective difference <= 1e-3 * max(1, |baseline|)",
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
