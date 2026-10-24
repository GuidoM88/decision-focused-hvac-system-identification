"""Solve a six-hour, five-zone schedule with a published-data-trained network.

No data download or proprietary solver is needed. This demonstrates the learned
model's optimizer; it is not a physical-building or EnergyPlus evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dfl_hvac.neural import ReLUDynamics
from dfl_hvac.neural_control import NeuralScenario, solve_neural_hvac


def main():
    root = Path(__file__).resolve().parents[1]
    model = ReLUDynamics.from_dict(
        json.loads((root / "reports/public_data/nn2_seed0.json").read_text())
    )
    scenario = NeuralScenario(
        np.full(6, 5.0),
        np.full(5, 21.0),
        np.full(6, 0.3),
        heating_capacity_kw=3.0,
        cooling_capacity_kw=3.0,
        temperature_bounds_c=(10.0, 35.0),
    )
    print("Five-zone neural scheduling demo (learned-model optimization only)")
    objectives = []
    for mode in ("static", "weather", "reachable"):
        result = solve_neural_hvac(model, scenario, bound_mode=mode, time_limit=30)
        objectives.append(result.objective)
        print(json.dumps({
            "bounds": mode,
            "objective": round(result.objective, 6),
            "binaries": result.binary_variables,
            "interval_width": round(result.preactivation_width, 3),
            "status": result.status,
            "gap": result.relative_gap,
            "dynamics_residual": result.max_dynamics_residual,
        }))
    if np.ptp(objectives) > 1e-3 * max(1.0, abs(objectives[0])):
        raise RuntimeError("bound modes returned inconsistent certified objectives")


if __name__ == "__main__":
    main()
