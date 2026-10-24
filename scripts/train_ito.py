"""Fit the Identify-Then-Optimize RC baseline from prepared transitions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from dfl_hvac.config import load_yaml
from dfl_hvac.training import HistoricalTransitions, TrainableRCParameters, fit_ito


def load_transitions(path: Path) -> HistoricalTransitions:
    with np.load(path) as data:
        required = {
            "indoor_temperature_c",
            "next_indoor_temperature_c",
            "ambient_temperature_c",
            "heating_power_kw",
            "cooling_power_kw",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"transition archive is missing arrays: {missing}")
        return HistoricalTransitions(
            indoor_temperature_c=data["indoor_temperature_c"],
            next_indoor_temperature_c=data["next_indoor_temperature_c"],
            ambient_temperature_c=data["ambient_temperature_c"],
            heating_power_kw=data["heating_power_kw"],
            cooling_power_kw=data["cooling_power_kw"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transitions", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/paper1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ito"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_yaml(args.config)
    hp = config["ito"]
    noise_divisor = float(config["dfl"]["parameter_noise_divisor"])
    transitions = load_transitions(args.transitions)
    arrays = transitions.arrays()
    n_zones = int(arrays[0].shape[1])

    model = TrainableRCParameters(n_zones, seed=args.seed)
    result = fit_ito(
        transitions,
        model=model,
        validation_fraction=float(hp["validation_fraction"]),
        batch_size=int(hp["batch_size"]),
        learning_rate=float(hp["learning_rate"]),
        max_epochs=int(hp["max_epochs"]),
        patience=int(hp["patience"]),
        seed=args.seed,
    )

    fitted_state = copy.deepcopy(result.model.state_dict())
    noisy_model = copy.deepcopy(result.model)
    torch.manual_seed(args.seed)
    noisy_model.add_relative_gaussian_noise(noise_divisor)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n_zones": n_zones,
            "seed": args.seed,
            "state_dict": fitted_state,
        },
        args.output_dir / "ito_fitted.pt",
    )
    torch.save(
        {
            "n_zones": n_zones,
            "seed": args.seed,
            "noise_divisor": noise_divisor,
            "state_dict": noisy_model.state_dict(),
        },
        args.output_dir / "ito_noisy_warm_start.pt",
    )

    summary = {
        "best_epoch": result.best_epoch,
        "best_validation_mse": min(result.validation_loss),
        "training_loss": list(result.training_loss),
        "validation_loss": list(result.validation_loss),
        "seed": args.seed,
        "zones": n_zones,
        "samples": int(arrays[0].shape[0]),
        "noise_divisor": noise_divisor,
    }
    (args.output_dir / "training.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_loss")}, indent=2))


if __name__ == "__main__":
    main()
