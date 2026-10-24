"""Independent identification on two published EnergyPlus years (not a control claim)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dfl_hvac.neural import fit_neural_ito
from dfl_hvac.public_data import (
    author_transitions,
    download_author_data,
    input_overlap_mask,
    weekly_validation_masks,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/public_data"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(1)
    train_path, train_source = download_author_data("1991-2020")
    test_path, test_source = download_author_data("2006-2020")
    x, y, index = author_transitions(pd.read_csv(train_path))
    xt, yt, _ = author_transitions(pd.read_csv(test_path))
    # Every fifth week is validation, with a one-row embargo on either side.
    # All months remain represented; the second weather year is untouched test.
    train, val = weekly_validation_masks(index)
    overlap = input_overlap_mask(x, xt)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for width in (2, 5, 10):
            net, fit = fit_neural_ito(
                x[train],
                y[train],
                x[val],
                y[val],
                hidden_sizes=(width,),
                seed=seed,
                max_epochs=args.epochs,
            )
            prediction = net.predict(xt)
            rows.append(
                {
                    "seed": seed,
                    "hidden_units": width,
                    "train_rows": int(train.sum()),
                    "validation_rows": int(val.sum()),
                    "test_rows": len(xt),
                    "test_rmse_c": float(np.sqrt(np.mean((prediction - yt) ** 2))),
                    "test_mae_c": float(np.mean(np.abs(prediction - yt))),
                    "persistence_rmse_c": float(np.sqrt(np.mean((xt[:, :5] - yt) ** 2))),
                    "disjoint_input_rows": int((~overlap).sum()),
                    "disjoint_input_rmse_c": float(
                        np.sqrt(np.mean((prediction[~overlap] - yt[~overlap]) ** 2))
                    ),
                    "disjoint_persistence_rmse_c": float(
                        np.sqrt(np.mean((xt[~overlap, :5] - yt[~overlap]) ** 2))
                    ),
                    "best_epoch": fit["best_epoch"],
                }
            )
            (args.output / f"nn{width}_seed{seed}.json").write_text(
                json.dumps(net.to_dict(), indent=2)
            )
            print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(args.output / "identification.csv", index=False)
    metadata = {
        "scope": "cross-weather-year one-step prediction; not closed-loop control",
        "sources": [train_source, test_source],
        "split": "held-out weeks with one-row embargo",
        "epochs": args.epochs,
        "seeds": args.seeds,
        "normalization": "training rows only",
        "time_alignment": "authors row-t convention",
        "units": "temperature Celsius; hourly Wh / 1000 -> mean kW",
        "weather_files": "two typical meteorological years, with overlapping source periods",
        "test_input_collisions_with_first_weather_file": int(overlap.sum()),
        "disjoint_diagnostic": "exclude exact input matches to any first-file input; no refitting",
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
