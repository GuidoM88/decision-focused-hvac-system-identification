"""ARX/Ridge and passive RC on the same public-data split as the neural models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dfl_hvac.linear_baselines import fit_passive_rc, fit_ridge
from dfl_hvac.public_data import (
    author_transitions,
    download_author_data,
    input_overlap_mask,
    weekly_validation_masks,
)


def main():
    output = Path("reports/prediction_baselines")
    output.mkdir(parents=True, exist_ok=True)
    first, source1 = download_author_data("1991-2020")
    second, source2 = download_author_data("2006-2020")
    x, y, index = author_transitions(pd.read_csv(first))
    xt, yt, _ = author_transitions(pd.read_csv(second))
    train, validation = weekly_validation_masks(index)
    disjoint = ~input_overlap_mask(x, xt)
    rows, selections = [], {}
    for name, fit in (("ARX-Ridge", fit_ridge), ("Passive-RC", fit_passive_rc)):
        model, selection = fit(x[train], y[train], x[validation], y[validation])
        prediction = model.predict(xt)
        row = dict(method=name, test_rmse_c=float(np.sqrt(np.mean((prediction - yt)**2))),
            disjoint_rmse_c=float(np.sqrt(np.mean((prediction[disjoint] - yt[disjoint])**2))),
            disjoint_mae_c=float(np.mean(np.abs(prediction[disjoint] - yt[disjoint]))),
            validation_rmse_c=selection["selected"]["validation_mse"]**.5)
        rows.append(row)
        selections[name] = selection
        (output / f"{name}.json").write_text(json.dumps(model.to_dict(), indent=2))
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(dict(sources=[source1, source2],
        scope="one-step prediction; deterministic fits; same public-data masks and units as neural study",
        training_rows=int(train.sum()), validation_rows=int(validation.sum()), test_rows=len(xt),
        disjoint_rows=int(disjoint.sum()), selection=selections), indent=2))


if __name__ == "__main__":
    main()
