"""Isolate RC warm-start perturbation from DFL with validation rollback."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from run_benchmark import rc_example, response, split_scenarios

from dfl_hvac.benchmark_system import NonlinearThermalSystem, synthetic_history
from dfl_hvac.dfl_training import decision_focused_loss, train_dfl
from dfl_hvac.neural_control import expost_metrics
from dfl_hvac.training import HistoricalTransitions, fit_ito


def main():
    torch.set_num_threads(1)
    output = Path("reports/rc_initialization")
    output.mkdir(parents=True, exist_ok=True)
    plant = NonlinearThermalSystem()
    training, validation, test, hot = split_scenarios(8)
    train_examples = [rc_example(s, plant) for s in training]
    val_examples = [rc_example(s, plant) for s in validation]
    rows, curves = [], []
    for seed in (0, 1, 2):
        x, y = synthetic_history(plant, seed=seed)
        transitions = HistoricalTransitions(x[:, :1], y, x[:, -1], np.maximum(x[:, 1:2], 0), np.maximum(-x[:, 1:2], 0))
        ito = fit_ito(transitions, seed=seed, max_epochs=150, learning_rate=.002).model.double()
        for initialization in ("unperturbed", "perturbed"):
            warm = copy.deepcopy(ito)
            if initialization == "perturbed":
                torch.manual_seed(seed)
                warm.add_relative_gaussian_noise(divisor=25.)
            initial_validation = float(np.mean([float(decision_focused_loss(warm, example).detach()) for example in val_examples]))
            fit = train_dfl(copy.deepcopy(warm), train_examples, val_examples, max_epochs=8, patience=8, learning_rate=1e-4)
            keep_warm = initial_validation <= min(fit.validation_loss)
            selected = warm if keep_warm else fit.model
            curves.append(dict(seed=seed, initialization=initialization, initial_validation_loss=initial_validation,
                training_loss=fit.training_loss, validation_loss=fit.validation_loss,
                selected_epoch=-1 if keep_warm else fit.best_epoch))
            for label, model in (("warm", warm), ("DFL-selected", selected)):
                serialized = {name: value.detach().numpy().tolist() for name, value in model.state_dict().items()}
                serialized["dt_hours"] = model.dt_hours
                (output / f"{initialization}_{label}_seed{seed}.json").write_text(json.dumps(serialized, indent=2))
                for split, days in (("test", test), ("hot", hot)):
                    for scenario in days:
                        solution = rc_example(scenario, plant).layer(model)
                        plan = SimpleNamespace(indoor_temperature_c=solution.indoor_temperature_c.detach().numpy(), hvac_power_kw=solution.hvac_power_kw.detach().numpy())
                        values = expost_metrics(plan, response(plant, plan, scenario), scenario)
                        rows.append(dict(seed=seed, initialization=initialization, method=label, split=split, day=scenario.name, **values))
            print(f"seed={seed} initialization={initialization} selected_epoch={curves[-1]['selected_epoch']}", flush=True)
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        (output / "learning_curves.json").write_text(json.dumps(curves, indent=2))
    summary = pd.DataFrame(rows).groupby(["initialization", "method", "split"]).expost_plus.agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    (output / "manifest.json").write_text(json.dumps(dict(scope="synthetic one-zone RC diagnostic; not EnergyPlus",
        seeds=[0, 1, 2], horizon=8, epochs=8, learning_rate=.0001, perturbation="relative Gaussian, divisor 25",
        checkpoint="minimum validation hierarchical power loss, including epoch -1 warm start",
        baseline_difference="original benchmark selects among trained epochs only; this diagnostic also permits rollback"), indent=2))
    print(summary.to_string())


if __name__ == "__main__":
    main()
