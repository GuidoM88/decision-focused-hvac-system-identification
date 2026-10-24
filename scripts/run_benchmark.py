"""Portable, paired method benchmark. Synthetic results are not paper table replications."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dfl_hvac.benchmark_system import NonlinearThermalSystem, synthetic_history
from dfl_hvac.dfl_training import DFLExample, train_dfl
from dfl_hvac.differentiable import DifferentiableHVACLayer
from dfl_hvac.neural import fit_neural_ito
from dfl_hvac.neural_control import NeuralScenario, expost_metrics, solve_neural_hvac
from dfl_hvac.neural_dfl import fit_neural_dfl
from dfl_hvac.smoothing import fit_smoothing
from dfl_hvac.training import HistoricalTransitions, fit_ito


def scenario(mean, phase, horizon, name):
    hours = np.arange(horizon)
    return NeuralScenario(
        mean + 3 * np.sin(2 * np.pi * hours / 24 + phase),
        np.array([21.0]),
        np.where((hours % 24 >= 6) & (hours % 24 < 19), 0.6, 0.3),
        heating_capacity_kw=12.0,
        cooling_capacity_kw=12.0,
        discomfort_weight=2.0,
        name=name,
    )


def split_scenarios(horizon):
    training = [
        scenario(a, p, horizon, f"train{i}")
        for i, (a, p) in enumerate(((5.0, 0.0), (18.0, 0.4), (29.0, 0.8)))
    ]
    validation = [
        scenario(a, p, horizon, f"val{i}") for i, (a, p) in enumerate(((10.0, 0.3), (25.0, 1.0)))
    ]
    test = [
        scenario(a, p, horizon, f"test{i}")
        for i, (a, p) in enumerate(((1.0, 0.2), (14.0, 0.8), (23.0, 0.5), (32.0, 1.2)))
    ]
    hot = [
        replace(s, ambient_temperature_c=s.ambient_temperature_c + 4, name=s.name + "_hot")
        for s in test
    ]
    return training, validation, test, hot


def response(plant, schedule, s):
    return plant.run(
        temperature_setpoints_c=schedule.indoor_temperature_c[1:],
        ambient_temperature_c=s.ambient_temperature_c,
        initial_temperature_c=s.initial_temperature_c,
    )


def rc_example(s, plant):
    layer = DifferentiableHVACLayer(
        ambient_temperature_c=s.ambient_temperature_c,
        initial_temperature_c=s.initial_temperature_c,
        energy_price_eur_per_kwh=s.energy_price,
        demand_charge_eur_per_kw=s.demand_charge,
        temperature_target_c=s.target_c,
        discomfort_weight=s.discomfort_weight,
        zone_heating_capacity_kw=s.heating_capacity_kw,
        zone_cooling_capacity_kw=s.cooling_capacity_kw,
        floor_groups=[[0]],
        floor_heating_capacity_kw=s.heating_capacity_kw,
        floor_cooling_capacity_kw=s.cooling_capacity_kw,
        import_capacity_kw=s.grid_capacity_kw,
    )
    return DFLExample(
        s.name,
        layer,
        plant,
        s.ambient_temperature_c,
        s.initial_temperature_c,
        s.energy_price,
        s.demand_charge,
        ((0,),),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--pretrain-epochs", type=int, default=250)
    parser.add_argument("--skip-surrogates", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip fully saved seeds with matching settings")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    train, val, test, hot = split_scenarios(args.horizon)
    plant = NonlinearThermalSystem()
    manifest = {
        "scope": "synthetic methodological benchmark; no EnergyPlus table claim",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "horizon": args.horizon,
        "skip_surrogates": args.skip_surrogates,
        "zones": 1,
        "ss_samples_per_update": 2,
        "sigma": 0.01,
        "test_selection": "predeclared days, never used for checkpoint selection",
        "stress": "test ambient +4 C; separate from Paper 1 hot-year construction",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "torch", "cvxpy", "cvxpylayers", "pyscipopt")
        },
    }
    rows, training, completed = [], [], set()
    manifest_path = args.output / "manifest.json"
    if args.resume:
        old = json.loads(manifest_path.read_text())
        for key in ("seeds", "epochs", "pretrain_epochs", "horizon", "skip_surrogates", "versions"):
            if old.get(key) != manifest[key]:
                raise ValueError(f"resume setting differs: {key}")
        manifest = old | manifest
        if (args.output / "metrics.csv").exists():
            existing = pd.read_csv(args.output / "metrics.csv")
            expected_methods = {"NN-ITO", "NN-SS", "NN-AS", "RC-ITO", "RC-warm", "RC-DFL"}
            if not args.skip_surrogates:
                expected_methods |= {"NN-QP", "NN-FB"}
            expected_keys = {(m, split, s.name) for m in expected_methods
                             for split, days in (("test", test), ("hot", hot)) for s in days}
            for seed, group in existing.groupby("seed"):
                keys = list(zip(group.method, group.split, group.day))
                if set(keys) != expected_keys or len(keys) != len(expected_keys):
                    raise ValueError(f"incomplete or duplicate saved outcomes for seed {seed}")
                completed.add(int(seed))
            if not completed.issubset(set(args.seeds)):
                raise ValueError("saved outcomes contain undeclared seeds")
            rows = existing.to_dict("records")
            training = json.loads((args.output / "training.json").read_text())
            for seed in completed:
                methods = [r["method"] for r in training if r["seed"] == seed]
                expected_logs = expected_methods - {"NN-ITO", "RC-ITO", "RC-warm"}
                if set(methods) != expected_logs or len(methods) != len(expected_logs):
                    raise ValueError(f"incomplete or duplicate training logs for seed {seed}")
    manifest["completed_seeds"] = sorted(completed)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    x_eval, y_eval = synthetic_history(plant, seed=90000, samples=1000)
    for seed in args.seeds:
        if seed in completed:
            print(f"Resumed completed seed {seed}", flush=True)
            continue
        x, y = synthetic_history(plant, seed=seed)
        nn, pretrain = fit_neural_ito(
            x[:1800],
            y[:1800],
            x[1800:],
            y[1800:],
            seed=seed,
            max_epochs=args.pretrain_epochs,
            hidden_sizes=(5,),
        )
        models = {"NN-ITO": nn}
        (args.output / f"nn_ito_seed{seed}.json").write_text(json.dumps(nn.to_dict(), indent=2))

        def objective(s):
            def loss(vector):
                net = nn.with_vector(vector)
                plan = solve_neural_hvac(net, s)
                return expost_metrics(plan, response(plant, plan, s), s)["expost_plus"]

            return loss

        for estimator, label in (("score", "NN-SS"), ("antithetic", "NN-AS")):
            start = time.perf_counter()
            fit = fit_smoothing(
                nn.vector(),
                [objective(s) for s in train],
                [objective(s) for s in val],
                seed=seed,
                max_epochs=args.epochs,
                patience=args.epochs,
                samples=2,
                sigma=0.01,
                estimator=estimator,
            )
            models[label] = nn.with_vector(fit.parameters)
            log = asdict(fit)
            del log["parameters"]
            training.append(
                {"seed": seed, "method": label, "seconds": time.perf_counter() - start, **log}
            )
            print(
                json.dumps({"seed": seed, "method": label, "best_epoch": fit.best_epoch}),
                flush=True,
            )
        if not args.skip_surrogates:
            for method, label in (("qp", "NN-QP"), ("fb", "NN-FB")):
                start = time.perf_counter()
                net, fit = fit_neural_dfl(
                    nn,
                    [(s, plant) for s in train],
                    [(s, plant) for s in val],
                    method=method,
                    max_epochs=args.epochs,
                    patience=args.epochs,
                )
                models[label] = net
                training.append(
                    {"seed": seed, "method": label, "seconds": time.perf_counter() - start, **fit}
                )
                print(
                    json.dumps({"seed": seed, "method": label, "best_epoch": fit["best_epoch"]}),
                    flush=True,
                )
        for label, net in models.items():
            prediction_mse = float(np.mean((net.predict(x_eval) - y_eval) ** 2))
            for split, days in (("test", test), ("hot", hot)):
                for s in days:
                    plan = solve_neural_hvac(net, s)
                    metrics = expost_metrics(plan, response(plant, plan, s), s)
                    rows.append(
                        {
                            "seed": seed,
                            "method": label,
                            "split": split,
                            "day": s.name,
                            "prediction_mse_c2": prediction_mse,
                            **metrics,
                        }
                    )

        transitions = HistoricalTransitions(
            x[:, :1], y, x[:, -1], np.maximum(x[:, 1:2], 0), np.maximum(-x[:, 1:2], 0)
        )
        ito = fit_ito(transitions, seed=seed, max_epochs=150, learning_rate=0.002).model.double()
        warm = copy.deepcopy(ito)
        torch.manual_seed(seed)
        warm.add_relative_gaussian_noise()
        dfl = train_dfl(
            copy.deepcopy(warm),
            [rc_example(s, plant) for s in train],
            [rc_example(s, plant) for s in val],
            max_epochs=args.epochs,
            patience=args.epochs,
            learning_rate=1e-4,
        )
        training.append(
            {
                "seed": seed,
                "method": "RC-DFL",
                "best_epoch": dfl.best_epoch,
                "validation_loss": dfl.validation_loss,
            }
        )
        for label, model in (("RC-ITO", ito), ("RC-warm", warm), ("RC-DFL", dfl.model)):
            with torch.no_grad():
                pred = model(
                    x_eval[:, :1],
                    x_eval[:, -1],
                    np.maximum(x_eval[:, 1:2], 0),
                    np.maximum(-x_eval[:, 1:2], 0),
                ).numpy()
            for split, days in (("test", test), ("hot", hot)):
                for s in days:
                    plan = rc_example(s, plant).layer(model)

                    # Convert the layer result without using the observed plant in backpropagation.
                    class NumericPlan:
                        pass

                    numeric = NumericPlan()
                    numeric.indoor_temperature_c = plan.indoor_temperature_c.detach().numpy()
                    numeric.hvac_power_kw = plan.hvac_power_kw.detach().numpy()
                    metrics = expost_metrics(numeric, response(plant, numeric, s), s)
                    rows.append(
                        {
                            "seed": seed,
                            "method": label,
                            "split": split,
                            "day": s.name,
                            "prediction_mse_c2": float(np.mean((pred - y_eval) ** 2)),
                            **metrics,
                        }
                    )
        pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
        (args.output / "training.json").write_text(json.dumps(training, indent=2))
        completed.add(seed)
        manifest["completed_seeds"] = sorted(completed)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Completed seed {seed}", flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["method", "split"]).mean(numeric_only=True).drop(columns="seed")
    summary.to_csv(args.output / "summary.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
