"""Five-zone, 24-hour control pilot on the pinned EnergyPlus SmallOffice.

Independent protocol, not original-paper tables. Retains gas, solver failures,
native warnings and raw trajectories. Test days never select checkpoints.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from dfl_hvac.energyplus_runtime import run_energyplus_day
from dfl_hvac.linear_baselines import LinearDynamics, solve_linear_hvac
from dfl_hvac.neural import ReLUDynamics
from dfl_hvac.neural_control import NeuralScenario, solve_neural_hvac
from dfl_hvac.public_data import ZONES, author_transitions, weekly_validation_masks
from dfl_hvac.smoothing import fit_smoothing

TRAIN = ["2017-01-10", "2017-07-11"]
VALIDATION = ["2017-04-11", "2017-10-10"]
TEST = ["2017-01-17", "2017-04-18", "2017-07-18", "2017-10-17"]
HOURS = np.arange(24)
PRICE = np.where((HOURS >= 16) & (HOURS < 20), .65, np.where((HOURS >= 7) & (HOURS < 16), .45, .25))
OCCUPIED = (HOURS >= 8) & (HOURS < 18)
WEIGHT = np.where(OCCUPIED, 1., .1)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def metrics(result, plan):
    electric, gas = np.array(result["electricity_kwh"]), np.array(result["gas_kwh"])
    temperature = np.array(result["temperature_c"])[1:]
    fine = np.array([row["temperature_c"] for row in result["quarter_hour"]])
    lower = np.repeat(np.where(OCCUPIED, 20., 18.), 4)[:, None]
    upper = np.repeat(np.where(OCCUPIED, 24., 26.), 4)[:, None]
    violation = np.maximum(lower - fine, 0) + np.maximum(fine - upper, 0)
    electricity_bill, gas_bill = float(PRICE @ electric.sum(1)), float(.10 * gas.sum())
    discomfort = float(np.sum(WEIGHT[:, None] * (temperature - 21.)**2))
    values = dict(electricity_kwh=float(electric.sum()), gas_kwh=float(gas.sum()),
        electricity_bill_eur=electricity_bill, gas_bill_eur=gas_bill, hvac_bill_eur=electricity_bill + gas_bill,
        comfort_zone_degree_hours=float(violation.sum() * .25),
        occupied_comfort_zone_degree_hours=float(violation[np.repeat(OCCUPIED, 4)].sum() * .25),
        max_comfort_violation_c=float(violation.max()), hourly_peak_electric_kw=float(electric.sum(1).max()),
        hourly_target_penalty=discomfort,
        tracking_mae_c=float(np.mean(np.abs(temperature - np.asarray(result["setpoints_c"])))),
        simulation_seconds=result["runtime_seconds"])
    if plan.get("predicted_power_kw") is not None:
        expected = PRICE * np.asarray(plan["predicted_power_kw"]).sum(1)
        mse = float(np.mean((expected - PRICE * electric.sum(1))**2))
        values.update(expected_electricity_bill_eur=float(expected.sum()),
            hourly_electric_cost_mse=mse, electrical_expost_plus=electricity_bill + discomfort + mse)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/energyplus"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=10.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    x, _, index = author_transitions(pd.read_csv("data/external/paper2_1991-2020.csv"))
    training_mask, _ = weekly_validation_masks(index)
    caps = np.max(np.abs(x[training_mask, 5:10]), axis=0) * 1.25
    model_paths = [Path(f"reports/public_data/nn{w}_seed0.json") for w in (2, 5)] + [Path(f"reports/prediction_baselines/{m}.json") for m in ("ARX-Ridge", "Passive-RC")]
    protocol = dict(scope="independent 5-zone 24-hour EnergyPlus pilot; not original paper control tables",
        train=TRAIN, validation=VALIDATION, test=TEST, preconditioning_days=2, preconditioning_setpoint_c=21.,
        actuation="hourly constant dual setpoints +/-0.05 C; original HVAC unchanged",
        weather="2006-2020 typical meteorological year; calendar 2017; perfect hourly weather forecast",
        tariff_eur_per_kwh=PRICE.tolist(), gas_eur_per_kwh=.10,
        tariff_scope="illustrative gross HVAC electricity and gas only; excludes PV, non-HVAC loads, standing and demand charges",
        target_c=21., target_penalty_weights=WEIGHT.tolist(), setpoint_domain_c=[18., 26.],
        comfort_bands="20..24 C 08:00-18:00; 18..26 C otherwise; project choices, not compliance claims",
        model_power_caps_kw=caps.tolist(), capacity_scope="1.25 times maximum absolute TRAINING electrical power; surrogate bounds, not equipment ratings",
        seed=0, neural_width=5, epochs=args.epochs, samples_per_update=2, sigma=.01, learning_rate=.001,
        solver_time_limit=args.time_limit, solver_relative_gap=1e-4,
        training_objective="Paper 2 electrical Expost+ with hourly endpoint comfort; gas reported separately, not optimized",
        failure_policy="all outcomes retained; failed schedules use occupied/unoccupied thermostat, training penalty 1000000",
        input_model_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in model_paths},
        adapter_sha256=hashlib.sha256(Path("src/dfl_hvac/energyplus_runtime.py").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "cvxpy", "pyscipopt")}, python=platform.python_version())
    protocol_path = output / "protocol.json"
    if args.resume and protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Resume protocol differs; use a fresh output directory")
    if protocol_path.exists() and not args.resume:
        raise ValueError("Output already contains a protocol; use --resume or a fresh directory")
    dump(protocol_path, protocol)
    reference, scenarios = {}, {}

    def reference_for(day):
        if day not in reference:
            reference[day] = run_energyplus_day(day, np.full((24, 5), 21.))
            r = reference[day]
            scenarios[day] = NeuralScenario(np.asarray(r["ambient_c"]), np.asarray(r["initial_temperature_c"]), PRICE,
                heating_capacity_kw=caps, cooling_capacity_kw=caps, discomfort_weight=WEIGHT,
                demand_charge=0., temperature_bounds_c=(18., 26.), name=day)
        return reference[day]

    def simulate(day, setpoints):
        ref = reference_for(day)
        result = run_energyplus_day(day, setpoints)
        if result["preconditioning_trace_sha256"] != ref["preconditioning_trace_sha256"] or not np.allclose(result["initial_temperature_c"], ref["initial_temperature_c"], atol=1e-10, rtol=0) or not np.allclose(result["ambient_c"], ref["ambient_c"], atol=1e-10, rtol=0):
            raise RuntimeError("Paired initial-state/weather check failed")
        return result

    def occupied_schedule():
        return np.broadcast_to(np.where(OCCUPIED, 21., 18.)[:, None], (24, 5)).copy()

    def plan_model(day, model):
        reference_for(day)
        key = hashlib.sha256(json.dumps(dict(model=model.to_dict(), day=day, protocol=protocol), sort_keys=True).encode()).hexdigest()
        path = Path("outputs/energyplus_plans") / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        try:
            if isinstance(model, ReLUDynamics):
                schedule = solve_neural_hvac(model, scenarios[day], time_limit=args.time_limit, relative_gap=1e-4, require_optimal=False)
                plan = dict(setpoints=schedule.indoor_temperature_c[1:].tolist(), predicted_power_kw=schedule.hvac_power_kw.tolist(),
                    objective=schedule.objective, status=schedule.status, gap=schedule.relative_gap,
                    solve_seconds=schedule.solve_seconds, build_seconds=schedule.build_seconds, dynamics_residual=schedule.max_dynamics_residual)
            else:
                plan = solve_linear_hvac(model, scenarios[day])
                plan = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in plan.items()}
            plan["fallback"] = False
        except RuntimeError as exc:
            plan = dict(setpoints=occupied_schedule().tolist(), predicted_power_kw=None,
                status="failed", gap=None, solve_seconds=None, fallback=True, error=str(exc))
        dump(path, plan)
        return plan

    base = ReLUDynamics.from_dict(json.loads(model_paths[1].read_text()))
    models = {"NN2-ITO": ReLUDynamics.from_dict(json.loads(model_paths[0].read_text())), "NN5-ITO": base}
    for path in model_paths[2:]:
        models[path.stem] = LinearDynamics.from_dict(json.loads(path.read_text()))
    for estimator, label in (("score", "NN5-SS"), ("antithetic", "NN5-AS")):
        model_path, log_path = output / f"{label}.json", output / f"{label}_training.json"
        if args.resume and model_path.exists() and log_path.exists():
            models[label] = ReLUDynamics.from_dict(json.loads(model_path.read_text()))
            continue
        calls = []

        def objective(day, split):
            def loss(vector):
                plan = plan_model(day, base.with_vector(vector))
                value = 1e6 if plan["fallback"] else metrics(simulate(day, plan["setpoints"]), plan)["electrical_expost_plus"]
                calls.append(dict(day=day, split=split, value=value, status=plan["status"], gap=plan["gap"], fallback=plan["fallback"]))
                dump(output / f"{label}_evaluations.json", calls)
                print(f"{label} {split} {day}: {value:.3f} ({plan['status']})", flush=True)
                return value
            return loss

        start = perf_counter()
        fit = fit_smoothing(base.vector(), [objective(day, "train") for day in TRAIN],
            [objective(day, "validation") for day in VALIDATION], estimator=estimator,
            samples=2, sigma=.01, seed=0, max_epochs=args.epochs, patience=args.epochs)
        models[label] = base.with_vector(fit.parameters)
        log = asdict(fit)
        del log["parameters"]
        dump(log_path, dict(**log, seconds=perf_counter() - start,
            initial_validation_loss=float(np.mean([c["value"] for c in calls[:len(VALIDATION)]]))))
        dump(model_path, models[label].to_dict())
    rows, hourly, fine_rows, audits = [], [], [], []
    for day in TEST:
        reference_for(day)
        plans = {"Fixed-21": dict(setpoints=np.full((24, 5), 21.).tolist(), predicted_power_kw=None, status="reference", gap=None, solve_seconds=0., fallback=False),
                 "Scheduled": dict(setpoints=occupied_schedule().tolist(), predicted_power_kw=None, status="reference", gap=None, solve_seconds=0., fallback=False)}
        plans.update({label: plan_model(day, model) for label, model in models.items()})
        for label, plan in plans.items():
            result = simulate(day, plan["setpoints"])
            row = dict(day=day, method=label, **metrics(result, plan), solver_status=plan["status"], solver_gap=plan["gap"], solve_seconds=plan["solve_seconds"], fallback=plan["fallback"])
            rows.append(row)
            print(json.dumps(row), flush=True)
            audits.append(dict(day=day, method=label, provenance=result["provenance"],
                derived_idf_sha256=result["derived_idf_sha256"], initial_temperature_c=result["initial_temperature_c"],
                preconditioning_trace_sha256=result["preconditioning_trace_sha256"], native_output_max_errors=result["native_output_max_errors"], warnings=result["warnings"],
                plan={k: v for k, v in plan.items() if k not in {"setpoints", "predicted_power_kw"}}))
            for hour in HOURS:
                for zone, name in enumerate(ZONES):
                    hourly.append(dict(day=day, method=label, hour=int(hour), zone=name,
                        ambient_c=result["ambient_c"][hour], price_eur_per_kwh=float(PRICE[hour]),
                        setpoint_c=result["setpoints_c"][hour][zone], temperature_end_c=result["temperature_c"][hour+1][zone],
                        electricity_kwh=result["electricity_kwh"][hour][zone], gas_kwh=result["gas_kwh"][hour][zone],
                        predicted_power_kw=None if plan["predicted_power_kw"] is None else plan["predicted_power_kw"][hour][zone]))
            for sample in result["quarter_hour"]:
                for zone, name in enumerate(ZONES):
                    fine_rows.append(dict(day=day, method=label, end_hour=sample["end_hour"], zone=name, temperature_c=sample["temperature_c"][zone]))
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(hourly).to_csv(output / "hourly.csv", index=False)
        pd.DataFrame(fine_rows).to_csv(output / "quarter_hour.csv", index=False)
        dump(output / "audits.json", audits)
    dump(output / "completion.json", dict(test_days=len(TEST), methods=len(models)+2, rows=len(rows), protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest()))


if __name__ == "__main__":
    main()
