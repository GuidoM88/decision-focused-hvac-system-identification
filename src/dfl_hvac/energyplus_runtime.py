"""Isolated EnergyPlus 22.1 experiments with native SQL energy accounting.

Targets the pinned SmallOffice case. Each run starts a fresh engine process and
repeats identical weather, warmup and preconditioning. No air-only reset is used.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .public_data import ZONES

ELECTRICITY = "Air System Electricity Energy"
GAS = "Air System NaturalGas Energy"
SYSTEMS = tuple(f"PSZ-AC:{i}" for i in range(1, 6))


def hourly_bucket(hour, minute):
    """SQL timestamps mark interval ends: 00:15 and 01:00 both belong to hour 1."""
    return np.ceil(np.asarray(hour) + np.asarray(minute) / 60).astype(int)


def energyplus_home():
    configured = os.environ.get("ENERGYPLUS_PATH")
    candidates = [Path(configured)] if configured else sorted(Path("data/external/energyplus").glob("EnergyPlus-22.1*"))
    for path in candidates:
        if (path / "pyenergyplus/api.py").is_file():
            return path.resolve()
    raise FileNotFoundError("Set ENERGYPLUS_PATH to EnergyPlus 22.1 or run scripts/setup_energyplus.py --install-linux")


def prepare_idf(source, destination, day, preconditioning_days=2):
    """Change run dates and output requests in this known IDF; preserve equipment."""
    day = dt.date.fromisoformat(day)
    if preconditioning_days < 1 or day.year != 2017:
        raise ValueError("Use calendar year 2017 and at least one preconditioning day")
    begin = day - dt.timedelta(days=preconditioning_days)
    if begin.year != day.year:
        raise ValueError("Preconditioning must remain within the same calendar year")
    clean = re.sub(r"!.*", "", Path(source).read_text())
    objects = [obj.strip() for obj in clean.split(";") if obj.strip()]
    if sum(obj.split(",")[0].strip().lower() == "runperiod" for obj in objects) != 1:
        raise ValueError("Expected exactly one RunPeriod in the pinned case")
    kept = [obj for obj in objects if obj.split(",")[0].strip().lower() not in {"runperiod", "output:sqlite"}]
    kept.append(f"RunPeriod, DFL_PILOT, {begin.month}, {begin.day}, 2017, {day.month}, {day.day}, 2017, {begin.strftime('%A')}, No, No, No, Yes, Yes")
    kept.append("Output:SQLite, SimpleAndTabular")
    for name in (ELECTRICITY, GAS):
        for key in SYSTEMS:
            for frequency in ("Hourly", "Timestep"):
                kept.append(f"Output:Variable, {key}, {name}, {frequency}")
    for name in ("Zone Mean Air Temperature", "Zone Thermostat Heating Setpoint Temperature", "Zone Thermostat Cooling Setpoint Temperature"):
        for zone in ZONES:
            kept.append(f"Output:Variable, {zone}, {name}, Timestep")
    Path(destination).write_text(";\n\n".join(kept) + ";\n")


def native_frame(sql_path, day, frequency):
    day = dt.date.fromisoformat(day)
    with sqlite3.connect(sql_path) as connection:
        return pd.read_sql_query(
            "SELECT t.Hour, t.Minute, d.Name, d.KeyValue, d.Units, r.Value "
            "FROM ReportData r JOIN ReportDataDictionary d USING(ReportDataDictionaryIndex) "
            "JOIN Time t USING(TimeIndex) JOIN EnvironmentPeriods e USING(EnvironmentPeriodIndex) "
            "WHERE t.Month=? AND t.Day=? AND e.EnvironmentType=3 "
            "AND t.WarmupFlag=0 AND d.ReportingFrequency=?",
            connection, params=(day.month, day.day, frequency))


def read_hourly_energy(sql_path, day):
    """Read exactly 24 × 5 native Joule totals; reject missing/duplicate data."""
    frame = native_frame(sql_path, day, "Hourly")
    result = {}
    for name, label in ((ELECTRICITY, "electricity_kwh"), (GAS, "gas_kwh")):
        selected = frame.loc[frame.Name == name]
        if len(selected) != 120 or set(selected.Units) != {"J"}:
            raise ValueError(f"Expected 120 hourly Joule observations for {name}; found {len(selected)}")
        wide = selected.pivot(index="Hour", columns="KeyValue", values="Value")
        if set(wide.index) != set(range(1, 25)) or set(wide.columns) != set(SYSTEMS):
            raise ValueError(f"Incomplete HVAC energy keys: {name}")
        values = wide.loc[range(1, 25), SYSTEMS].to_numpy(float) / 3_600_000
        if not np.isfinite(values).all() or np.any(values < -1e-9):
            raise ValueError("Invalid native energy totals")
        result[label] = np.maximum(values, 0)
    return result


def audit_native_outputs(sql_path, day, setpoints, observations, energy):
    """Independent native checks for actuation, state timing and energy sums."""
    frame = native_frame(sql_path, day, "Zone Timestep")
    errors = {}
    for name, expected in (
        ("Zone Thermostat Heating Setpoint Temperature", np.repeat(np.asarray(setpoints) - .05, 4, axis=0)),
        ("Zone Thermostat Cooling Setpoint Temperature", np.repeat(np.asarray(setpoints) + .05, 4, axis=0)),
        ("Zone Mean Air Temperature", np.asarray([row["temperature_c"] for row in observations])),
    ):
        selected = frame.loc[frame.Name == name]
        wide = selected.pivot(index=["Hour", "Minute"], columns="KeyValue", values="Value").sort_index()
        actual = wide.loc[:, [z.upper() for z in ZONES]].to_numpy()
        if actual.shape != (96, 5) or set(selected.Units) != {"C"} or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"Incomplete or invalid native state/actuator output: {name}")
        errors[name] = float(np.max(np.abs(actual - expected)))
    for name, label in ((ELECTRICITY, "electricity_kwh"), (GAS, "gas_kwh")):
        selected = frame.loc[frame.Name == name]
        if len(selected) != 480 or set(selected.Units) != {"J"} or not np.isfinite(selected.Value).all():
            raise ValueError(f"Incomplete native subhourly energy output: {name}")
        selected = selected.assign(bucket=hourly_bucket(selected.Hour, selected.Minute))
        summed = selected.groupby(["bucket", "KeyValue"]).Value.sum().unstack().loc[range(1, 25), SYSTEMS].to_numpy() / 3_600_000
        errors[label] = float(np.max(np.abs(summed - energy[label])))
    if max(errors.values()) > 1e-7:
        raise ValueError(f"Native output audit failed: {errors}")
    return errors


def run_energyplus_day(day, setpoints, *, cache="outputs/energyplus", preconditioning_days=2, force=False):
    setpoints = np.asarray(setpoints, dtype=float)
    if setpoints.shape != (24, 5) or not np.isfinite(setpoints).all() or np.any((setpoints < 16) | (setpoints > 30)):
        raise ValueError("Setpoints must be finite (24, 5) values between 16 and 30 Celsius")
    case = Path("data/external/energyplus_case").resolve()
    model = case / "6zones_ASHRAE901_OfficeSmall_STD2022_Denver.idf"
    weather = case / "USA_CO_Denver.Intl.AP.725650_US.Normals.2006-2020.epw"
    home = energyplus_home()
    executable = home / ("energyplus.exe" if os.name == "nt" else "energyplus")
    version = subprocess.check_output([str(executable), "--version"], text=True).strip().split(", YMD")[0]
    if "22.1.0-ed759b17ee" not in version:
        raise ValueError(f"Expected pinned EnergyPlus 22.1.0, found {version}")
    config = dict(day=day, setpoints=setpoints.tolist(), preconditioning_days=preconditioning_days,
        model=str(model), weather=str(weather), home=str(home), version=version,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        weather_sha256=hashlib.sha256(weather.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = Path(cache).resolve() / key
    result_path = output / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    config["output"] = str(output)
    (output / "request.json").write_text(json.dumps(config, indent=2))
    with (output / "process.log").open("w") as log:
        completed = subprocess.run([sys.executable, "-m", "dfl_hvac.energyplus_runtime", str(output / "request.json")], stdout=log, stderr=log, timeout=300)
    if completed.returncode or not result_path.exists():
        raise RuntimeError(f"EnergyPlus day failed; inspect {output / 'process.log'}")
    return json.loads(result_path.read_text())


def _worker(request_path):
    config = json.loads(Path(request_path).read_text())
    output = Path(config["output"])
    prepare_idf(config["model"], output / "case.idf", config["day"], config["preconditioning_days"])
    sys.path.insert(0, config["home"])
    from pyenergyplus.api import EnergyPlusAPI

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    exchange, runtime = api.exchange, api.runtime
    runtime.set_console_output_status(state, False)
    handles = {}
    rows, prefix, errors, initial = [], [], [], []
    target = dt.date.fromisoformat(config["day"])
    setpoints = np.array(config["setpoints"])
    for zone in ZONES:
        exchange.request_variable(state, "Zone Mean Air Temperature", zone)
    exchange.request_variable(state, "Site Outdoor Air Drybulb Temperature", "Environment")

    def ready(s):
        if not exchange.api_data_fully_ready(s) or exchange.warmup_flag(s) or exchange.kind_of_sim(s) != 3:
            return False
        if not handles:
            handles["temperature"] = [exchange.get_variable_handle(s, "Zone Mean Air Temperature", zone) for zone in ZONES]
            handles["ambient"] = [exchange.get_variable_handle(s, "Site Outdoor Air Drybulb Temperature", "Environment")]
            for mode in ("Heating", "Cooling"):
                handles[mode] = [exchange.get_actuator_handle(s, "Zone Temperature Control", f"{mode} Setpoint", zone) for zone in ZONES]
            if any(h < 0 for values in handles.values() for h in values):
                (output / "api_variables.csv").write_bytes(exchange.list_available_api_data_csv(s))
                raise RuntimeError(f"Missing API handle: {handles}")
        return True

    def is_target(s):
        return exchange.month(s) == target.month and exchange.day_of_month(s) == target.day

    def safe(callback):
        def wrapped(s):
            try:
                callback(s)
            except Exception as exc:
                errors.append(repr(exc))
                runtime.stop_simulation(s)
        return wrapped

    @safe
    def actuate(s):
        if not ready(s):
            return
        hour = exchange.hour(s)
        if is_target(s) and not initial:
            initial.extend(exchange.get_variable_value(s, h) for h in handles["temperature"])
        requested = setpoints[hour] if is_target(s) else np.full(5, 21.)
        for mode, offset in (("Heating", -.05), ("Cooling", .05)):
            for handle, value in zip(handles[mode], requested + offset):
                exchange.set_actuator_value(s, handle, float(value))

    @safe
    def observe(s):
        if not ready(s):
            return
        record = dict(hour=exchange.hour(s), end_hour=exchange.current_time(s),
            dt_hours=1. / exchange.num_time_steps_in_hour(s),
            ambient_c=exchange.get_variable_value(s, handles["ambient"][0]),
            temperature_c=[exchange.get_variable_value(s, h) for h in handles["temperature"]])
        (rows if is_target(s) else prefix).append(record)

    runtime.callback_begin_system_timestep_before_predictor(state, actuate)
    runtime.callback_end_zone_timestep_after_zone_reporting(state, observe)
    start = perf_counter()
    code = runtime.run_energyplus(state, ["-w", config["weather"], "-d", str(output), str(output / "case.idf")])
    seconds = perf_counter() - start
    api.state_manager.delete_state(state)
    if code or errors:
        raise RuntimeError(f"EnergyPlus exit={code}; callback errors={errors}; see eplusout.err")
    frame = pd.DataFrame(rows)
    if len(frame) != 96 or set(frame.hour) != set(range(24)) or not (frame.groupby("hour").size() == 4).all() or not np.isclose(frame.dt_hours.sum(), 24):
        raise ValueError(f"Expected 96 quarter-hour target-day observations, got {len(frame)}")
    energy = read_hourly_energy(output / "eplusout.sql", config["day"])
    audit = audit_native_outputs(output / "eplusout.sql", config["day"], setpoints, rows, energy)
    log = (output / "eplusout.err").read_text(errors="replace")
    if "** Severe  **" in log or "**  Fatal  **" in log:
        raise RuntimeError("Native EnergyPlus log contains a severe/fatal error")
    temperatures = [initial] + frame.groupby("hour", sort=True).temperature_c.last().tolist()
    if np.asarray(temperatures).shape != (25, 5) or not np.isfinite(temperatures).all():
        raise ValueError("Incomplete initial/final thermal state")
    result = dict(day=config["day"], initial_temperature_c=initial, temperature_c=temperatures,
        ambient_c=frame.groupby("hour", sort=True).ambient_c.mean().tolist(),
        setpoints_c=config["setpoints"], quarter_hour=rows,
        **{k: v.tolist() for k, v in energy.items()}, runtime_seconds=seconds,
        provenance={k: v for k, v in config.items() if k not in {"setpoints", "home", "output", "model", "weather"}},
        derived_idf_sha256=hashlib.sha256((output / "case.idf").read_bytes()).hexdigest(),
        preconditioning_trace_sha256=hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest(),
        native_output_max_errors=audit,
        warnings=sorted({line.strip() for line in log.splitlines() if "** Warning **" in line or "Completed Successfully" in line}),
        output_directory=str(output))
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    _worker(sys.argv[1])
