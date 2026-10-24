"""Audit the published experiment independently of its metric/report code."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PILOT = REPORTS / "energyplus"
METHODS = {"Fixed-21", "Scheduled", "ARX-Ridge", "Passive-RC", "NN2-ITO", "NN5-ITO", "NN5-SS", "NN5-AS"}


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recorded_provenance_and_complete_trajectories():
    for name in ("artifacts.json", "portfolio_artifacts.json"):
        for relative, expected in read_json(REPORTS / name).items():
            assert digest(REPORTS / relative) == expected, relative
    protocol = read_json(PILOT / "protocol.json")
    assert digest(PILOT / "protocol.json") == read_json(PILOT / "completion.json")["protocol_sha256"]
    assert digest(ROOT / "scripts/run_energyplus_pilot.py") == protocol["script_sha256"]
    assert digest(ROOT / "src/dfl_hvac/energyplus_runtime.py") == protocol["adapter_sha256"]
    for path, expected in protocol["input_model_sha256"].items():
        assert digest(ROOT / path) == expected
    expected_pairs = {(day, method) for day in protocol["test"] for method in METHODS}
    metrics = pd.read_csv(PILOT / "metrics.csv")
    assert set(zip(metrics.day, metrics.method)) == expected_pairs
    assert len(metrics) == len(expected_pairs)
    for name, time_key, times in (("hourly.csv", "hour", np.arange(24)),
                                  ("quarter_hour.csv", "end_hour", np.arange(1, 97) / 4)):
        frame = pd.read_csv(PILOT / name)
        assert not frame.duplicated(["day", "method", "zone", time_key]).any()
        assert set(zip(frame.day, frame.method)) == expected_pairs
        assert frame.zone.nunique() == 5
        assert len(frame) == len(expected_pairs) * len(times) * 5
        for _, rows in frame.groupby(["day", "method", "zone"]):
            assert np.array_equal(np.sort(rows[time_key]), times)


def test_published_cost_and_comfort_recomputed_from_raw_traces():
    protocol = read_json(PILOT / "protocol.json")
    hourly = pd.read_csv(PILOT / "hourly.csv")
    fine = pd.read_csv(PILOT / "quarter_hour.csv")
    for row in pd.read_csv(PILOT / "metrics.csv").itertuples():
        hours = hourly[(hourly.day == row.day) & (hourly.method == row.method)]
        samples = fine[(fine.day == row.day) & (fine.method == row.method)]
        assert np.isfinite(hours[["electricity_kwh", "gas_kwh", "temperature_end_c"]]).all().all()
        assert np.isfinite(samples.temperature_c).all()
        electricity_bill = (hours.electricity_kwh * hours.price_eur_per_kwh).sum()
        gas_bill = hours.gas_kwh.sum() * protocol["gas_eur_per_kwh"]
        assert row.electricity_kwh == pytest.approx(hours.electricity_kwh.sum())
        assert row.gas_kwh == pytest.approx(hours.gas_kwh.sum())
        assert row.electricity_bill_eur == pytest.approx(electricity_bill)
        assert row.gas_bill_eur == pytest.approx(gas_bill)
        assert row.hvac_bill_eur == pytest.approx(electricity_bill + gas_bill)
        sample_hour = np.ceil(samples.end_hour).astype(int) - 1
        occupied = (sample_hour >= 8) & (sample_hour < 18)
        violation = np.maximum(np.where(occupied, 20, 18) - samples.temperature_c, 0)
        violation += np.maximum(samples.temperature_c - np.where(occupied, 24, 26), 0)
        assert row.comfort_zone_degree_hours == pytest.approx(violation.sum() / 4)
        assert row.occupied_comfort_zone_degree_hours == pytest.approx(violation[occupied].sum() / 4)
        assert row.max_comfort_violation_c == pytest.approx(violation.max())
        electric_by_hour = hours.groupby("hour").electricity_kwh.sum()
        assert row.hourly_peak_electric_kw == pytest.approx(electric_by_hour.max())
        weight = np.where((hours.hour >= 8) & (hours.hour < 18), 1, .1)
        penalty = (weight * (hours.temperature_end_c - protocol["target_c"])**2).sum()
        assert row.hourly_target_penalty == pytest.approx(penalty)
        endpoints = samples[samples.end_hour % 1 == 0].set_index(["end_hour", "zone"]).temperature_c
        for h in hours.itertuples():
            assert h.temperature_end_c == pytest.approx(endpoints.loc[h.hour + 1, h.zone])
        if not hours.predicted_power_kw.isna().any():
            prediction = hours.groupby("hour").predicted_power_kw.sum()
            prices = np.asarray(protocol["tariff_eur_per_kwh"])
            cost_mse = np.mean((prices * (prediction - electric_by_hour))**2)
            assert row.expected_electricity_bill_eur == pytest.approx(np.dot(prices, prediction))
            assert row.hourly_electric_cost_mse == pytest.approx(cost_mse)
            assert row.electrical_expost_plus == pytest.approx(electricity_bill + penalty + cost_mse)


def test_paired_physics_and_validation_only_checkpoint_selection():
    audits = read_json(PILOT / "audits.json")
    protocol = read_json(PILOT / "protocol.json")
    for day in protocol["test"]:
        rows = [a for a in audits if a["day"] == day]
        assert {r["method"] for r in rows} == METHODS
        assert len({r["preconditioning_trace_sha256"] for r in rows}) == 1
        assert len({r["derived_idf_sha256"] for r in rows}) == 1
        for r in rows:
            assert np.allclose(r["initial_temperature_c"], rows[0]["initial_temperature_c"], atol=1e-10, rtol=0)
            assert max(r["native_output_max_errors"].values()) <= 1e-7
    budgets = []
    for label in ("NN5-SS", "NN5-AS"):
        log = read_json(PILOT / f"{label}_training.json")
        calls = read_json(PILOT / f"{label}_evaluations.json")
        for split in ("train", "validation"):
            assert {r["day"] for r in calls if r["split"] == split} == set(protocol[split])
        budgets.append(tuple(sum(r["split"] == s for r in calls) for s in ("train", "validation")))
        assert budgets[-1] == (log["training_evaluations"], log["validation_evaluations"])
        values = [log["initial_validation_loss"], *log["validation_loss"]]
        assert values[log["best_epoch"] + 1] == min(values)
    assert budgets[0] == budgets[1]
