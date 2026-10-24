"""Boundary/accounting checks plus an opt-in test against the real engine."""

import os
import sqlite3

import numpy as np
import pytest

from dfl_hvac.energyplus_runtime import (
    ELECTRICITY,
    GAS,
    SYSTEMS,
    hourly_bucket,
    prepare_idf,
    read_hourly_energy,
    run_energyplus_day,
)


def test_sql_interval_end_buckets():
    np.testing.assert_array_equal(hourly_bucket([0, 0, 0, 1, 1, 23, 24], [15, 30, 45, 0, 15, 45, 0]), [1, 1, 1, 1, 2, 24, 24])


def native_fixture(path, units="J", duplicate=False, missing=False):
    with sqlite3.connect(path) as connection:
        connection.executescript("""
        CREATE TABLE EnvironmentPeriods(EnvironmentPeriodIndex INT, EnvironmentType INT);
        CREATE TABLE Time(TimeIndex INT, Hour INT, Minute INT, Month INT, Day INT, WarmupFlag INT, EnvironmentPeriodIndex INT);
        CREATE TABLE ReportDataDictionary(ReportDataDictionaryIndex INT, Name TEXT, KeyValue TEXT, Units TEXT, ReportingFrequency TEXT);
        CREATE TABLE ReportData(ReportDataDictionaryIndex INT, TimeIndex INT, Value REAL);
        INSERT INTO EnvironmentPeriods VALUES(1,3), (2,1);
        """)
        for hour in range(1, 25):
            connection.execute("INSERT INTO Time VALUES(?,?,0,1,17,0,1)", (hour, hour))
        for i, (name, key) in enumerate((n, s) for n in (ELECTRICITY, GAS) for s in SYSTEMS):
            connection.execute("INSERT INTO ReportDataDictionary VALUES(?,?,?,?, 'Hourly')", (i, name, key, units))
            for hour in range(1, 25):
                if not (missing and i == 0 and hour == 1):
                    connection.execute("INSERT INTO ReportData VALUES(?,?,?)", (i, hour, 3_600_000 * hour))
        if duplicate:
            connection.execute("INSERT INTO ReportData VALUES(0,1,3600000)")


def test_native_joule_conversion(tmp_path):
    path = tmp_path / "case.sql"
    native_fixture(path)
    result = read_hourly_energy(path, "2017-01-17")
    assert result["electricity_kwh"].shape == (24, 5)
    assert result["electricity_kwh"].sum() == 1500.
    np.testing.assert_array_equal(result["gas_kwh"][:, 0], np.arange(1, 25))


@pytest.mark.parametrize("kwargs", [dict(units="Wh"), dict(duplicate=True), dict(missing=True)])
def test_native_accounting_rejects_ambiguous_data(tmp_path, kwargs):
    path = tmp_path / "case.sql"
    native_fixture(path, **kwargs)
    with pytest.raises(ValueError):
        read_hourly_energy(path, "2017-01-17")


def test_case_preparation_preserves_source_and_equipment(tmp_path):
    source = tmp_path / "original.idf"
    text = "Version,22.1; ! original\nRunPeriod,old,1,1,,12,31,,Sunday;\nCoil:Heating:Fuel,Backup,NaturalGas;"
    source.write_text(text)
    target = tmp_path / "case.idf"
    prepare_idf(source, target, "2017-01-17")
    assert source.read_text() == text
    assert "Coil:Heating:Fuel,Backup,NaturalGas;" in target.read_text()
    assert "1, 15, 2017, 1, 17, 2017, Sunday" in target.read_text()
    with pytest.raises(ValueError):
        prepare_idf(source, target, "2017-01-01")


@pytest.mark.skipif(os.environ.get("RUN_ENERGYPLUS_TESTS") != "1", reason="opt-in native EnergyPlus 22.1 integration")
def test_real_engine_actuation_repeatability_and_accounting(tmp_path):
    constant = np.full((24, 5), 21.)
    changed = constant.copy()
    changed[8:18] = 22.
    first = run_energyplus_day("2017-01-17", constant, cache=tmp_path)
    repeat = run_energyplus_day("2017-01-17", constant, cache=tmp_path, force=True)
    perturbed = run_energyplus_day("2017-01-17", changed, cache=tmp_path)
    assert first["preconditioning_trace_sha256"] == repeat["preconditioning_trace_sha256"] == perturbed["preconditioning_trace_sha256"]
    np.testing.assert_array_equal(first["initial_temperature_c"], perturbed["initial_temperature_c"])
    np.testing.assert_array_equal(first["electricity_kwh"], repeat["electricity_kwh"])
    assert abs(np.sum(first["electricity_kwh"]) - np.sum(perturbed["electricity_kwh"])) > 1.
    assert max(perturbed["native_output_max_errors"].values()) < 1e-7
