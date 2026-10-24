from pathlib import Path

import numpy as np

from dfl_hvac.epw import epw_daily_temperature_profiles, read_epw_dry_bulb


def write_epw(path: Path, days: int = 2) -> None:
    headers = [
        "LOCATION,Denver,CO,USA,TMY,724690,39.74,-104.99,-7.0,1609",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,test",
        "COMMENTS 2,test",
        "DATA PERIODS,1,1,Data,Sunday,1/1,12/31",
    ]
    rows = []
    for day in range(1, days + 1):
        for hour in range(1, 25):
            dry_bulb = float(day * 10 + hour / 10)
            rows.append(f"2001,1,{day},{hour},60,A7A7A7,{dry_bulb}")
    path.write_text("\n".join(headers + rows) + "\n", encoding="utf-8")


def test_epw_reader_uses_official_dry_bulb_field_position(tmp_path):
    path = tmp_path / "weather.epw"
    write_epw(path)
    frame = read_epw_dry_bulb(path)
    assert len(frame) == 48
    assert np.isclose(frame.iloc[0]["dry_bulb_temperature_c"], 10.1)


def test_epw_daily_profiles_preserve_hour_order(tmp_path):
    path = tmp_path / "weather.epw"
    write_epw(path)
    profiles, labels = epw_daily_temperature_profiles(path)
    assert profiles.shape == (2, 24)
    assert labels.tolist() == ["01-01", "01-02"]
    assert np.isclose(profiles[1, -1], 22.4)
