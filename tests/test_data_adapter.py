import numpy as np
import pandas as pd

from dfl_hvac.data_adapter import (
    EnergyPlusColumnMap,
    daily_ambient_profiles,
    historical_transitions_from_frame,
)


def test_frame_conversion_preserves_one_step_alignment():
    frame = pd.DataFrame(
        {
            "t1": [20.0, 20.5, 21.0],
            "t2": [21.0, 21.2, 21.4],
            "ambient": [10.0, 11.0, 12.0],
            "h1": [1.0, 2.0, 3.0],
            "h2": [0.0, 1.0, 0.0],
            "c1": [0.0, 0.0, 1.0],
            "c2": [1.0, 0.0, 0.0],
        }
    )
    data = historical_transitions_from_frame(
        frame,
        EnergyPlusColumnMap(
            indoor_temperature=("t1", "t2"),
            ambient_temperature="ambient",
            heating_power=("h1", "h2"),
            cooling_power=("c1", "c2"),
        ),
    )
    np.testing.assert_allclose(data.indoor_temperature_c, [[20.0, 21.0], [20.5, 21.2]])
    np.testing.assert_allclose(data.next_indoor_temperature_c, [[20.5, 21.2], [21.0, 21.4]])
    np.testing.assert_allclose(data.ambient_temperature_c, [10.0, 11.0])


def test_daily_profiles_drop_incomplete_days():
    timestamps = pd.date_range("2026-01-01", periods=50, freq="h")
    values = np.arange(50, dtype=float)
    profiles, dates = daily_ambient_profiles(timestamps, values)
    assert profiles.shape == (2, 24)
    assert dates.shape == (2,)
