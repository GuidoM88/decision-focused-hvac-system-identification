import numpy as np
import pandas as pd
import pytest

from dfl_hvac.public_data import (
    ZONES,
    author_transitions,
    input_overlap_mask,
    weekly_validation_masks,
)


def test_hourly_energy_conversion_sign_alignment_and_gap_filtering():
    frame = pd.DataFrame({
        "Timestep": pd.to_datetime([
            "2001-01-01 00:00", "2001-01-01 01:00", "2001-01-01 03:00"
        ]),
        "Site Outdoor Air Drybulb Temperature,ENVIRONMENT": [4, 5, 6],
    })
    for i, zone in enumerate(ZONES, 1):
        frame[f"Zone Mean Air Temperature,{zone}"] = [20, 21, 22]
        frame[f"Zone Air System Sensible Heating Rate,{zone}"] = [i % 2, 0, 0]
        frame[f"Zone Air System Sensible Cooling Rate,{zone}"] = [1 - i % 2, 1, 1]
        frame[f"Air System Electricity Energy,PSZ-AC:{i}[Wh]"] = [2000, 9000, 8000]
    x, y, times = author_transitions(frame)
    np.testing.assert_array_equal(x, [[20] * 5 + [2, -2, 2, -2, 2] + [4]])
    np.testing.assert_array_equal(y, [[21] * 5])
    assert len(times) == 1
    with pytest.raises(ValueError, match="unique"):
        author_transitions(pd.concat([frame, frame]))


def test_embargo_prevents_shared_thermal_states_across_splits():
    index = pd.date_range("2001-01-01", periods=24 * 100, freq="h")
    train, val = weekly_validation_masks(index)
    assert train.any() and val.any()
    train_states = set(index[train]) | set(index[train] + pd.Timedelta(hours=1))
    val_states = set(index[val]) | set(index[val] + pd.Timedelta(hours=1))
    assert train_states.isdisjoint(val_states)
    np.testing.assert_array_equal(
        input_overlap_mask([[1, 2], [3, 4]], [[3, 4], [3, 5], [1, 2]]),
        [True, False, True],
    )
