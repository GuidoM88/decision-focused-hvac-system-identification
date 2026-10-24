import numpy as np
import pytest

from dfl_hvac.building_data import (
    allocate_floor_power_by_mass_flow,
    infer_capacity_limits,
    zonal_hvac_electricity,
)


def test_ahu_power_is_conserved_under_mass_flow_allocation():
    floor_power = np.array([[12.0, 8.0], [9.0, 6.0]])
    mass_flow = np.array([[1.0, 2.0, 1.0, 3.0], [2.0, 1.0, 2.0, 1.0]])
    groups = [[0, 1], [2, 3]]

    allocated = allocate_floor_power_by_mass_flow(
        floor_power,
        mass_flow,
        floor_groups=groups,
    )
    np.testing.assert_allclose(allocated[:, :2].sum(axis=1), floor_power[:, 0])
    np.testing.assert_allclose(allocated[:, 2:].sum(axis=1), floor_power[:, 1])


def test_reheat_power_is_assigned_directly_to_zone():
    floor_power = np.array([[6.0]])
    mass_flow = np.array([[1.0, 2.0]])
    reheat = np.array([[0.5, 1.0]])

    zonal = zonal_hvac_electricity(
        floor_power,
        mass_flow,
        reheat,
        floor_groups=[[0, 1]],
    )
    np.testing.assert_allclose(zonal, [[2.5, 5.0]])


def test_positive_ahu_power_at_zero_flow_is_rejected():
    with pytest.raises(ValueError, match="zero air flow"):
        allocate_floor_power_by_mass_flow(
            [[1.0]],
            [[0.0, 0.0]],
            floor_groups=[[0, 1]],
        )


def test_capacity_limits_match_historical_maxima_by_default():
    heating = np.array([[1.0, 3.0], [4.0, 2.0]])
    cooling = np.array([[2.0, 1.0], [1.0, 5.0]])
    limits = infer_capacity_limits(
        heating,
        cooling,
        floor_groups=[[0, 1]],
    )
    np.testing.assert_allclose(limits.zone_heating_kw, [4.0, 3.0])
    np.testing.assert_allclose(limits.zone_cooling_kw, [2.0, 5.0])
    np.testing.assert_allclose(limits.floor_heating_kw, [6.0])
    np.testing.assert_allclose(limits.floor_cooling_kw, [6.0])
