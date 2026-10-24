import numpy as np

from dfl_hvac.rc_model import MultiZoneRCParameters
from dfl_hvac.simulation import TrackingRCSystem


def test_tracking_system_reaches_setpoints_when_capacity_is_sufficient():
    params = MultiZoneRCParameters(
        alpha=np.array([[1.1]]),
        eta_heating=np.array([1.0]),
        eta_cooling=np.array([1.0]),
        resistance=np.array([2.0]),
        capacitance=np.array([5.0]),
    )
    system = TrackingRCSystem(
        params,
        heating_capacity_kw=100.0,
        cooling_capacity_kw=100.0,
    )
    target = np.array([[21.0], [20.5], [21.5]])
    response = system.run(
        temperature_setpoints_c=target,
        ambient_temperature_c=np.array([10.0, 12.0, 11.0]),
        initial_temperature_c=np.array([21.0]),
    )
    np.testing.assert_allclose(response.indoor_temperature_c[1:], target, atol=1e-9)
    np.testing.assert_allclose(
        response.hvac_power_kw,
        response.heating_power_kw + response.cooling_power_kw,
    )
