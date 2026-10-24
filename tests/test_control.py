import numpy as np

from dfl_hvac.control import solve_day_ahead_hvac
from dfl_hvac.rc_model import MultiZoneRCParameters, simulate_multizone_rc


def test_multizone_schedule_satisfies_core_accounting_and_dynamics():
    zones = 2
    horizon = 4
    resistance = np.full(zones, 2.0)
    capacitance = np.full(zones, 5.0)
    alpha = np.eye(zones) + np.diag(1.0 / (resistance * capacitance))
    params = MultiZoneRCParameters(
        alpha=alpha,
        eta_heating=np.ones(zones),
        eta_cooling=np.ones(zones),
        resistance=resistance,
        capacitance=capacitance,
    )

    schedule = solve_day_ahead_hvac(
        ambient_temperature_c=np.full(horizon, 15.0),
        initial_temperature_c=np.full(zones, 21.0),
        energy_price_eur_per_kwh=np.array([0.3, 0.6, 0.6, 0.3]),
        demand_charge_eur_per_kw=0.2,
        temperature_target_c=21.0,
        discomfort_weight=10.0,
        params=params,
        zone_heating_capacity_kw=6.0,
        zone_cooling_capacity_kw=6.0,
        floor_groups=[[0, 1]],
        floor_heating_capacity_kw=10.0,
        floor_cooling_capacity_kw=10.0,
        import_capacity_kw=12.0,
    )

    np.testing.assert_allclose(
        schedule.hvac_power_kw,
        schedule.heating_power_kw + schedule.cooling_power_kw,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        schedule.imported_power_kw,
        schedule.hvac_power_kw.sum(axis=1),
        atol=1e-6,
    )
    assert schedule.peak_power_kw + 1e-6 >= schedule.imported_power_kw.max()

    simulated = simulate_multizone_rc(
        np.full(horizon, 15.0),
        schedule.heating_power_kw,
        schedule.cooling_power_kw,
        np.full(zones, 21.0),
        params,
    )
    np.testing.assert_allclose(schedule.indoor_temperature_c, simulated, atol=1e-5)
