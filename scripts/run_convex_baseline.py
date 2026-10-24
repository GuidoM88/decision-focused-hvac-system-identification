"""Run a small multi-zone convex scheduling instance."""

import numpy as np

from dfl_hvac import MultiZoneRCParameters, solve_day_ahead_hvac


def main() -> None:
    zones = 3
    horizon = 24
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

    hours = np.arange(horizon)
    ambient = 8.0 + 5.0 * np.sin(2 * np.pi * (hours - 8) / 24)
    price = np.where((hours >= 6) & (hours < 19), 0.6, 0.3)

    schedule = solve_day_ahead_hvac(
        ambient_temperature_c=ambient,
        initial_temperature_c=np.full(zones, 21.0),
        energy_price_eur_per_kwh=price,
        demand_charge_eur_per_kw=0.2,
        temperature_target_c=21.0,
        discomfort_weight=np.where((hours >= 7) & (hours < 18), 10.0, 1.0),
        params=params,
        zone_heating_capacity_kw=5.0,
        zone_cooling_capacity_kw=5.0,
        floor_groups=[[0, 1, 2]],
        floor_heating_capacity_kw=12.0,
        floor_cooling_capacity_kw=12.0,
        import_capacity_kw=15.0,
    )

    print(f"status={schedule.status}")
    print(f"objective_eur={schedule.objective_eur:.3f}")
    print(f"peak_power_kw={schedule.peak_power_kw:.3f}")


if __name__ == "__main__":
    main()
