"""Supervised objectives used for the Identify-Then-Optimize baseline."""

from __future__ import annotations

import numpy as np

from .rc_model import MultiZoneRCParameters, simulate_multizone_rc


def trajectory_mse(
    observed_temperature_c,
    ambient_temperature_c,
    heating_power_kw,
    cooling_power_kw,
    params: MultiZoneRCParameters,
    *,
    dt_hours: float = 1.0,
) -> float:
    """MSE between observed and RC-predicted temperatures over one trajectory."""
    observed = np.asarray(observed_temperature_c, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != params.n_zones:
        raise ValueError("observed_temperature_c must have shape (T+1, Z)")

    predicted = simulate_multizone_rc(
        ambient_temperature_c=ambient_temperature_c,
        heating_power_kw=heating_power_kw,
        cooling_power_kw=cooling_power_kw,
        initial_temperature_c=observed[0],
        params=params,
        dt_hours=dt_hours,
    )
    if predicted.shape != observed.shape:
        raise ValueError("observed trajectory length does not match control inputs")
    return float(np.mean((predicted - observed) ** 2))
