"""Multi-zone resistance-capacitance dynamics used by the HVAC controller.

The discrete dynamics follow Eq. (10) of Favaro et al. (2025):

    tau[t+1] = dt * (
        alpha @ tau[t]
        + (eta_h * p_h[t] - eta_c * p_c[t]) / C
        + (tau_amb[t] - tau[t]) / (R * C)
    )

All vector products except ``alpha @ tau`` are element-wise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultiZoneRCParameters:
    """Parameters of the published multi-zone RC proxy model."""

    alpha: np.ndarray
    eta_heating: np.ndarray
    eta_cooling: np.ndarray
    resistance: np.ndarray
    capacitance: np.ndarray

    @property
    def n_zones(self) -> int:
        return int(np.asarray(self.capacitance).size)

    def validate(self) -> None:
        z = self.n_zones
        alpha = np.asarray(self.alpha, dtype=float)
        vectors = {
            "eta_heating": self.eta_heating,
            "eta_cooling": self.eta_cooling,
            "resistance": self.resistance,
            "capacitance": self.capacitance,
        }

        if alpha.shape != (z, z):
            raise ValueError(f"alpha must have shape {(z, z)}; got {alpha.shape}")
        if not np.isfinite(alpha).all():
            raise ValueError("alpha must contain only finite values")

        for name, value in vectors.items():
            array = np.asarray(value, dtype=float)
            if array.shape != (z,):
                raise ValueError(f"{name} must have shape {(z,)}; got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")

        if np.any(np.asarray(self.resistance, dtype=float) <= 0):
            raise ValueError("resistance must be strictly positive")
        if np.any(np.asarray(self.capacitance, dtype=float) <= 0):
            raise ValueError("capacitance must be strictly positive")
        if np.any(np.asarray(self.eta_heating, dtype=float) < 0):
            raise ValueError("eta_heating must be nonnegative")
        if np.any(np.asarray(self.eta_cooling, dtype=float) < 0):
            raise ValueError("eta_cooling must be nonnegative")


def rc_step(
    indoor_temperature_c: np.ndarray,
    ambient_temperature_c: float | np.ndarray,
    heating_power_kw: np.ndarray,
    cooling_power_kw: np.ndarray,
    params: MultiZoneRCParameters,
    dt_hours: float = 1.0,
) -> np.ndarray:
    """Evaluate one discrete RC transition."""
    params.validate()
    if dt_hours <= 0:
        raise ValueError("dt_hours must be strictly positive")

    indoor = np.asarray(indoor_temperature_c, dtype=float)
    ambient = np.asarray(ambient_temperature_c, dtype=float)
    heating = np.asarray(heating_power_kw, dtype=float)
    cooling = np.asarray(cooling_power_kw, dtype=float)
    z = params.n_zones

    for name, value in {
        "indoor_temperature_c": indoor,
        "heating_power_kw": heating,
        "cooling_power_kw": cooling,
    }.items():
        if value.shape != (z,):
            raise ValueError(f"{name} must have shape {(z,)}; got {value.shape}")

    if ambient.ndim == 0:
        ambient = np.full(z, float(ambient))
    if ambient.shape != (z,):
        raise ValueError(f"ambient_temperature_c must be scalar or shape {(z,)}")

    alpha = np.asarray(params.alpha, dtype=float)
    eta_h = np.asarray(params.eta_heating, dtype=float)
    eta_c = np.asarray(params.eta_cooling, dtype=float)
    resistance = np.asarray(params.resistance, dtype=float)
    capacitance = np.asarray(params.capacitance, dtype=float)

    return dt_hours * (
        alpha @ indoor
        + (eta_h * heating - eta_c * cooling) / capacitance
        + (ambient - indoor) / (resistance * capacitance)
    )


def simulate_multizone_rc(
    ambient_temperature_c: np.ndarray,
    heating_power_kw: np.ndarray,
    cooling_power_kw: np.ndarray,
    initial_temperature_c: np.ndarray,
    params: MultiZoneRCParameters,
    dt_hours: float = 1.0,
) -> np.ndarray:
    """Simulate a sequence of RC transitions."""
    params.validate()

    ambient = np.asarray(ambient_temperature_c, dtype=float)
    heating = np.asarray(heating_power_kw, dtype=float)
    cooling = np.asarray(cooling_power_kw, dtype=float)
    initial = np.asarray(initial_temperature_c, dtype=float)

    if heating.ndim != 2 or cooling.shape != heating.shape:
        raise ValueError("heating_power_kw and cooling_power_kw must share shape (T, Z)")
    horizon, z = heating.shape
    if z != params.n_zones:
        raise ValueError("power arrays do not match parameter zone count")
    if initial.shape != (z,):
        raise ValueError(f"initial_temperature_c must have shape {(z,)}")

    if ambient.ndim == 1:
        if ambient.shape != (horizon,):
            raise ValueError("1D ambient_temperature_c must have shape (T,)")
        ambient = np.repeat(ambient[:, None], z, axis=1)
    elif ambient.shape != (horizon, z):
        raise ValueError("ambient_temperature_c must have shape (T,) or (T, Z)")

    temperatures = np.empty((horizon + 1, z), dtype=float)
    temperatures[0] = initial
    for t in range(horizon):
        temperatures[t + 1] = rc_step(
            temperatures[t], ambient[t], heating[t], cooling[t], params, dt_hours
        )
    return temperatures
