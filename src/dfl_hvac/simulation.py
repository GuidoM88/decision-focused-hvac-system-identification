"""Interfaces for the non-differentiable physical-system response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .rc_model import MultiZoneRCParameters, rc_step


@dataclass(frozen=True)
class SystemResponse:
    indoor_temperature_c: np.ndarray
    heating_power_kw: np.ndarray
    cooling_power_kw: np.ndarray
    hvac_power_kw: np.ndarray


class SystemRunner(Protocol):
    def run(
        self,
        *,
        temperature_setpoints_c: np.ndarray,
        ambient_temperature_c: np.ndarray,
        initial_temperature_c: np.ndarray,
    ) -> SystemResponse:
        ...


class TrackingRCSystem:
    """Synthetic black-box used only for end-to-end tests."""

    def __init__(
        self,
        params: MultiZoneRCParameters,
        *,
        heating_capacity_kw=np.inf,
        cooling_capacity_kw=np.inf,
        dt_hours: float = 1.0,
    ):
        params.validate()
        if np.any(np.asarray(params.eta_heating) <= 0) or np.any(np.asarray(params.eta_cooling) <= 0):
            raise ValueError("tracking simulator requires strictly positive HVAC efficiencies")
        self.params = params
        self.dt_hours = float(dt_hours)
        self.heating_capacity_kw = np.broadcast_to(
            np.asarray(heating_capacity_kw, dtype=float), (params.n_zones,)
        ).copy()
        self.cooling_capacity_kw = np.broadcast_to(
            np.asarray(cooling_capacity_kw, dtype=float), (params.n_zones,)
        ).copy()

    def run(
        self,
        *,
        temperature_setpoints_c,
        ambient_temperature_c,
        initial_temperature_c,
    ) -> SystemResponse:
        setpoints = np.asarray(temperature_setpoints_c, dtype=float)
        ambient = np.asarray(ambient_temperature_c, dtype=float)
        initial = np.asarray(initial_temperature_c, dtype=float)

        if setpoints.ndim != 2 or setpoints.shape[1] != self.params.n_zones:
            raise ValueError("temperature_setpoints_c must have shape (T, Z)")
        horizon, zones = setpoints.shape
        if initial.shape != (zones,):
            raise ValueError("initial_temperature_c must have shape (Z,)")
        if ambient.ndim == 1:
            if ambient.shape != (horizon,):
                raise ValueError("ambient_temperature_c has incompatible length")
            ambient = np.repeat(ambient[:, None], zones, axis=1)
        if ambient.shape != (horizon, zones):
            raise ValueError("ambient_temperature_c must have shape (T,) or (T, Z)")

        alpha = np.asarray(self.params.alpha, dtype=float)
        eta_h = np.asarray(self.params.eta_heating, dtype=float)
        eta_c = np.asarray(self.params.eta_cooling, dtype=float)
        resistance = np.asarray(self.params.resistance, dtype=float)
        capacitance = np.asarray(self.params.capacitance, dtype=float)

        temperature = np.empty((horizon + 1, zones), dtype=float)
        heating = np.zeros((horizon, zones), dtype=float)
        cooling = np.zeros((horizon, zones), dtype=float)
        temperature[0] = initial

        for t in range(horizon):
            current = temperature[t]
            requested = setpoints[t]
            required_net = capacitance * (
                requested / self.dt_hours
                - alpha @ current
                - (ambient[t] - current) / (resistance * capacitance)
            )
            heating[t] = np.clip(
                np.maximum(required_net, 0.0) / eta_h,
                0.0,
                self.heating_capacity_kw,
            )
            cooling[t] = np.clip(
                np.maximum(-required_net, 0.0) / eta_c,
                0.0,
                self.cooling_capacity_kw,
            )
            temperature[t + 1] = rc_step(
                current,
                ambient[t],
                heating[t],
                cooling[t],
                self.params,
                self.dt_hours,
            )

        return SystemResponse(
            indoor_temperature_c=temperature,
            heating_power_kw=heating,
            cooling_power_kw=cooling,
            hvac_power_kw=heating + cooling,
        )
