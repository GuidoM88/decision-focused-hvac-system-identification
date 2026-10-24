"""Differentiable convex HVAC layer for the DFL stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DifferentiableHVACSolution:
    indoor_temperature_c: torch.Tensor
    heating_power_kw: torch.Tensor
    cooling_power_kw: torch.Tensor
    hvac_power_kw: torch.Tensor
    imported_power_kw: torch.Tensor
    peak_power_kw: torch.Tensor


class DifferentiableHVACLayer:
    """Scenario-specific CvxpyLayer implementing Eqs. (9)-(18)."""

    def __init__(
        self,
        *,
        ambient_temperature_c,
        initial_temperature_c,
        energy_price_eur_per_kwh,
        demand_charge_eur_per_kw: float,
        temperature_target_c,
        discomfort_weight,
        zone_heating_capacity_kw,
        zone_cooling_capacity_kw,
        floor_groups: Sequence[Sequence[int]],
        floor_heating_capacity_kw,
        floor_cooling_capacity_kw,
        import_capacity_kw: float,
        dt_hours: float = 1.0,
    ):
        try:
            import cvxpy as cp
            from cvxpylayers.torch import CvxpyLayer
        except ImportError as exc:
            raise ImportError(
                "DifferentiableHVACLayer requires the research optional dependencies"
            ) from exc

        price = np.asarray(energy_price_eur_per_kwh, dtype=float)
        ambient = np.asarray(ambient_temperature_c, dtype=float)
        initial = np.asarray(initial_temperature_c, dtype=float)
        if price.ndim != 1 or initial.ndim != 1:
            raise ValueError("price and initial temperature must be one-dimensional")

        horizon = price.size
        zones = initial.size
        if ambient.ndim == 1:
            if ambient.shape != (horizon,):
                raise ValueError("ambient temperature length does not match the horizon")
            ambient = np.repeat(ambient[:, None], zones, axis=1)
        if ambient.shape != (horizon, zones):
            raise ValueError("ambient temperature must have shape (T,) or (T, Z)")

        def time_zone(value, name):
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                return np.full((horizon, zones), float(array))
            if array.shape == (horizon,):
                return np.repeat(array[:, None], zones, axis=1)
            if array.shape == (zones,):
                return np.repeat(array[None, :], horizon, axis=0)
            if array.shape != (horizon, zones):
                raise ValueError(f"{name} has incompatible shape")
            return array

        target = time_zone(temperature_target_c, "temperature_target_c")
        weights = time_zone(discomfort_weight, "discomfort_weight")
        heat_cap = time_zone(zone_heating_capacity_kw, "zone_heating_capacity_kw")
        cool_cap = time_zone(zone_cooling_capacity_kw, "zone_cooling_capacity_kw")

        groups = [tuple(int(i) for i in group) for group in floor_groups]
        flattened = [i for group in groups for i in group]
        if sorted(flattened) != list(range(zones)):
            raise ValueError("floor_groups must partition all zones exactly once")
        floors = len(groups)

        def time_floor(value, name):
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                return np.full((horizon, floors), float(array))
            if array.shape == (floors,):
                return np.repeat(array[None, :], horizon, axis=0)
            if array.shape != (horizon, floors):
                raise ValueError(f"{name} has incompatible shape")
            return array

        floor_heat = time_floor(floor_heating_capacity_kw, "floor_heating_capacity_kw")
        floor_cool = time_floor(floor_cooling_capacity_kw, "floor_cooling_capacity_kw")

        alpha = cp.Parameter((zones, zones), name="alpha")
        beta_heating = cp.Parameter(zones, name="beta_heating")
        beta_cooling = cp.Parameter(zones, name="beta_cooling")
        gamma = cp.Parameter(zones, name="gamma")

        tau = cp.Variable((horizon + 1, zones), name="tau")
        p_h = cp.Variable((horizon, zones), nonneg=True, name="p_h")
        p_c = cp.Variable((horizon, zones), nonneg=True, name="p_c")
        p_import = cp.Variable(horizon, nonneg=True, name="p_import")
        p_peak = cp.Variable(nonneg=True, name="p_peak")

        constraints = [tau[0] == initial, p_h <= heat_cap, p_c <= cool_cap]
        for t in range(horizon):
            constraints.append(
                tau[t + 1]
                == dt_hours
                * (
                    alpha @ tau[t]
                    + cp.multiply(beta_heating, p_h[t])
                    - cp.multiply(beta_cooling, p_c[t])
                    + cp.multiply(gamma, ambient[t] - tau[t])
                )
            )

        for floor, zone_ids in enumerate(groups):
            constraints.append(cp.sum(p_h[:, zone_ids], axis=1) <= floor_heat[:, floor])
            constraints.append(cp.sum(p_c[:, zone_ids], axis=1) <= floor_cool[:, floor])

        hvac = p_h + p_c
        constraints += [
            p_import == cp.sum(hvac, axis=1),
            p_peak >= p_import,
            p_peak <= float(import_capacity_kw),
        ]

        objective = cp.Minimize(
            float(demand_charge_eur_per_kw) * p_peak
            + dt_hours * cp.sum(cp.multiply(price, p_import))
            + dt_hours * cp.sum(cp.multiply(weights, cp.square(tau[1:] - target)))
        )
        problem = cp.Problem(objective, constraints)
        if not problem.is_dpp():
            raise ValueError("constructed HVAC problem is not DPP-compliant")

        self._layer = CvxpyLayer(
            problem,
            parameters=[alpha, beta_heating, beta_cooling, gamma],
            variables=[tau, p_h, p_c, p_import, p_peak],
        )

    def __call__(self, model, *, solver_args=None) -> DifferentiableHVACSolution:
        alpha, beta_h, beta_c, gamma = model.affine_coefficients()
        tau, p_h, p_c, p_import, p_peak = self._layer(
            alpha,
            beta_h,
            beta_c,
            gamma,
            solver_args={} if solver_args is None else solver_args,
        )
        return DifferentiableHVACSolution(
            indoor_temperature_c=tau,
            heating_power_kw=p_h,
            cooling_power_kw=p_c,
            hvac_power_kw=p_h + p_c,
            imported_power_kw=p_import,
            peak_power_kw=p_peak,
        )
