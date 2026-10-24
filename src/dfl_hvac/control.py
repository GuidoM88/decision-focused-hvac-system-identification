"""Convex multi-zone HVAC scheduling problem from Eqs. (9)-(18)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .rc_model import MultiZoneRCParameters


@dataclass(frozen=True)
class HVACSchedule:
    indoor_temperature_c: np.ndarray
    heating_power_kw: np.ndarray
    cooling_power_kw: np.ndarray
    hvac_power_kw: np.ndarray
    imported_power_kw: np.ndarray
    peak_power_kw: float
    energy_cost_eur: float
    demand_cost_eur: float
    discomfort_cost_eur: float
    objective_eur: float
    status: str


def _as_time_zone(name: str, value, horizon: int, zones: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full((horizon, zones), float(array))
    if array.shape == (horizon,):
        return np.repeat(array[:, None], zones, axis=1)
    if array.shape == (zones,):
        return np.repeat(array[None, :], horizon, axis=0)
    if array.shape != (horizon, zones):
        raise ValueError(
            f"{name} must be scalar, shape {(horizon,)}, shape {(zones,)}, "
            f"or shape {(horizon, zones)}"
        )
    return array


def solve_day_ahead_hvac(
    *,
    ambient_temperature_c,
    initial_temperature_c,
    energy_price_eur_per_kwh,
    demand_charge_eur_per_kw: float,
    temperature_target_c,
    discomfort_weight,
    params: MultiZoneRCParameters,
    zone_heating_capacity_kw,
    zone_cooling_capacity_kw,
    floor_groups: Sequence[Sequence[int]],
    floor_heating_capacity_kw,
    floor_cooling_capacity_kw,
    import_capacity_kw: float,
    dt_hours: float = 1.0,
    solver: str | None = "CLARABEL",
) -> HVACSchedule:
    """Solve the paper-shaped convex HVAC scheduling problem."""
    params.validate()
    if dt_hours <= 0:
        raise ValueError("dt_hours must be strictly positive")
    if demand_charge_eur_per_kw < 0 or import_capacity_kw <= 0:
        raise ValueError("demand charge must be nonnegative and import capacity positive")

    price = np.asarray(energy_price_eur_per_kwh, dtype=float)
    if price.ndim != 1 or price.size == 0:
        raise ValueError("energy_price_eur_per_kwh must be a nonempty 1D array")
    horizon = int(price.size)
    zones = params.n_zones

    ambient = _as_time_zone("ambient_temperature_c", ambient_temperature_c, horizon, zones)
    target = _as_time_zone("temperature_target_c", temperature_target_c, horizon, zones)
    weights = _as_time_zone("discomfort_weight", discomfort_weight, horizon, zones)
    heat_cap = _as_time_zone("zone_heating_capacity_kw", zone_heating_capacity_kw, horizon, zones)
    cool_cap = _as_time_zone("zone_cooling_capacity_kw", zone_cooling_capacity_kw, horizon, zones)
    initial = np.asarray(initial_temperature_c, dtype=float)
    if initial.shape != (zones,):
        raise ValueError(f"initial_temperature_c must have shape {(zones,)}")
    if np.any(weights < 0) or np.any(heat_cap < 0) or np.any(cool_cap < 0):
        raise ValueError("weights and capacities must be nonnegative")

    groups = [tuple(int(i) for i in group) for group in floor_groups]
    if not groups or any(not group for group in groups):
        raise ValueError("floor_groups must contain nonempty zone groups")
    flattened = [i for group in groups for i in group]
    if sorted(flattened) != list(range(zones)):
        raise ValueError("floor_groups must partition all zones exactly once")

    floors = len(groups)
    floor_heat_cap = _as_time_zone(
        "floor_heating_capacity_kw", floor_heating_capacity_kw, horizon, floors
    )
    floor_cool_cap = _as_time_zone(
        "floor_cooling_capacity_kw", floor_cooling_capacity_kw, horizon, floors
    )

    tau = cp.Variable((horizon + 1, zones))
    p_h = cp.Variable((horizon, zones), nonneg=True)
    p_c = cp.Variable((horizon, zones), nonneg=True)
    p_hvac = cp.Variable((horizon, zones), nonneg=True)
    p_import = cp.Variable(horizon, nonneg=True)
    p_peak = cp.Variable(nonneg=True)

    alpha = np.asarray(params.alpha, dtype=float)
    eta_h = np.asarray(params.eta_heating, dtype=float)
    eta_c = np.asarray(params.eta_cooling, dtype=float)
    resistance = np.asarray(params.resistance, dtype=float)
    capacitance = np.asarray(params.capacitance, dtype=float)

    constraints: list[cp.Constraint] = [tau[0] == initial]
    constraints += [p_h <= heat_cap, p_c <= cool_cap]

    for t in range(horizon):
        rhs = dt_hours * (
            alpha @ tau[t]
            + cp.multiply(eta_h / capacitance, p_h[t])
            - cp.multiply(eta_c / capacitance, p_c[t])
            + cp.multiply(1.0 / (resistance * capacitance), ambient[t] - tau[t])
        )
        constraints.append(tau[t + 1] == rhs)

    for f, zone_ids in enumerate(groups):
        constraints.append(cp.sum(p_h[:, zone_ids], axis=1) <= floor_heat_cap[:, f])
        constraints.append(cp.sum(p_c[:, zone_ids], axis=1) <= floor_cool_cap[:, f])

    constraints += [p_hvac == p_h + p_c]
    constraints += [p_import == cp.sum(p_hvac, axis=1)]
    constraints += [p_peak >= p_import, p_peak <= import_capacity_kw]

    energy_cost = dt_hours * cp.sum(cp.multiply(price, p_import))
    demand_cost = demand_charge_eur_per_kw * p_peak
    discomfort_cost = dt_hours * cp.sum(cp.multiply(weights, cp.square(tau[1:] - target)))
    objective = cp.Minimize(energy_cost + demand_cost + discomfort_cost)

    problem = cp.Problem(objective, constraints)
    solve_kwargs = {} if solver is None else {"solver": solver}
    problem.solve(**solve_kwargs)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"HVAC optimization failed with status {problem.status}")

    return HVACSchedule(
        indoor_temperature_c=np.asarray(tau.value, dtype=float),
        heating_power_kw=np.asarray(p_h.value, dtype=float),
        cooling_power_kw=np.asarray(p_c.value, dtype=float),
        hvac_power_kw=np.asarray(p_hvac.value, dtype=float),
        imported_power_kw=np.asarray(p_import.value, dtype=float),
        peak_power_kw=float(p_peak.value),
        energy_cost_eur=float(energy_cost.value),
        demand_cost_eur=float(demand_cost.value),
        discomfort_cost_eur=float(discomfort_cost.value),
        objective_eur=float(problem.value),
        status=str(problem.status),
    )
