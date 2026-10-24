"""Building-specific power accounting and historical capacity extraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def allocate_floor_power_by_mass_flow(
    floor_power_kw,
    zone_air_mass_flow,
    *,
    floor_groups: Sequence[Sequence[int]],
    atol: float = 1e-9,
) -> np.ndarray:
    """Allocate each floor/AHU power to its zones in proportion to air mass flow."""
    floor_power = np.asarray(floor_power_kw, dtype=float)
    mass_flow = np.asarray(zone_air_mass_flow, dtype=float)
    if floor_power.ndim != 2 or mass_flow.ndim != 2:
        raise ValueError("power and mass-flow arrays must be two-dimensional")

    horizon, zones = mass_flow.shape
    groups = [tuple(int(i) for i in group) for group in floor_groups]
    if floor_power.shape != (horizon, len(groups)):
        raise ValueError("floor_power_kw must have shape (T, F)")
    if sorted(i for group in groups for i in group) != list(range(zones)):
        raise ValueError("floor_groups must partition all zones exactly once")
    if np.any(mass_flow < -atol) or np.any(floor_power < -atol):
        raise ValueError("power and mass flow must be nonnegative")

    allocated = np.zeros_like(mass_flow)
    for floor, group in enumerate(groups):
        group_flow = mass_flow[:, group]
        total_flow = group_flow.sum(axis=1)
        active_power = floor_power[:, floor]

        invalid = (total_flow <= atol) & (active_power > atol)
        if np.any(invalid):
            raise ValueError("positive floor power cannot be allocated at zero air flow")

        shares = np.divide(
            group_flow,
            total_flow[:, None],
            out=np.zeros_like(group_flow),
            where=total_flow[:, None] > atol,
        )
        allocated[:, group] = shares * active_power[:, None]

    return allocated


def zonal_hvac_electricity(
    floor_ahu_power_kw,
    zone_air_mass_flow,
    zone_reheat_power_kw,
    *,
    floor_groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Combine allocated AHU electricity and zone-specific reheat electricity."""
    reheat = np.asarray(zone_reheat_power_kw, dtype=float)
    allocated = allocate_floor_power_by_mass_flow(
        floor_ahu_power_kw,
        zone_air_mass_flow,
        floor_groups=floor_groups,
    )
    if reheat.shape != allocated.shape:
        raise ValueError("zone_reheat_power_kw must have shape (T, Z)")
    if np.any(reheat < 0):
        raise ValueError("reheat power must be nonnegative")
    return allocated + reheat


@dataclass(frozen=True)
class CapacityLimits:
    zone_heating_kw: np.ndarray
    zone_cooling_kw: np.ndarray
    floor_heating_kw: np.ndarray
    floor_cooling_kw: np.ndarray


def infer_capacity_limits(
    heating_power_kw,
    cooling_power_kw,
    *,
    floor_groups: Sequence[Sequence[int]],
    quantile: float = 1.0,
) -> CapacityLimits:
    """Estimate zone and floor capacities from historical operation."""
    heating = np.asarray(heating_power_kw, dtype=float)
    cooling = np.asarray(cooling_power_kw, dtype=float)
    if heating.ndim != 2 or cooling.shape != heating.shape:
        raise ValueError("heating and cooling must share shape (T, Z)")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must lie in (0, 1]")
    if np.any(heating < 0) or np.any(cooling < 0):
        raise ValueError("historical HVAC power must be nonnegative")

    zones = heating.shape[1]
    groups = [tuple(int(i) for i in group) for group in floor_groups]
    if sorted(i for group in groups for i in group) != list(range(zones)):
        raise ValueError("floor_groups must partition all zones exactly once")

    zone_heating = np.quantile(heating, quantile, axis=0)
    zone_cooling = np.quantile(cooling, quantile, axis=0)
    floor_heating = np.asarray(
        [np.quantile(heating[:, group].sum(axis=1), quantile) for group in groups]
    )
    floor_cooling = np.asarray(
        [np.quantile(cooling[:, group].sum(axis=1), quantile) for group in groups]
    )

    return CapacityLimits(
        zone_heating_kw=zone_heating,
        zone_cooling_kw=zone_cooling,
        floor_heating_kw=floor_heating,
        floor_cooling_kw=floor_cooling,
    )
