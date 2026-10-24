"""Paper 1 case-study structure and documented reconstruction defaults."""

from __future__ import annotations

import numpy as np

N_ZONES = 15
N_FLOORS = 3
ZONES_PER_FLOOR = 5
TARGET_TEMPERATURE_C = 21.0
FLOOR_GROUPS = tuple(
    tuple(range(floor * ZONES_PER_FLOOR, (floor + 1) * ZONES_PER_FLOOR))
    for floor in range(N_FLOORS)
)


def paper1_energy_price() -> np.ndarray:
    """Return the 24-hour time-of-use import tariff stated in Paper 1."""
    return np.asarray([0.3] * 6 + [0.6] * 13 + [0.3] * 5, dtype=float)


def reconstructed_comfort_weights() -> np.ndarray:
    """Return the auxiliary 24-hour comfort-weight schedule.

    The numerical weights are recovered from the authors' later public HVAC
    implementation and are not stated numerically in Paper 1.
    """
    return np.asarray([0.1] * 8 + [1.0] * 11 + [0.3] * 5, dtype=float)


def reconstructed_demand_charge_eur_per_kw() -> float:
    """Auxiliary demand-charge default from the authors' later HVAC code."""
    return 0.4


def reconstructed_import_capacity_kw() -> float:
    """Auxiliary grid-capacity default from the authors' later HVAC code."""
    return 10_000.0
