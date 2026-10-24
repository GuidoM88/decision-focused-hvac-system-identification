"""Mapping-driven conversion from simulator outputs to model-ready arrays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .training import HistoricalTransitions


@dataclass(frozen=True)
class EnergyPlusColumnMap:
    indoor_temperature: tuple[str, ...]
    ambient_temperature: str
    heating_power: tuple[str, ...]
    cooling_power: tuple[str, ...]


def _matrix(frame: pd.DataFrame, columns: Sequence[str], name: str) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing {name} columns: {missing}")
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def historical_transitions_from_frame(
    frame: pd.DataFrame,
    columns: EnergyPlusColumnMap,
) -> HistoricalTransitions:
    """Construct one-step RC transitions from chronologically ordered output."""
    if len(frame) < 2:
        raise ValueError("at least two consecutive records are required")

    indoor = _matrix(frame, columns.indoor_temperature, "indoor temperature")
    heating = _matrix(frame, columns.heating_power, "heating power")
    cooling = _matrix(frame, columns.cooling_power, "cooling power")

    if not (
        indoor.shape[1] == heating.shape[1] == cooling.shape[1]
        and len(columns.indoor_temperature) > 0
    ):
        raise ValueError("temperature and HVAC mappings must contain the same zones")

    if columns.ambient_temperature not in frame.columns:
        raise ValueError(f"missing ambient-temperature column: {columns.ambient_temperature}")
    ambient = frame[columns.ambient_temperature].to_numpy(dtype=float)
    if not np.isfinite(ambient).all():
        raise ValueError("ambient temperature contains non-finite values")

    return HistoricalTransitions(
        indoor_temperature_c=indoor[:-1],
        next_indoor_temperature_c=indoor[1:],
        ambient_temperature_c=ambient[:-1],
        heating_power_kw=heating[:-1],
        cooling_power_kw=cooling[:-1],
    )


def daily_ambient_profiles(
    timestamps,
    ambient_temperature_c,
    *,
    steps_per_day: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete daily profiles and their normalized dates."""
    index = pd.DatetimeIndex(timestamps)
    ambient = np.asarray(ambient_temperature_c, dtype=float)
    if len(index) != len(ambient):
        raise ValueError("timestamps and ambient temperatures must have equal length")
    if steps_per_day < 1:
        raise ValueError("steps_per_day must be positive")
    if not np.isfinite(ambient).all():
        raise ValueError("ambient temperature contains non-finite values")

    table = pd.DataFrame({"temperature": ambient}, index=index)
    profiles = []
    dates = []
    for date, group in table.groupby(table.index.normalize(), sort=True):
        if len(group) == steps_per_day:
            profiles.append(group["temperature"].to_numpy(dtype=float))
            dates.append(date)

    if not profiles:
        raise ValueError("no complete daily profiles were found")
    return np.asarray(profiles), np.asarray(dates, dtype="datetime64[ns]")
