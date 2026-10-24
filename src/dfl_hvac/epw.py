"""Minimal EPW reader for the ambient-temperature scenario pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

EPW_HEADER_ROWS = 8
DRY_BULB_MISSING_VALUE_C = 99.9


def read_epw_dry_bulb(path) -> pd.DataFrame:
    """Read calendar fields and outdoor dry-bulb temperature from an EPW file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    records = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(EPW_HEADER_ROWS):
            next(reader, None)

        for line_number, row in enumerate(reader, start=EPW_HEADER_ROWS + 1):
            if not row:
                continue
            if len(row) < 7:
                raise ValueError(f"EPW row {line_number} contains fewer than seven fields")

            month = int(row[1])
            day = int(row[2])
            hour = int(row[3])
            dry_bulb = float(row[6])

            if not 1 <= month <= 12:
                raise ValueError(f"invalid EPW month at row {line_number}: {month}")
            if not 1 <= hour <= 24:
                raise ValueError(f"invalid EPW hour at row {line_number}: {hour}")
            if np.isclose(dry_bulb, DRY_BULB_MISSING_VALUE_C):
                raise ValueError(f"missing dry-bulb temperature at EPW row {line_number}")

            records.append((month, day, hour, dry_bulb))

    if not records:
        raise ValueError("EPW file contains no weather records")

    return pd.DataFrame(
        records,
        columns=["month", "day", "hour", "dry_bulb_temperature_c"],
    )


def epw_daily_temperature_profiles(path) -> tuple[np.ndarray, np.ndarray]:
    """Return complete 24-hour dry-bulb profiles and month-day labels."""
    frame = read_epw_dry_bulb(path)
    profiles = []
    labels = []

    for (month, day), group in frame.groupby(["month", "day"], sort=False):
        ordered = group.sort_values("hour")
        if len(ordered) != 24 or set(ordered["hour"]) != set(range(1, 25)):
            continue
        profiles.append(ordered["dry_bulb_temperature_c"].to_numpy(dtype=float))
        labels.append(f"{int(month):02d}-{int(day):02d}")

    if not profiles:
        raise ValueError("EPW file contains no complete 24-hour days")

    return np.asarray(profiles), np.asarray(labels, dtype=str)
