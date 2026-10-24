"""Immutable, checksum-verified public EnergyPlus data from the Paper 2 authors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

AUTHOR_COMMIT = "656589444a66819f2338632b6a9931f867f954b0"
DATA_BLOBS = {
    "1991-2020": "53bf2baf4598c837793dbde3b49c1b7c51a06472",
    "2006-2020": "a9791baaf5d38732e024da2a2725b770a0f7311f",
}
ZONES = ("Core_ZN", "Perimeter_ZN_1", "Perimeter_ZN_2", "Perimeter_ZN_3", "Perimeter_ZN_4")


def weekly_validation_masks(index):
    """Hold out every fifth week; embargo adjacent transitions sharing a state."""
    index = pd.DatetimeIndex(index)
    week = (index.dayofyear.to_numpy() - 1) // 7
    validation = week % 5 == 4
    training = ~validation
    training[1:] &= ~validation[:-1]
    training[:-1] &= ~validation[1:]
    return training, validation


def input_overlap_mask(reference, candidate):
    """Flag exact input collisions across weather files, independent of labels."""
    known = {tuple(row) for row in np.asarray(reference)}
    return np.array([tuple(row) in known for row in np.asarray(candidate)], dtype=bool)


def download_author_data(period, directory="data/external"):
    if period not in DATA_BLOBS:
        raise ValueError("unknown meteorological period")
    path = Path(directory) / f"paper2_{period}.csv"
    source = (
        f"https://raw.githubusercontent.com/PSMRB/dfl_hvac_management/{AUTHOR_COMMIT}/"
        f"data/SmallOffice/Output-6zones_ASHRAE901_OfficeSmall_STD2022_Denver_{period}/"
        "Datasets/OutputVariablesTraining.csv"
    )
    data = path.read_bytes() if path.exists() else urlopen(source, timeout=120).read()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != DATA_BLOBS[period]:
        raise ValueError(f"upstream Git blob checksum mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, {
        "url": source,
        "author_commit": AUTHOR_COMMIT,
        "git_blob_sha1": blob,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def author_transitions(frame):
    """Match authors' row-t power to row-(t+1) temperature convention.

    Hourly HVAC energy is Wh: dividing by 1000 gives mean kW over one hour.
    Heating/cooling sign follows zonal sensible heat rate, as in their code.
    """
    index = pd.DatetimeIndex(frame["Timestep"])
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("chronologically ordered unique hourly timestamps required")
    temp = frame[[f"Zone Mean Air Temperature,{z}" for z in ZONES]].to_numpy(float)
    heat = frame[[f"Zone Air System Sensible Heating Rate,{z}" for z in ZONES]].to_numpy(float)
    cool = frame[[f"Zone Air System Sensible Cooling Rate,{z}" for z in ZONES]].to_numpy(float)
    power = (
        frame[[f"Air System Electricity Energy,PSZ-AC:{i}[Wh]" for i in range(1, 6)]].to_numpy(
            float
        )
        / 1000
    )
    signed = np.where(heat > cool, power, -power)
    ambient = frame["Site Outdoor Air Drybulb Temperature,ENVIRONMENT"].to_numpy(float)
    valid = np.asarray(index[1:] - index[:-1] == pd.Timedelta(hours=1))
    x = np.column_stack((temp[:-1], signed[:-1], ambient[:-1]))[valid]
    y = temp[1:][valid]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite public data")
    return x, y, index[:-1][valid]
