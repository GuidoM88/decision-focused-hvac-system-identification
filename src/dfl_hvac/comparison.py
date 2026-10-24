"""Comparison utilities for reported and reproduced experiment metrics."""

from __future__ import annotations

import math


def compare_metrics(observed, reference):
    rows = {}
    for metric, target in reference.items():
        if metric not in observed:
            continue
        value = float(observed[metric])
        target = float(target)
        absolute_error = value - target
        relative_error = (
            absolute_error / abs(target)
            if target != 0
            else math.nan
        )
        rows[metric] = {
            "observed": value,
            "reference": target,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
        }
    return rows


def within_tolerance(comparison, *, relative: float = 0.10, absolute: float = 1e-9) -> bool:
    if relative < 0 or absolute < 0:
        raise ValueError("tolerances must be nonnegative")
    return all(
        abs(row["absolute_error"])
        <= absolute + relative * abs(row["reference"])
        for row in comparison.values()
    )
