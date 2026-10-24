"""Outer bounds for learned dynamics; these do not certify the physical building."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def affine_interval(weight, bias, lower, upper):
    weight, bias, lower, upper = [np.asarray(a, dtype=float) for a in (weight, bias, lower, upper)]
    if (
        lower.shape != upper.shape
        or lower.ndim != 1
        or np.any(lower > upper)
        or weight.shape != (len(bias), len(lower))
        or not all(np.isfinite(a).all() for a in (weight, bias, lower, upper))
    ):
        raise ValueError("invalid finite affine interval")
    pos, neg = np.maximum(weight, 0), np.minimum(weight, 0)
    lo, hi = pos @ lower + neg @ upper + bias, pos @ upper + neg @ lower + bias
    margin = 1e-10 * (1 + np.abs(weight) @ np.maximum(np.abs(lower), np.abs(upper)) + np.abs(bias))
    return lo - margin, hi + margin


def network_interval(network, lower, upper):
    lo = (np.asarray(lower) - network.input_mean) / network.input_scale
    hi = (np.asarray(upper) - network.input_mean) / network.input_scale
    pre = []
    for i, (w, b) in enumerate(zip(network.weights, network.biases)):
        lo, hi = affine_interval(w, b, lo, hi)
        if i < len(network.weights) - 1:
            pre.append((lo.copy(), hi.copy()))
            lo, hi = np.maximum(lo, 0), np.maximum(hi, 0)
    return (
        lo * network.output_scale + network.output_mean,
        hi * network.output_scale + network.output_mean,
        tuple(pre),
    )


@dataclass(frozen=True)
class HorizonBounds:
    state_lower: np.ndarray
    state_upper: np.ndarray
    preactivations: tuple
    unstable_relus: int
    total_relus: int
    preactivation_width: float


def horizon_bounds(network, scenario, *, mode="reachable"):
    """'weather' is Paper 2's singleton-input baseline; 'reachable' adds recursion."""
    if mode not in {"static", "weather", "reachable"}:
        raise ValueError("mode must be static, weather, or reachable")
    d = scenario.arrays(network.n_zones)
    horizon, zones = d["heat_cap"].shape
    lower = np.broadcast_to(d["state_lower"], (horizon + 1, zones)).copy()
    upper = np.broadcast_to(d["state_upper"], (horizon + 1, zones)).copy()
    lower[0] = upper[0] = d["initial"]
    all_pre, unstable, total, width = [], 0, 0, 0.0
    for t in range(horizon):
        if mode == "static":
            amin, amax = scenario.ambient_bounds_c
        else:
            amin = amax = d["ambient"][t]
        xlo = np.r_[lower[t], -d["cool_cap"][t], amin]
        xhi = np.r_[upper[t], d["heat_cap"][t], amax]
        out_lo, out_hi, pre = network_interval(network, xlo, xhi)
        all_pre.append(pre)
        for lo, hi in pre:
            unstable += int(np.sum((lo < 0) & (hi > 0)))
            total += len(lo)
            width += float(np.sum(hi - lo))
        if mode == "reachable":
            lower[t + 1] = np.maximum(lower[t + 1], out_lo)
            upper[t + 1] = np.minimum(upper[t + 1], out_hi)
            if np.any(lower[t + 1] > upper[t + 1]):
                raise ValueError(f"empty reachable domain at step {t + 1}")
    return HorizonBounds(lower, upper, tuple(all_pre), unstable, total, width)
