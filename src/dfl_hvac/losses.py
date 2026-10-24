"""Decision-focused loss functions derived from Eq. (19)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def all_inclusive_price_weights(
    energy_price_eur_per_kwh,
    demand_charge_eur_per_kw: float,
    ex_post_building_power_kw,
) -> np.ndarray:
    """Construct the price weights described immediately before Eq. (19)."""
    energy_price = np.asarray(energy_price_eur_per_kwh, dtype=float)
    power = np.asarray(ex_post_building_power_kw, dtype=float)
    if energy_price.ndim != 1 or power.shape != energy_price.shape:
        raise ValueError("energy price and building power must share shape (T,)")
    if demand_charge_eur_per_kw < 0:
        raise ValueError("demand charge must be nonnegative")

    weights = energy_price.copy()
    weights[int(np.argmax(power))] += demand_charge_eur_per_kw
    return weights


def hierarchical_power_mae(
    predicted_power_kw,
    realized_power_kw,
    *,
    floor_groups: Sequence[Sequence[int]],
    price_weights=None,
    building_weight: float | None = None,
    floor_weight: float | None = None,
) -> float:
    """Evaluate the three-level hierarchical MAE in Eq. (19)."""
    predicted = np.asarray(predicted_power_kw, dtype=float)
    realized = np.asarray(realized_power_kw, dtype=float)
    if predicted.ndim != 2 or realized.shape != predicted.shape:
        raise ValueError("predicted_power_kw and realized_power_kw must share shape (T, Z)")

    horizon, zones = predicted.shape
    groups = [tuple(int(i) for i in group) for group in floor_groups]
    flattened = [i for group in groups for i in group]
    if sorted(flattened) != list(range(zones)):
        raise ValueError("floor_groups must partition all zones exactly once")

    if building_weight is None:
        building_weight = float(zones)
    if floor_weight is None:
        unique_sizes = {len(group) for group in groups}
        if len(unique_sizes) != 1:
            raise ValueError("floor_weight is required when floor sizes differ")
        floor_weight = float(next(iter(unique_sizes)))

    if price_weights is None:
        weights = np.ones(horizon, dtype=float)
    else:
        weights = np.asarray(price_weights, dtype=float)
        if weights.shape != (horizon,):
            raise ValueError(f"price_weights must have shape {(horizon,)}")

    zone_error = np.abs(predicted - realized).sum(axis=1)
    floor_error = np.zeros(horizon, dtype=float)
    for group in groups:
        floor_error += np.abs(
            predicted[:, group].sum(axis=1) - realized[:, group].sum(axis=1)
        )
    building_error = np.abs(predicted.sum(axis=1) - realized.sum(axis=1))

    per_period = (
        building_weight * building_error
        + floor_weight * floor_error
        + zone_error
    )
    return float(np.mean(weights * per_period))
