"""Evaluation metrics aligned with the tables reported in Paper 1."""

from __future__ import annotations

import numpy as np

from .losses import all_inclusive_price_weights, hierarchical_power_mae


def electricity_bill(
    building_power_kw,
    energy_price_eur_per_kwh,
    demand_charge_eur_per_kw: float,
    *,
    dt_hours: float = 1.0,
) -> float:
    power = np.asarray(building_power_kw, dtype=float)
    price = np.asarray(energy_price_eur_per_kwh, dtype=float)
    if power.ndim != 1 or price.shape != power.shape:
        raise ValueError("building power and energy price must share shape (T,)")
    return float(
        dt_hours * np.dot(power, price)
        + demand_charge_eur_per_kw * np.max(power)
    )


def evaluate_power_prediction(
    predicted_power_kw,
    realized_power_kw,
    *,
    floor_groups,
    energy_price_eur_per_kwh,
    demand_charge_eur_per_kw: float,
    dt_hours: float = 1.0,
):
    predicted = np.asarray(predicted_power_kw, dtype=float)
    realized = np.asarray(realized_power_kw, dtype=float)
    if predicted.ndim != 2 or realized.shape != predicted.shape:
        raise ValueError("predicted and realized power must share shape (T, Z)")

    error = predicted - realized
    predicted_building = predicted.sum(axis=1)
    realized_building = realized.sum(axis=1)

    price_weights = all_inclusive_price_weights(
        energy_price_eur_per_kwh,
        demand_charge_eur_per_kw,
        realized_building,
    )
    expected_cost = electricity_bill(
        predicted_building,
        energy_price_eur_per_kwh,
        demand_charge_eur_per_kw,
        dt_hours=dt_hours,
    )
    ex_post_cost = electricity_bill(
        realized_building,
        energy_price_eur_per_kwh,
        demand_charge_eur_per_kw,
        dt_hours=dt_hours,
    )

    return {
        "hierarchical_loss": hierarchical_power_mae(
            predicted,
            realized,
            floor_groups=floor_groups,
            price_weights=price_weights,
        ),
        "mae_kw": float(np.mean(np.abs(error))),
        "mse_kw2": float(np.mean(error**2)),
        "error_mean_kw": float(np.mean(error)),
        "error_std_kw": float(np.std(error)),
        "expected_cost_eur": expected_cost,
        "ex_post_cost_eur": ex_post_cost,
        "cost_error_eur": ex_post_cost - expected_cost,
    }
