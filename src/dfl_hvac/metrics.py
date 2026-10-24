"""Evaluation metrics used in the reproduction."""

from __future__ import annotations

import numpy as np


def power_error_metrics(predicted_power_kw, realized_power_kw) -> dict[str, float]:
    predicted = np.asarray(predicted_power_kw, dtype=float)
    realized = np.asarray(realized_power_kw, dtype=float)
    if predicted.shape != realized.shape:
        raise ValueError("predicted and realized power must have the same shape")

    error = predicted - realized
    return {
        "mae_kw": float(np.mean(np.abs(error))),
        "mse_kw2": float(np.mean(error**2)),
        "error_mean_kw": float(np.mean(error)),
        "error_std_kw": float(np.std(error)),
    }


def cost_error(expected_cost_eur: float, ex_post_cost_eur: float) -> float:
    return float(ex_post_cost_eur - expected_cost_eur)
