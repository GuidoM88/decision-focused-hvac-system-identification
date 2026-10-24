"""Transparent ARX/Ridge and passive RC comparators for the public split.

The passive RC comparator is additional; it does not change Paper 1's literal
equation in rc_model.py and training.py.
"""

from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class LinearDynamics:
    state: np.ndarray
    heating: np.ndarray
    cooling: np.ndarray
    ambient: np.ndarray
    bias: np.ndarray

    def predict(self, x):
        x = np.asarray(x)
        z = len(self.bias)
        signed = x[..., z:2*z]
        return (x[..., :z] @ self.state.T + np.maximum(signed, 0) @ self.heating.T
                - np.maximum(-signed, 0) @ self.cooling.T + x[..., -1, None] * self.ambient + self.bias)

    def to_dict(self):
        return {k: v.tolist() for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: np.asarray(v, dtype=float) for k, v in data.items()})


def fit_ridge(x, y, xv, yv, penalties=(0., .0001, .001, .01, .1, 1., 10.)):
    """ARX(1): train-only scaling; validation-only ridge selection."""
    z = y.shape[1]
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    target_mean, target_scale = y.mean(0), np.maximum(y.std(0), 1e-8)
    features, target = (x - mean) / scale, (y - target_mean) / target_scale
    candidates, history = [], []
    for penalty in penalties:
        coefficient = np.linalg.lstsq(features.T @ features / len(x) + penalty * np.eye(x.shape[1]), features.T @ target / len(x), rcond=None)[0]
        weights = coefficient.T * target_scale[:, None] / scale
        model = LinearDynamics(weights[:, :z], weights[:, z:2*z], weights[:, z:2*z], weights[:, -1], target_mean - weights @ mean)
        history.append(dict(penalty=penalty, validation_mse=float(np.mean((model.predict(xv) - yv) ** 2))))
        candidates.append(model)
    best = int(np.argmin([row["validation_mse"] for row in history]))
    return candidates[best], dict(selected=history[best], candidates=history)


def fit_passive_rc(x, y, xv, yv, penalties=(0., .0001, .001, .01, .1)):
    """Passive Euler RC network plus a constant latent-load offset.

    Nonnegative heat-transfer rates sum to <= .99 per zone. State weights are
    nonnegative with row sums <= 1. Electrical input is still a proxy for heat;
    backup gas is unobserved in these identification features.
    """
    z = y.shape[1]
    history, candidates = [], []
    for penalty in penalties:
        state, heating, cooling, ambient, bias = np.eye(z), np.zeros((z, z)), np.zeros((z, z)), np.zeros(z), np.zeros(z)
        for j in range(z):
            others = [k for k in range(z) if k != j]
            features = np.column_stack([x[:, k] - x[:, j] for k in others] + [x[:, -1] - x[:, j], np.maximum(x[:, z+j], 0), -np.maximum(-x[:, z+j], 0), np.ones(len(x))])
            coefficient = cp.Variable(z + 3)
            problem = cp.Problem(cp.Minimize(cp.sum_squares(features @ coefficient - (y[:, j] - x[:, j])) / len(x) + penalty * cp.sum_squares(coefficient[:-1])), [coefficient[:-1] >= 0, cp.sum(coefficient[:z]) <= .99])
            problem.solve(solver="CLARABEL")
            if problem.status != "optimal":
                raise RuntimeError(f"Passive RC fit failed: {problem.status}")
            values = coefficient.value
            state[j, others] = values[:z-1]
            state[j, j] -= sum(values[:z])
            ambient[j], heating[j, j], cooling[j, j], bias[j] = values[z-1:]
        model = LinearDynamics(state, heating, cooling, ambient, bias)
        history.append(dict(penalty=penalty, validation_mse=float(np.mean((model.predict(xv) - yv) ** 2))))
        candidates.append(model)
    best = int(np.argmin([row["validation_mse"] for row in history]))
    return candidates[best], dict(selected=history[best], candidates=history)


def solve_linear_hvac(model, scenario):
    """Convex comparator with the pilot's electrical scheduling objective."""
    start = perf_counter()
    d = scenario.arrays(len(model.bias))
    horizon, zones = d["heat_cap"].shape
    temperature = cp.Variable((horizon + 1, zones))
    heating, cooling = cp.Variable((horizon, zones), nonneg=True), cp.Variable((horizon, zones), nonneg=True)
    constraints = [temperature[0] == d["initial"], temperature >= d["state_lower"], temperature <= d["state_upper"], heating <= d["heat_cap"], cooling <= d["cool_cap"]]
    for t in range(horizon):
        constraints.append(temperature[t+1] == model.state @ temperature[t] + model.heating @ heating[t] - model.cooling @ cooling[t] + model.ambient * d["ambient"][t] + model.bias)
    power = cp.sum(heating + cooling, axis=1)
    if scenario.demand_charge or np.any(d["generation"]) or np.any(d["load"]):
        raise ValueError("Linear pilot controller expects zero demand charge and no generation/load")
    objective = d["price"] @ power * scenario.dt_hours + scenario.dt_hours * cp.sum(cp.multiply(d["weight"], cp.square(temperature[1:] - d["target"])))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver="CLARABEL")
    if problem.status != "optimal":
        raise RuntimeError(f"Linear scheduling failed: {problem.status}")
    return dict(setpoints=temperature.value[1:], predicted_power_kw=heating.value + cooling.value,
                objective=float(problem.value), status=problem.status, gap=0., solve_seconds=perf_counter() - start)
