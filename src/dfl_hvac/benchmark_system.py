"""Portable nonlinear synthetic plant; not EnergyPlus or a calibrated building."""

from __future__ import annotations

import numpy as np

from .simulation import SystemResponse


class NonlinearThermalSystem:
    def __init__(self, n_zones=1, capacity_kw=12.0):
        if n_zones < 1 or capacity_kw <= 0:
            raise ValueError("positive zone count and capacity required")
        self.n_zones, self.capacity_kw = n_zones, float(capacity_kw)

    def coefficients(self, indoor, ambient):
        indoor = np.asarray(indoor, dtype=float)
        natural = (
            0.90 * indoor
            + 0.025 * indoor.mean()
            + 0.075 * ambient
            + 0.025 * max(ambient - 18, 0)
            + 0.03 * np.sin(indoor - 21)
        )
        return (
            natural,
            0.32 + 0.04 * np.tanh((ambient - 5) / 10),
            0.30 - 0.04 * np.tanh((ambient - 25) / 10),
        )

    def step(self, indoor, ambient, heating, cooling):
        natural, bh, bc = self.coefficients(indoor, ambient)
        return natural + bh * heating - bc * cooling

    def run(self, *, temperature_setpoints_c, ambient_temperature_c, initial_temperature_c):
        target, ambient, initial = [
            np.asarray(a, dtype=float)
            for a in (temperature_setpoints_c, ambient_temperature_c, initial_temperature_c)
        ]
        if (
            target.ndim != 2
            or target.shape[1] != self.n_zones
            or ambient.shape != (len(target),)
            or initial.shape != (self.n_zones,)
            or not all(np.isfinite(a).all() for a in (target, ambient, initial))
        ):
            raise ValueError("invalid plant input shapes or values")
        tau = np.empty((len(target) + 1, self.n_zones))
        tau[0] = initial
        heat, cool = np.zeros_like(target), np.zeros_like(target)
        for t in range(len(target)):
            natural, bh, bc = self.coefficients(tau[t], ambient[t])
            delta = target[t] - natural
            heat[t] = np.clip(delta / bh, 0, self.capacity_kw)
            cool[t] = np.clip(-delta / bc, 0, self.capacity_kw)
            tau[t + 1] = natural + bh * heat[t] - bc * cool[t]
        return SystemResponse(tau, heat, cool, heat + cool)


def synthetic_history(plant, *, seed=0, samples=2400):
    """Exploratory historical transitions, independent of control-evaluation days."""
    rng = np.random.default_rng(seed)
    indoor = rng.uniform(18, 24, (samples, plant.n_zones))
    ambient = rng.uniform(-2, 34, samples)
    signed = rng.uniform(-8, 8, (samples, plant.n_zones))
    heat, cool = np.maximum(signed, 0), np.maximum(-signed, 0)
    target = np.asarray([plant.step(x, a, h, c) for x, a, h, c in zip(indoor, ambient, heat, cool)])
    target += rng.normal(0, 0.02, target.shape)
    return np.column_stack((indoor, signed, ambient)), target
