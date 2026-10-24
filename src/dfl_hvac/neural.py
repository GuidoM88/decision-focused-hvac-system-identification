"""Portable normalized ReLU dynamics. Inputs: [temperatures, signed powers, ambient]."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReLUDynamics:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    input_mean: np.ndarray
    input_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray

    def __post_init__(self):
        if len(self.weights) < 2 or len(self.weights) != len(self.biases):
            raise ValueError("at least one hidden and one output layer required")
        n = len(self.input_mean)
        for w, b in zip(self.weights, self.biases):
            if w.ndim != 2 or b.ndim != 1 or w.shape != (len(b), n):
                raise ValueError("incompatible layer dimensions")
            n = len(b)
        z = len(self.output_mean)
        if n != z or len(self.input_mean) != 2 * z + 1:
            raise ValueError("inputs must be [Z temperatures, Z signed powers, ambient]")
        arrays = (
            *self.weights,
            *self.biases,
            self.input_mean,
            self.input_scale,
            self.output_mean,
            self.output_scale,
        )
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("network arrays must be finite")
        if (
            self.input_scale.shape != self.input_mean.shape
            or self.output_scale.shape != self.output_mean.shape
            or np.any(self.input_scale <= 0)
            or np.any(self.output_scale <= 0)
        ):
            raise ValueError("normalization scales must be positive and match means")

    @property
    def n_zones(self):
        return len(self.output_mean)

    def predict(self, inputs):
        x = (np.asarray(inputs, dtype=float) - self.input_mean) / self.input_scale
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w.T + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)
        return x * self.output_scale + self.output_mean

    def vector(self):
        return np.concatenate([a.ravel() for pair in zip(self.weights, self.biases) for a in pair])

    def with_vector(self, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != self.vector().shape:
            raise ValueError("parameter vector has the wrong shape")
        cursor, weights, biases = 0, [], []
        for w, b in zip(self.weights, self.biases):
            weights.append(vector[cursor : cursor + w.size].reshape(w.shape).copy())
            cursor += w.size
            biases.append(vector[cursor : cursor + b.size].copy())
            cursor += b.size
        return ReLUDynamics(
            tuple(weights),
            tuple(biases),
            self.input_mean.copy(),
            self.input_scale.copy(),
            self.output_mean.copy(),
            self.output_scale.copy(),
        )

    def to_dict(self):
        return {
            k: [a.tolist() for a in getattr(self, k)]
            if k in {"weights", "biases"}
            else getattr(self, k).tolist()
            for k in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            **{
                k: tuple(np.asarray(a, dtype=float) for a in v)
                if k in {"weights", "biases"}
                else np.asarray(v, dtype=float)
                for k, v in data.items()
            }
        )


def fit_neural_ito(
    x,
    y,
    xv,
    yv,
    *,
    hidden_sizes=(5,),
    seed=0,
    max_epochs=500,
    patience=20,
    learning_rate=1e-3,
    batch_size=128,
    l1_weight=1e-3,
):
    """MSE + L1 pre-training. All normalization is fitted on training rows only."""
    import copy

    import torch
    from torch import nn

    x, y, xv, yv = [np.asarray(a, dtype=float) for a in (x, y, xv, yv)]
    if (
        any(a.ndim != 2 or len(a) == 0 or not np.isfinite(a).all() for a in (x, y, xv, yv))
        or len(x) != len(y)
        or len(xv) != len(yv)
        or x.shape[1] != xv.shape[1]
        or y.shape[1] != yv.shape[1]
        or x.shape[1] != 2 * y.shape[1] + 1
    ):
        raise ValueError("finite aligned training/validation matrices required")
    if (
        not hidden_sizes
        or min(hidden_sizes) < 1
        or min(max_epochs, patience, batch_size) < 1
        or learning_rate <= 0
        or l1_weight < 0
    ):
        raise ValueError("invalid neural training hyperparameters")
    torch.manual_seed(seed)
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-6)
    ym, ys = y.mean(0), np.maximum(y.std(0), 1e-6)
    xt, yt, xvt, yvt = [
        torch.tensor(a, dtype=torch.float64)
        for a in ((x - xm) / xs, (y - ym) / ys, (xv - xm) / xs, (yv - ym) / ys)
    ]
    widths = (x.shape[1], *hidden_sizes, y.shape[1])
    layers = []
    for i, (nin, nout) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(nin, nout))
        if i < len(widths) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    best, best_state, stale = float("inf"), copy.deepcopy(net.state_dict()), 0
    history = []
    for epoch in range(max_epochs):
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
            loss += l1_weight * sum(m.weight.abs().sum() for m in net if isinstance(m, nn.Linear))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val = float(((net(xvt) - yvt) ** 2).mean())
        if not np.isfinite(val):
            raise RuntimeError("nonfinite neural validation loss")
        history.append(val)
        if val < best:
            best, best_state, best_epoch, stale = val, copy.deepcopy(net.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(best_state)
    linear = [m for m in net if isinstance(m, nn.Linear)]
    result = ReLUDynamics(
        tuple(m.weight.detach().numpy().copy() for m in linear),
        tuple(m.bias.detach().numpy().copy() for m in linear),
        xm,
        xs,
        ym,
        ys,
    )
    return result, {
        "best_epoch": best_epoch,
        "validation_mse_normalized": history,
        "validation_mse_c2": float(np.mean((result.predict(xv) - yv) ** 2)),
    }
