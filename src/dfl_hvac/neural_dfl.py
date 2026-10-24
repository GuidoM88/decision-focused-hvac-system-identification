"""Paper 2 QP-relaxation and fixed-activation DFL comparison methods."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .neural import ReLUDynamics
from .neural_control import solve_neural_hvac


class TrainableNeuralDynamics(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.tensor(w, dtype=torch.float64)) for w in network.weights]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.tensor(b, dtype=torch.float64)) for b in network.biases]
        )
        self.template = network

    def export(self):
        n = self.template
        return ReLUDynamics(
            tuple(w.detach().numpy().copy() for w in self.weights),
            tuple(b.detach().numpy().copy() for b in self.biases),
            n.input_mean.copy(),
            n.input_scale.copy(),
            n.output_mean.copy(),
            n.output_scale.copy(),
        )


def neural_surrogate_loss(model, scenario, system, *, method="qp"):
    """Differentiate supervised Eq. (44), never the observed plant response.

    QP differentiates weather-conditioned bounds too. FB freezes the exact-MIP
    activation pattern. Both use the same explicit physical temperature domain.
    """
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    if method not in {"qp", "fb"}:
        raise ValueError("method must be qp or fb")
    n = model.template
    d = scenario.arrays(n.n_zones)
    horizon, zones = d["heat_cap"].shape
    tau = cp.Variable((horizon + 1, zones))
    heat, cool = (
        cp.Variable((horizon, zones), nonneg=True),
        cp.Variable((horizon, zones), nonneg=True),
    )
    imported, exported = cp.Variable(horizon, nonneg=True), cp.Variable(horizon, nonneg=True)
    peak = cp.Variable(nonneg=True)
    constraints = [
        tau[0] == d["initial"],
        tau >= d["state_lower"],
        tau <= d["state_upper"],
        heat <= d["heat_cap"],
        cool <= d["cool_cap"],
        imported <= scenario.grid_capacity_kw,
        exported <= scenario.grid_capacity_kw,
        imported - exported == cp.sum(heat + cool, axis=1) + d["load"] - d["generation"],
        peak >= imported + exported,
    ]
    parameters, values, weights, biases = [], [], [], []
    for wt, bt in zip(model.weights, model.biases):
        wp, bp = cp.Parameter(tuple(wt.shape)), cp.Parameter(tuple(bt.shape))
        weights.append(wp)
        biases.append(bp)
        parameters.extend((wp, bp))
        values.extend((wt, bt))
    exact = solve_neural_hvac(model.export(), scenario) if method == "fb" else None
    for t in range(horizon):
        inputs = cp.hstack([tau[t], heat[t] - cool[t], d["ambient"][t]])
        inputs = cp.multiply(1 / n.input_scale, inputs - n.input_mean)
        lo = np.r_[d["initial"] if t == 0 else d["state_lower"], -d["cool_cap"][t], d["ambient"][t]]
        hi = np.r_[d["initial"] if t == 0 else d["state_upper"], d["heat_cap"][t], d["ambient"][t]]
        lo = torch.tensor((lo - n.input_mean) / n.input_scale, dtype=torch.float64)
        hi = torch.tensor((hi - n.input_mean) / n.input_scale, dtype=torch.float64)
        if method == "fb":
            active_input = np.r_[
                exact.indoor_temperature_c[t],
                exact.heating_power_kw[t] - exact.cooling_power_kw[t],
                d["ambient"][t],
            ]
            active_input = (active_input - n.input_mean) / n.input_scale
        for k, (wp, bp, wt, bt) in enumerate(zip(weights, biases, model.weights, model.biases)):
            affine = wp @ inputs + bp
            if k == len(weights) - 1:
                inputs = affine
                continue
            y = cp.Variable(len(n.biases[k]), nonneg=True)
            if method == "fb":
                a = wt.detach().numpy() @ active_input + bt.detach().numpy()
                for j, enabled in enumerate(a >= 0):
                    constraints += (
                        [y[j] == affine[j], affine[j] >= 0]
                        if enabled
                        else [y[j] == 0, affine[j] <= 0]
                    )
                active_input = np.maximum(a, 0)
            else:
                pos, neg = torch.clamp(wt, min=0), torch.clamp(wt, max=0)
                low, high = pos @ lo + neg @ hi + bt, pos @ hi + neg @ lo + bt
                margin = 1e-10 * (1 + wt.abs() @ torch.maximum(lo.abs(), hi.abs()) + bt.abs())
                low, high = low - margin, high + margin
                lp, up = cp.Parameter(len(low)), cp.Parameter(len(high))
                parameters.extend((lp, up))
                values.extend((low, high))
                relaxed = cp.Variable(len(low))
                constraints += [
                    relaxed >= 0,
                    relaxed <= 1,
                    y >= affine,
                    y <= cp.multiply(up, relaxed),
                    y <= affine - cp.multiply(lp, 1 - relaxed),
                ]
                lo, hi = torch.clamp(low, min=0), torch.clamp(high, min=0)
            inputs = y
        constraints.append(tau[t + 1] == cp.multiply(n.output_scale, inputs) + n.output_mean)
    objective = scenario.demand_charge * peak + scenario.dt_hours * (
        d["price"] @ imported
        - d["sell"] @ exported
        + cp.sum(cp.multiply(d["weight"], cp.square(tau[1:] - d["target"])))
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    if not problem.is_dpp():
        raise RuntimeError("neural surrogate must be DPP")
    layer = CvxpyLayer(problem, parameters=parameters, variables=[tau, heat, cool])
    temperatures, heating, cooling = layer(*values, solver_args={"eps": 1e-6, "max_iters": 20000})
    response = system.run(
        temperature_setpoints_c=temperatures[1:].detach().numpy(),
        ambient_temperature_c=d["ambient"],
        initial_temperature_c=d["initial"],
    )
    observed = torch.tensor(response.hvac_power_kw, dtype=torch.float64)
    price = d["price"].copy() * scenario.dt_hours
    price[int(np.argmax(response.hvac_power_kw.sum(1)))] += scenario.demand_charge
    errors = heating + cooling - observed
    return (torch.tensor(price) * (errors.sum(1).abs() + errors.abs().mean(1))).mean()


def fit_neural_dfl(
    network,
    training_examples,
    validation_examples,
    *,
    method="qp",
    max_epochs=100,
    patience=15,
    learning_rate=0.001,
    decay=0.98,
):
    """Validation-only checkpoint selection, including the unchanged warm start."""
    if not training_examples or not validation_examples or min(max_epochs, patience) < 1:
        raise ValueError("nonempty splits and positive epochs/patience required")
    model = TrainableNeuralDynamics(network)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def validate():
        return float(
            np.mean(
                [
                    float(neural_surrogate_loss(model, s, p, method=method).detach())
                    for s, p in validation_examples
                ]
            )
        )

    best_loss = validate()
    best, best_epoch, stale = copy.deepcopy(model.state_dict()), -1, 0
    history = []
    for epoch in range(max_epochs):
        for scenario, plant in training_examples:
            optimizer.zero_grad()
            loss = neural_surrogate_loss(model, scenario, plant, method=method)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite neural DFL loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite neural DFL gradient")
            optimizer.step()
        val = validate()
        history.append(val)
        if val < best_loss:
            best_loss, best, best_epoch, stale = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
        for group in optimizer.param_groups:
            group["lr"] *= decay
    model.load_state_dict(best)
    return model.export(), {
        "best_epoch": best_epoch,
        "validation_loss": history,
        "selection_metric": "supervised power loss (Eq. 44)",
    }
