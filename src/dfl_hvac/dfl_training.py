"""Decision-focused training with a non-differentiable system response."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .differentiable import DifferentiableHVACLayer
from .simulation import SystemRunner
from .training import TrainableRCParameters


@dataclass(frozen=True)
class DFLExample:
    name: str
    layer: DifferentiableHVACLayer
    system: SystemRunner
    ambient_temperature_c: np.ndarray
    initial_temperature_c: np.ndarray
    energy_price_eur_per_kwh: np.ndarray
    demand_charge_eur_per_kw: float
    floor_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DFLFitResult:
    model: TrainableRCParameters
    best_epoch: int
    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def torch_hierarchical_power_loss(
    predicted_power_kw: torch.Tensor,
    realized_power_kw: torch.Tensor,
    *,
    floor_groups: Sequence[Sequence[int]],
    price_weights: torch.Tensor,
) -> torch.Tensor:
    if predicted_power_kw.shape != realized_power_kw.shape:
        raise ValueError("predicted and realized power arrays must share shape (T, Z)")
    if predicted_power_kw.ndim != 2:
        raise ValueError("power arrays must have shape (T, Z)")

    zones = predicted_power_kw.shape[1]
    groups = [tuple(int(i) for i in group) for group in floor_groups]
    if sorted(i for group in groups for i in group) != list(range(zones)):
        raise ValueError("floor_groups must partition zones exactly once")

    zone_error = torch.abs(predicted_power_kw - realized_power_kw).sum(dim=1)
    floor_error = predicted_power_kw.new_zeros(predicted_power_kw.shape[0])
    for group in groups:
        floor_error = floor_error + torch.abs(
            predicted_power_kw[:, group].sum(dim=1)
            - realized_power_kw[:, group].sum(dim=1)
        )
    building_error = torch.abs(
        predicted_power_kw.sum(dim=1) - realized_power_kw.sum(dim=1)
    )

    floor_weight = float(len(groups[0])) if len({len(g) for g in groups}) == 1 else 1.0
    per_period = zones * building_error + floor_weight * floor_error + zone_error
    return torch.mean(price_weights * per_period)


def decision_focused_loss(model, example: DFLExample, *, solver_args=None):
    solution = example.layer(model, solver_args=solver_args)

    setpoints = solution.indoor_temperature_c[1:].detach().cpu().numpy()
    response = example.system.run(
        temperature_setpoints_c=setpoints,
        ambient_temperature_c=example.ambient_temperature_c,
        initial_temperature_c=example.initial_temperature_c,
    )

    realized = torch.as_tensor(
        response.hvac_power_kw,
        dtype=solution.hvac_power_kw.dtype,
        device=solution.hvac_power_kw.device,
    )
    price = np.asarray(example.energy_price_eur_per_kwh, dtype=float).copy()
    realized_building = response.hvac_power_kw.sum(axis=1)
    price[int(np.argmax(realized_building))] += float(example.demand_charge_eur_per_kw)
    price_weights = torch.as_tensor(
        price,
        dtype=solution.hvac_power_kw.dtype,
        device=solution.hvac_power_kw.device,
    )

    return torch_hierarchical_power_loss(
        solution.hvac_power_kw,
        realized,
        floor_groups=example.floor_groups,
        price_weights=price_weights,
    )


def train_dfl(
    model: TrainableRCParameters,
    training_examples: Sequence[DFLExample],
    validation_examples: Sequence[DFLExample],
    *,
    learning_rate: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 10,
    polynomial_decay_power: float = 0.9,
    solver_args=None,
) -> DFLFitResult:
    if not training_examples or not validation_examples:
        raise ValueError("training and validation example sets must be nonempty")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=max_epochs,
        power=polynomial_decay_power,
    )

    train_history = []
    val_history = []
    best_loss = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(max_epochs):
        sample_losses = []
        model.train()
        for example in training_examples:
            optimizer.zero_grad()
            loss = decision_focused_loss(model, example, solver_args=solver_args)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite DFL loss on example {example.name}")
            loss.backward()
            optimizer.step()
            sample_losses.append(float(loss.detach()))

        train_history.append(float(np.mean(sample_losses)))
        scheduler.step()

        model.eval()
        validation_losses = [
            float(decision_focused_loss(model, example, solver_args=solver_args).detach())
            for example in validation_examples
        ]
        validation_loss = float(np.mean(validation_losses))
        val_history.append(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs > patience:
                break

    model.load_state_dict(best_state)
    return DFLFitResult(
        model=model,
        best_epoch=best_epoch,
        training_loss=tuple(train_history),
        validation_loss=tuple(val_history),
    )
