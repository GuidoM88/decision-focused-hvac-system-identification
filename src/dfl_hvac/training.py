"""Historical pre-training for the Identify-Then-Optimize RC baseline."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class HistoricalTransitions:
    indoor_temperature_c: np.ndarray
    next_indoor_temperature_c: np.ndarray
    ambient_temperature_c: np.ndarray
    heating_power_kw: np.ndarray
    cooling_power_kw: np.ndarray

    def arrays(self):
        indoor = np.asarray(self.indoor_temperature_c, dtype=np.float32)
        target = np.asarray(self.next_indoor_temperature_c, dtype=np.float32)
        ambient = np.asarray(self.ambient_temperature_c, dtype=np.float32)
        heating = np.asarray(self.heating_power_kw, dtype=np.float32)
        cooling = np.asarray(self.cooling_power_kw, dtype=np.float32)

        if indoor.ndim != 2 or target.shape != indoor.shape:
            raise ValueError("indoor and next-indoor arrays must share shape (N, Z)")
        if heating.shape != indoor.shape or cooling.shape != indoor.shape:
            raise ValueError("heating and cooling arrays must share shape (N, Z)")
        if ambient.ndim == 1:
            ambient = ambient[:, None]
        if ambient.shape[0] != indoor.shape[0] or ambient.shape[1] not in {1, indoor.shape[1]}:
            raise ValueError("ambient temperature must have shape (N,), (N, 1), or (N, Z)")
        if not all(np.isfinite(a).all() for a in (indoor, target, ambient, heating, cooling)):
            raise ValueError("historical arrays must be finite")
        return indoor, target, ambient, heating, cooling


class TrainableRCParameters(nn.Module):
    """Trainable representation of the RC parameters in Eq. (10)."""

    def __init__(self, n_zones: int, *, seed: int = 0, dt_hours: float = 1.0):
        super().__init__()
        if n_zones < 1 or dt_hours <= 0:
            raise ValueError("n_zones and dt_hours must be positive")

        generator = torch.Generator().manual_seed(seed)
        alpha = 0.1 * torch.rand((n_zones, n_zones), generator=generator)
        alpha.fill_diagonal_(1.0)

        self.alpha = nn.Parameter(alpha)
        self.eta_heating = nn.Parameter(torch.full((n_zones,), 3.0))
        self.eta_cooling = nn.Parameter(torch.full((n_zones,), 2.0))
        self.resistance = nn.Parameter(5.0 + 16.0 * torch.rand(n_zones, generator=generator))
        self.capacitance = nn.Parameter(5.0 + 16.0 * torch.rand(n_zones, generator=generator))
        self.dt_hours = float(dt_hours)

    @property
    def n_zones(self) -> int:
        return int(self.capacitance.numel())

    def affine_coefficients(self):
        beta_heating = self.eta_heating / self.capacitance
        beta_cooling = self.eta_cooling / self.capacitance
        gamma = 1.0 / (self.resistance * self.capacitance)
        return self.alpha, beta_heating, beta_cooling, gamma

    def forward(self, indoor, ambient, heating, cooling):
        indoor = torch.as_tensor(indoor, dtype=self.alpha.dtype, device=self.alpha.device)
        ambient = torch.as_tensor(ambient, dtype=self.alpha.dtype, device=self.alpha.device)
        heating = torch.as_tensor(heating, dtype=self.alpha.dtype, device=self.alpha.device)
        cooling = torch.as_tensor(cooling, dtype=self.alpha.dtype, device=self.alpha.device)

        if ambient.ndim == 1:
            ambient = ambient[:, None]
        if ambient.shape[-1] == 1:
            ambient = ambient.expand(-1, self.n_zones)

        alpha, beta_h, beta_c, gamma = self.affine_coefficients()
        return self.dt_hours * (
            indoor @ alpha.T
            + heating * beta_h
            - cooling * beta_c
            + (ambient - indoor) * gamma
        )

    def add_relative_gaussian_noise(self, divisor: float = 25.0) -> None:
        if divisor <= 0:
            raise ValueError("divisor must be positive")
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.add_(torch.randn_like(parameter) * parameter.abs() / divisor)


@dataclass(frozen=True)
class ITOFitResult:
    model: TrainableRCParameters
    best_epoch: int
    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def fit_ito(
    transitions: HistoricalTransitions,
    *,
    model: TrainableRCParameters | None = None,
    validation_fraction: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 1e-2,
    max_epochs: int = 500,
    patience: int = 20,
    seed: int = 0,
) -> ITOFitResult:
    """Fit Eq. (10) by MSE on historical transitions."""
    indoor, target, ambient, heating, cooling = transitions.arrays()
    n_samples, n_zones = indoor.shape

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if min(batch_size, max_epochs, patience) < 1:
        raise ValueError("batch_size, max_epochs and patience must be positive")

    torch.manual_seed(seed)
    if model is None:
        model = TrainableRCParameters(n_zones, seed=seed)
    if model.n_zones != n_zones:
        raise ValueError("model zone count does not match the data")

    tensors = [torch.as_tensor(a, dtype=torch.float32)
               for a in (indoor, target, ambient, heating, cooling)]

    permutation = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, round(validation_fraction * n_samples))
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val:]
    if train_idx.numel() == 0:
        raise ValueError("validation split leaves no training samples")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    train_history = []
    val_history = []
    best_loss = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        shuffled = train_idx[torch.randperm(train_idx.numel())]

        for start in range(0, shuffled.numel(), batch_size):
            idx = shuffled[start:start + batch_size]
            optimizer.zero_grad()
            prediction = model(tensors[0][idx], tensors[2][idx], tensors[3][idx], tensors[4][idx])
            loss = mse(prediction, tensors[1][idx])
            if not torch.isfinite(loss):
                raise RuntimeError("ITO training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        train_history.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            val_prediction = model(
                tensors[0][val_idx], tensors[2][val_idx], tensors[3][val_idx], tensors[4][val_idx]
            )
            val_loss = float(mse(val_prediction, tensors[1][val_idx]))
        val_history.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs > patience:
                break

    model.load_state_dict(best_state)
    return ITOFitResult(
        model=model,
        best_epoch=best_epoch,
        training_loss=tuple(train_history),
        validation_loss=tuple(val_history),
    )
