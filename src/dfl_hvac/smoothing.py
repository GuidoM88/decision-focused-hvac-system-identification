"""Gaussian score-function learning with explicit objective-evaluation budgets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GradientEstimate:
    gradient: np.ndarray
    losses: np.ndarray
    evaluations: int


def smoothing_gradient(loss, theta, *, sigma, samples, rng, estimator="score"):
    """Estimate grad E[L(theta+sigma*epsilon)]; samples counts ALL evaluations.

    'score' is Paper 2's uncentered estimator. 'antithetic' is a standard
    variance-reduction comparator, not a claim of a new gradient estimator.
    """
    theta = np.asarray(theta, dtype=float)
    scale = np.broadcast_to(np.asarray(sigma, dtype=float), theta.shape)
    if (
        theta.ndim != 1
        or not np.isfinite(theta).all()
        or np.any(scale <= 0)
        or not np.isfinite(scale).all()
        or samples < 1
    ):
        raise ValueError("finite theta, positive sigma and samples required")
    losses, terms = [], []
    if estimator == "score":
        for _ in range(samples):
            noise = rng.normal(size=theta.shape)
            value = float(loss(theta + scale * noise))
            losses.append(value)
            terms.append(value * noise / scale)
    elif estimator == "antithetic":
        if samples % 2:
            raise ValueError("antithetic sampling requires an even total solve budget")
        for _ in range(samples // 2):
            noise = rng.normal(size=theta.shape)
            plus, minus = float(loss(theta + scale * noise)), float(loss(theta - scale * noise))
            losses.extend((plus, minus))
            terms.append((plus - minus) * noise / (2 * scale))
    else:
        raise ValueError("estimator must be score or antithetic")
    gradient = np.mean(terms, axis=0)
    if not np.isfinite(losses).all() or not np.isfinite(gradient).all():
        raise RuntimeError("nonfinite loss or smoothing gradient")
    return GradientEstimate(gradient, np.asarray(losses), len(losses))


@dataclass(frozen=True)
class SmoothingFit:
    parameters: np.ndarray
    best_epoch: int
    validation_loss: tuple[float, ...]
    training_loss: tuple[float, ...]
    training_evaluations: int
    validation_evaluations: int


def fit_smoothing(
    theta,
    training_losses,
    validation_losses,
    *,
    sigma=0.01,
    samples=1,
    estimator="score",
    seed=0,
    max_epochs=100,
    patience=15,
    learning_rate=0.001,
    decay=0.98,
):
    """Sample-wise Adam; warm start competes with validation-selected checkpoints.

    Failed solves are not silently discarded or replaced by successful draws.
    """
    theta = np.asarray(theta, dtype=float).copy()
    if (
        not training_losses
        or not validation_losses
        or min(max_epochs, patience) < 1
        or learning_rate <= 0
        or not 0 < decay <= 1
    ):
        raise ValueError("invalid split or optimizer settings")
    rng = np.random.default_rng(seed)

    def validate(v):
        value = float(np.mean([f(v) for f in validation_losses]))
        if not np.isfinite(value):
            raise RuntimeError("nonfinite validation loss")
        return value

    best_value = validate(theta)
    best, best_epoch, stale = theta.copy(), -1, 0
    val_history, train_history = [], []
    n_train, n_val = 0, len(validation_losses)
    first, second, step = np.zeros_like(theta), np.zeros_like(theta), 0
    for epoch in range(max_epochs):
        values = []
        for objective in training_losses:
            estimate = smoothing_gradient(
                objective, theta, sigma=sigma, samples=samples, rng=rng, estimator=estimator
            )
            n_train += estimate.evaluations
            values.extend(estimate.losses)
            step += 1
            first = 0.9 * first + 0.1 * estimate.gradient
            second = 0.999 * second + 0.001 * estimate.gradient**2
            theta -= (
                learning_rate
                * decay**epoch
                * (first / (1 - 0.9**step))
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
        train_history.append(float(np.mean(values)))
        val = validate(theta)
        n_val += len(validation_losses)
        val_history.append(val)
        if val < best_value:
            best_value, best, best_epoch, stale = val, theta.copy(), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return SmoothingFit(best, best_epoch, tuple(val_history), tuple(train_history), n_train, n_val)
