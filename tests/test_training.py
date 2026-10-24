import numpy as np
import torch

from dfl_hvac.training import HistoricalTransitions, TrainableRCParameters, fit_ito


def synthetic_transitions(n=256):
    rng = np.random.default_rng(3)
    zones = 2
    indoor = rng.normal(21.0, 1.0, size=(n, zones)).astype(np.float32)
    ambient = rng.normal(10.0, 4.0, size=n).astype(np.float32)
    heating = rng.uniform(0, 3, size=(n, zones)).astype(np.float32)
    cooling = rng.uniform(0, 2, size=(n, zones)).astype(np.float32)

    truth = TrainableRCParameters(zones, seed=5)
    with torch.no_grad():
        target = truth(indoor, ambient, heating, cooling).numpy()

    return HistoricalTransitions(indoor, target, ambient, heating, cooling)


def test_ito_training_produces_finite_loss_and_selects_a_checkpoint():
    result = fit_ito(
        synthetic_transitions(),
        model=TrainableRCParameters(2, seed=11),
        max_epochs=25,
        patience=6,
        batch_size=32,
        learning_rate=5e-3,
        seed=9,
    )
    assert result.validation_loss
    assert np.isfinite(result.validation_loss).all()
    assert 0 <= result.best_epoch < len(result.validation_loss)
    assert result.validation_loss[result.best_epoch] == min(result.validation_loss)


def test_relative_noise_changes_parameters():
    torch.manual_seed(1)
    model = TrainableRCParameters(2, seed=1)
    before = [p.detach().clone() for p in model.parameters()]
    model.add_relative_gaussian_noise(25.0)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
