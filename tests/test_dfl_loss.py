import numpy as np
import torch

from dfl_hvac.dfl_training import torch_hierarchical_power_loss
from dfl_hvac.losses import hierarchical_power_mae


def test_torch_loss_matches_numpy_implementation():
    predicted = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]])
    realized = np.array([[1.5, 1.5, 2.0, 5.0], [1.0, 2.0, 3.0, 4.0]])
    weights = np.array([0.3, 1.2])
    groups = [[0, 1], [2, 3]]

    expected = hierarchical_power_mae(
        predicted,
        realized,
        floor_groups=groups,
        price_weights=weights,
    )
    actual = torch_hierarchical_power_loss(
        torch.tensor(predicted, dtype=torch.float64),
        torch.tensor(realized, dtype=torch.float64),
        floor_groups=groups,
        price_weights=torch.tensor(weights, dtype=torch.float64),
    )
    assert np.isclose(float(actual), expected)
