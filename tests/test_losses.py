import numpy as np

from dfl_hvac.losses import all_inclusive_price_weights, hierarchical_power_mae


def test_hierarchical_loss_zero_for_exact_prediction():
    power = np.arange(24, dtype=float).reshape(4, 6)
    loss = hierarchical_power_mae(power, power, floor_groups=[[0, 1, 2], [3, 4, 5]])
    assert loss == 0.0


def test_hierarchical_loss_matches_manual_two_level_aggregation():
    predicted = np.zeros((1, 4))
    realized = np.array([[1.0, 0.0, 2.0, 0.0]])
    loss = hierarchical_power_mae(
        predicted,
        realized,
        floor_groups=[[0, 1], [2, 3]],
        building_weight=4.0,
        floor_weight=2.0,
    )
    assert np.isclose(loss, 21.0)


def test_peak_price_weight_is_applied_once():
    weights = all_inclusive_price_weights([0.3, 0.6, 0.6], 5.0, [2.0, 7.0, 4.0])
    np.testing.assert_allclose(weights, [0.3, 5.6, 0.6])
