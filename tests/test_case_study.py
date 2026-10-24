import numpy as np

from dfl_hvac.case_study import (
    FLOOR_GROUPS,
    N_ZONES,
    paper1_energy_price,
    reconstructed_comfort_weights,
)


def test_floor_groups_partition_all_controlled_zones():
    assert sorted(i for group in FLOOR_GROUPS for i in group) == list(range(N_ZONES))
    assert all(len(group) == 5 for group in FLOOR_GROUPS)


def test_paper_tariff_matches_reported_time_blocks():
    price = paper1_energy_price()
    np.testing.assert_allclose(price[:6], 0.3)
    np.testing.assert_allclose(price[6:19], 0.6)
    np.testing.assert_allclose(price[19:], 0.3)


def test_reconstructed_comfort_profile_has_three_daily_blocks():
    weights = reconstructed_comfort_weights()
    np.testing.assert_allclose(weights[:8], 0.1)
    np.testing.assert_allclose(weights[8:19], 1.0)
    np.testing.assert_allclose(weights[19:], 0.3)
