import numpy as np
import pytest

from dfl_hvac.rc_model import MultiZoneRCParameters, rc_step, simulate_multizone_rc


def parameters():
    return MultiZoneRCParameters(
        alpha=np.array([[0.90, 0.02], [0.01, 0.92]]),
        eta_heating=np.array([0.9, 0.8]),
        eta_cooling=np.array([0.7, 0.75]),
        resistance=np.array([2.0, 2.5]),
        capacitance=np.array([5.0, 4.0]),
    )


def test_rc_step_matches_published_equation():
    params = parameters()
    indoor = np.array([21.0, 22.0])
    ambient = 10.0
    heating = np.array([2.0, 1.0])
    cooling = np.array([0.5, 0.0])
    dt = 0.5

    expected = dt * (
        params.alpha @ indoor
        + (params.eta_heating * heating - params.eta_cooling * cooling) / params.capacitance
        + (ambient - indoor) / (params.resistance * params.capacitance)
    )
    np.testing.assert_allclose(rc_step(indoor, ambient, heating, cooling, params, dt), expected)


def test_simulation_has_expected_shape_and_initial_state():
    params = parameters()
    heating = np.zeros((3, 2))
    cooling = np.zeros((3, 2))
    initial = np.array([21.0, 22.0])
    result = simulate_multizone_rc(np.array([10.0, 11.0, 12.0]), heating, cooling, initial, params)
    assert result.shape == (4, 2)
    np.testing.assert_allclose(result[0], initial)


def test_nonpositive_resistance_is_rejected():
    params = parameters()
    bad = MultiZoneRCParameters(
        params.alpha,
        params.eta_heating,
        params.eta_cooling,
        np.array([0.0, 2.0]),
        params.capacitance,
    )
    with pytest.raises(ValueError, match="resistance"):
        bad.validate()
