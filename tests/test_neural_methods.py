from itertools import product

import cvxpy as cp
import numpy as np
import pytest

from dfl_hvac.bounds import affine_interval, horizon_bounds, network_interval
from dfl_hvac.neural import ReLUDynamics
from dfl_hvac.neural_control import NeuralScenario, expost_metrics, solve_neural_hvac
from dfl_hvac.smoothing import fit_smoothing, smoothing_gradient


def fixture_network():
    return ReLUDynamics(
        (np.array([[0.9, 0.3, 0.1], [-1.0, 0.0, 0.0]]), np.array([[1.0, 0.1]])),
        (np.array([-18.0, 23.0]), np.array([18.0])),
        np.zeros(3),
        np.ones(3),
        np.zeros(1),
        np.ones(1),
    )


def fixture_scenario(horizon=2):
    return NeuralScenario(
        np.full(horizon, 15.0),
        np.array([21.0]),
        np.linspace(0.3, 0.6, horizon),
        heating_capacity_kw=3.0,
        cooling_capacity_kw=3.0,
        discomfort_weight=2.0,
        temperature_bounds_c=(10, 35),
        grid_capacity_kw=10.0,
    )


def test_expost_cost_at_tied_peaks_is_charged_once():
    from types import SimpleNamespace

    s = fixture_scenario()
    observed = SimpleNamespace(
        hvac_power_kw=np.array([[2.0], [2.0]]),
        indoor_temperature_c=np.full((3, 1), 21.0),
    )
    predicted = SimpleNamespace(
        hvac_power_kw=np.array([[1.0], [1.0]]),
        indoor_temperature_c=observed.indoor_temperature_c,
    )
    metrics = expost_metrics(predicted, observed, s)
    assert metrics["realized_bill"] == pytest.approx(2 * (0.3 + 0.6) + 0.4 * 2)
    assert metrics["cost_mse"] == pytest.approx(((0.3 + 0.4) ** 2 + 0.6**2) / 2)
    assert metrics["expost_plus"] == pytest.approx(2.6 + 0.425)


def test_parameter_roundtrip_and_mixed_sign_interval_corners():
    model = fixture_network()
    copied = ReLUDynamics.from_dict(model.to_dict()).with_vector(model.vector())
    lo, hi = np.array([18, -3, 5.0]), np.array([24, 3, 30.0])
    lower, upper, _ = network_interval(model, lo, hi)
    for corner in product(*zip(lo, hi)):
        expected = model.predict(corner)
        np.testing.assert_allclose(expected, copied.predict(corner))
        assert np.all(expected >= lower) and np.all(expected <= upper)
    lower, upper = affine_interval(
        np.array([[-2.0, 3.0]]), np.array([1.0]), np.array([-1.0, 4.0]), np.array([2.0, 5.0])
    )
    assert lower[0] <= 9 and upper[0] >= 18
    assert upper[0] - lower[0] == pytest.approx(9, abs=1e-7)


def test_recursive_bounds_enclose_rollouts_and_shrink_weather_bounds():
    model, scenario = fixture_network(), fixture_scenario(6)
    weather, reach = [horizon_bounds(model, scenario, mode=m) for m in ("weather", "reachable")]
    assert reach.unstable_relus <= weather.unstable_relus
    assert reach.preactivation_width < weather.preactivation_width
    rng = np.random.default_rng(42)
    for _ in range(100):
        state = scenario.initial_temperature_c.copy()
        for t in range(6):
            state = model.predict(np.r_[state, rng.uniform(-3, 3), 15])
            assert np.all(state >= reach.state_lower[t + 1] - 1e-8)
            assert np.all(state <= reach.state_upper[t + 1] + 1e-8)


def test_exact_mip_matches_independent_activation_enumeration():
    pytest.importorskip("pyscipopt")
    network, scenario = fixture_network(), fixture_scenario()
    exact = solve_neural_hvac(network, scenario)
    weather = solve_neural_hvac(network, scenario, bound_mode="weather")
    best = float("inf")
    for pattern in product((0, 1), repeat=4):
        tau, heat, cool = cp.Variable(3), cp.Variable(2, nonneg=True), cp.Variable(2, nonneg=True)
        peak = cp.Variable(nonneg=True)
        constraints = [
            tau[0] == 21,
            tau >= 10,
            tau <= 35,
            heat <= 3,
            cool <= 3,
            peak >= heat + cool,
        ]
        for t in range(2):
            x = cp.hstack([tau[t], heat[t] - cool[t], 15])
            pre = network.weights[0] @ x + network.biases[0]
            active = np.asarray(pattern[2 * t : 2 * t + 2])
            constraints += [cp.multiply(2 * active - 1, pre) >= 0]
            constraints += [tau[t + 1] == network.weights[1] @ cp.multiply(active, pre) + 18]
        objective = (
            scenario.energy_price @ (heat + cool) + 0.4 * peak + 2 * cp.sum_squares(tau[1:] - 21)
        )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver="CLARABEL")
        if problem.status in {"optimal", "optimal_inaccurate"}:
            best = min(best, problem.value)
    assert exact.objective == pytest.approx(best, abs=2e-4)
    assert weather.objective == pytest.approx(exact.objective, abs=2e-4)
    assert exact.max_dynamics_residual < 1e-5
    assert exact.binary_variables <= weather.binary_variables


def test_smoothing_budget_shift_invariance_and_analytic_gradient():
    theta = np.array([0.4, -0.2])
    kwargs = dict(sigma=0.05, samples=2000, estimator="antithetic")
    first = smoothing_gradient(
        lambda v: float(v @ v), theta, rng=np.random.default_rng(7), **kwargs
    )
    shifted = smoothing_gradient(
        lambda v: float(v @ v) + 100, theta, rng=np.random.default_rng(7), **kwargs
    )
    np.testing.assert_allclose(first.gradient, shifted.gradient, atol=1e-10)
    noise = np.random.default_rng(7).normal(size=(1000, 2))
    empirical_exact = (noise.T @ noise / 1000) @ (2 * theta)
    np.testing.assert_allclose(first.gradient, empirical_exact, atol=1e-12)
    assert first.evaluations == 2000
    with pytest.raises(ValueError, match="even"):
        smoothing_gradient(
            lambda x: 0.0,
            theta,
            sigma=0.1,
            samples=3,
            rng=np.random.default_rng(0),
            estimator="antithetic",
        )


def test_smoothing_retains_warm_start_and_counts_validation():
    def objective(v):
        return float(v @ v)

    result = fit_smoothing(
        np.zeros(1),
        [objective],
        [objective],
        sigma=0.1,
        samples=2,
        estimator="score",
        max_epochs=2,
        patience=2,
        seed=0,
    )
    assert result.best_epoch == -1
    np.testing.assert_array_equal(result.parameters, [0.0])
    assert result.training_evaluations == 4
    assert result.validation_evaluations == 3


@pytest.mark.parametrize("method", ["qp", "fb"])
def test_neural_surrogate_has_finite_gradients(method):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cvxpylayers")
    pytest.importorskip("pyscipopt")
    from dfl_hvac.benchmark_system import NonlinearThermalSystem
    from dfl_hvac.neural_dfl import TrainableNeuralDynamics, neural_surrogate_loss

    model = TrainableNeuralDynamics(fixture_network())
    loss = neural_surrogate_loss(model, fixture_scenario(), NonlinearThermalSystem(), method=method)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_scenario_rejects_arbitrage_and_weather_outside_declared_bounds():
    with pytest.raises(ValueError, match="arbitrage"):
        NeuralScenario(
            np.array([10.0]), np.array([21.0]), np.array([0.3]), export_price=0.6
        ).arrays(1)
    with pytest.raises(ValueError, match="ambient"):
        NeuralScenario(np.array([100.0]), np.array([21.0]), np.array([0.3])).arrays(1)
