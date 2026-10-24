import numpy as np

from dfl_hvac.linear_baselines import LinearDynamics, fit_passive_rc, fit_ridge, solve_linear_hvac
from dfl_hvac.neural_control import NeuralScenario


def test_linear_schedule_against_analytic_one_step_optimum():
    model = LinearDynamics(np.ones((1, 1)), np.full((1, 1), .5), np.full((1, 1), .5), np.zeros(1), np.zeros(1))
    scenario = NeuralScenario(np.zeros(1), np.array([20.]), np.ones(1), discomfort_weight=2., demand_charge=0.)
    solution = solve_linear_hvac(model, scenario)
    # min_u u + 2(20 + .5u - 21)^2 has u*=1, next temperature=20.5.
    np.testing.assert_allclose(solution["predicted_power_kw"], [[1.]], atol=1e-5)
    np.testing.assert_allclose(solution["setpoints"], [[20.5]], atol=1e-5)
    assert abs(solution["objective"] - 1.5) < 1e-6


def test_baselines_recover_known_thermal_process():
    rng = np.random.default_rng(82)
    x = np.column_stack((rng.uniform(18, 25, 500), rng.uniform(-3, 3, 500), rng.uniform(0, 30, 500)))
    y = (.9*x[:, 0] + .3*x[:, 1] + .1*x[:, 2] + .2)[:, None]
    for fit in (fit_ridge, fit_passive_rc):
        model, _ = fit(x[:400], y[:400], x[400:], y[400:])
        np.testing.assert_allclose(model.predict(x[400:]), y[400:], atol=1e-4)
        np.testing.assert_allclose(LinearDynamics.from_dict(model.to_dict()).predict(x), model.predict(x))
    assert np.all(model.state >= 0) and np.all(model.state.sum(1) <= 1.)
    assert np.all(model.heating >= 0) and np.all(model.cooling >= 0)
