import numpy as np

from dfl_hvac.evaluation import electricity_bill, evaluate_power_prediction


def test_bill_includes_energy_and_peak_demand_charge():
    power = np.array([2.0, 5.0, 3.0])
    price = np.array([0.3, 0.6, 0.3])
    expected = np.dot(power, price) + 4.0 * 5.0
    assert np.isclose(electricity_bill(power, price, 4.0), expected)


def test_prediction_error_sign_matches_paper_convention():
    predicted = np.ones((2, 2))
    realized = np.full((2, 2), 2.0)
    metrics = evaluate_power_prediction(
        predicted,
        realized,
        floor_groups=[[0, 1]],
        energy_price_eur_per_kwh=[0.3, 0.6],
        demand_charge_eur_per_kw=2.0,
    )
    assert metrics["error_mean_kw"] < 0
    assert metrics["cost_error_eur"] > 0
