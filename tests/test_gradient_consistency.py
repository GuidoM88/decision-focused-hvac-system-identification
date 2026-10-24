import numpy as np
import pytest


def test_rc_layer_matches_independent_controller_and_finite_difference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cvxpylayers")
    from dfl_hvac.control import solve_day_ahead_hvac
    from dfl_hvac.differentiable import DifferentiableHVACLayer
    from dfl_hvac.rc_model import MultiZoneRCParameters
    from dfl_hvac.training import TrainableRCParameters

    model = TrainableRCParameters(1, seed=0).double()
    with torch.no_grad():
        model.alpha.fill_(1)
        model.eta_heating.fill_(1)
        model.eta_cooling.fill_(1)
        model.resistance.fill_(2)
        model.capacitance.fill_(5)
    args = dict(
        ambient_temperature_c=np.array([10.0, 11.0, 12.0]),
        initial_temperature_c=np.array([21.0]),
        energy_price_eur_per_kwh=np.array([0.3, 0.6, 0.6]),
        demand_charge_eur_per_kw=0.4,
        temperature_target_c=21.0,
        discomfort_weight=2.0,
        zone_heating_capacity_kw=10.0,
        zone_cooling_capacity_kw=10.0,
        floor_groups=[[0]],
        floor_heating_capacity_kw=10.0,
        floor_cooling_capacity_kw=10.0,
        import_capacity_kw=15.0,
    )
    layer = DifferentiableHVACLayer(**args)
    result = layer(model, solver_args={"eps": 1e-8})
    physical = MultiZoneRCParameters(
        *[
            getattr(model, k).detach().numpy()
            for k in ("alpha", "eta_heating", "eta_cooling", "resistance", "capacitance")
        ]
    )
    reference = solve_day_ahead_hvac(params=physical, **args)
    np.testing.assert_allclose(
        result.indoor_temperature_c.detach(), reference.indoor_temperature_c, atol=5e-4
    )
    result.hvac_power_kw.sum().backward()
    analytic = model.eta_heating.grad.item()
    values, epsilon = [], 1e-3
    for sign in (1, -1):
        with torch.no_grad():
            model.eta_heating.fill_(1 + sign * epsilon)
        values.append(float(layer(model, solver_args={"eps": 1e-8}).hvac_power_kw.sum().detach()))
    assert analytic == pytest.approx((values[0] - values[1]) / (2 * epsilon), rel=0.02, abs=0.02)
