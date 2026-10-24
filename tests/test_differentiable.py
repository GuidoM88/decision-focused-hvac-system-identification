import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpylayers")

from dfl_hvac.differentiable import DifferentiableHVACLayer
from dfl_hvac.training import TrainableRCParameters


def test_differentiable_layer_produces_finite_parameter_gradients():
    model = TrainableRCParameters(1, seed=4)
    layer = DifferentiableHVACLayer(
        ambient_temperature_c=np.array([12.0, 11.0]),
        initial_temperature_c=np.array([21.0]),
        energy_price_eur_per_kwh=np.array([0.3, 0.6]),
        demand_charge_eur_per_kw=0.2,
        temperature_target_c=21.0,
        discomfort_weight=10.0,
        zone_heating_capacity_kw=10.0,
        zone_cooling_capacity_kw=10.0,
        floor_groups=[[0]],
        floor_heating_capacity_kw=10.0,
        floor_cooling_capacity_kw=10.0,
        import_capacity_kw=12.0,
    )
    solution = layer(model)
    loss = solution.imported_power_kw.sum() + 0.01 * solution.indoor_temperature_c.sum()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
