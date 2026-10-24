import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpylayers")

from dfl_hvac.dfl_training import DFLExample, decision_focused_loss, train_dfl
from dfl_hvac.differentiable import DifferentiableHVACLayer
from dfl_hvac.rc_model import MultiZoneRCParameters
from dfl_hvac.simulation import TrackingRCSystem
from dfl_hvac.training import TrainableRCParameters


def test_full_decision_focused_chain_has_finite_gradients():
    model = TrainableRCParameters(1, seed=4)

    true_system = TrackingRCSystem(
        MultiZoneRCParameters(
            alpha=np.array([[1.10]]),
            eta_heating=np.array([1.0]),
            eta_cooling=np.array([1.0]),
            resistance=np.array([2.0]),
            capacitance=np.array([5.0]),
        ),
        heating_capacity_kw=50.0,
        cooling_capacity_kw=50.0,
    )

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
    example = DFLExample(
        name="synthetic",
        layer=layer,
        system=true_system,
        ambient_temperature_c=np.array([12.0, 11.0]),
        initial_temperature_c=np.array([21.0]),
        energy_price_eur_per_kwh=np.array([0.3, 0.6]),
        demand_charge_eur_per_kw=0.2,
        floor_groups=((0,),),
    )

    loss = decision_focused_loss(model, example)
    assert torch.isfinite(loss)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_one_epoch_dfl_training_loop_completes():
    model = TrainableRCParameters(1, seed=6)
    true_system = TrackingRCSystem(
        MultiZoneRCParameters(
            alpha=np.array([[1.10]]),
            eta_heating=np.array([1.0]),
            eta_cooling=np.array([1.0]),
            resistance=np.array([2.0]),
            capacitance=np.array([5.0]),
        ),
        heating_capacity_kw=50.0,
        cooling_capacity_kw=50.0,
    )

    def example(name, ambient):
        layer = DifferentiableHVACLayer(
            ambient_temperature_c=np.asarray(ambient, dtype=float),
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
        return DFLExample(
            name=name,
            layer=layer,
            system=true_system,
            ambient_temperature_c=np.asarray(ambient, dtype=float),
            initial_temperature_c=np.array([21.0]),
            energy_price_eur_per_kwh=np.array([0.3, 0.6]),
            demand_charge_eur_per_kw=0.2,
            floor_groups=((0,),),
        )

    result = train_dfl(
        model,
        [example("train", [12.0, 11.0])],
        [example("validation", [13.0, 10.0])],
        learning_rate=1e-5,
        max_epochs=1,
        patience=1,
    )

    assert result.best_epoch == 0
    assert len(result.training_loss) == 1
    assert len(result.validation_loss) == 1
    assert np.isfinite(result.training_loss[0])
    assert np.isfinite(result.validation_loss[0])
