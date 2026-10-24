"""Exercise the complete differentiable DFL chain on a synthetic one-zone system.

This script validates integration only. It does not reproduce a result from the
reference paper and is not used as scientific evidence.
"""

from __future__ import annotations

import json

import numpy as np

from dfl_hvac.dfl_training import DFLExample, decision_focused_loss, train_dfl
from dfl_hvac.differentiable import DifferentiableHVACLayer
from dfl_hvac.rc_model import MultiZoneRCParameters
from dfl_hvac.simulation import TrackingRCSystem
from dfl_hvac.training import TrainableRCParameters


def make_example(name: str, ambient: np.ndarray, system: TrackingRCSystem) -> DFLExample:
    horizon = len(ambient)
    price = np.where(np.arange(horizon) < horizon // 2, 0.3, 0.6)
    layer = DifferentiableHVACLayer(
        ambient_temperature_c=ambient,
        initial_temperature_c=np.array([21.0]),
        energy_price_eur_per_kwh=price,
        demand_charge_eur_per_kw=0.2,
        temperature_target_c=21.0,
        discomfort_weight=10.0,
        zone_heating_capacity_kw=12.0,
        zone_cooling_capacity_kw=12.0,
        floor_groups=[[0]],
        floor_heating_capacity_kw=12.0,
        floor_cooling_capacity_kw=12.0,
        import_capacity_kw=15.0,
    )
    return DFLExample(
        name=name,
        layer=layer,
        system=system,
        ambient_temperature_c=ambient,
        initial_temperature_c=np.array([21.0]),
        energy_price_eur_per_kwh=price,
        demand_charge_eur_per_kw=0.2,
        floor_groups=((0,),),
    )


def main() -> None:
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

    model = TrainableRCParameters(1, seed=7)
    train = make_example("train", np.array([12.0, 11.0, 10.0, 11.0]), true_system)
    validation = make_example(
        "validation", np.array([13.0, 12.0, 11.0, 12.0]), true_system
    )

    initial_validation_loss = float(decision_focused_loss(model, validation).detach())
    result = train_dfl(
        model,
        [train],
        [validation],
        learning_rate=1e-5,
        max_epochs=3,
        patience=3,
    )
    selected_validation_loss = float(decision_focused_loss(result.model, validation).detach())

    print(
        json.dumps(
            {
                "scope": "synthetic integration check only",
                "best_epoch": result.best_epoch,
                "initial_validation_loss": initial_validation_loss,
                "selected_validation_loss": selected_validation_loss,
                "training_history": list(result.training_loss),
                "validation_history": list(result.validation_loss),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
