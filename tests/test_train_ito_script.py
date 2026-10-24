from pathlib import Path

import numpy as np

from scripts.train_ito import load_transitions


def test_transition_archive_loader(tmp_path):
    path = Path(tmp_path) / "transitions.npz"
    np.savez_compressed(
        path,
        indoor_temperature_c=np.zeros((3, 2)),
        next_indoor_temperature_c=np.ones((3, 2)),
        ambient_temperature_c=np.zeros(3),
        heating_power_kw=np.zeros((3, 2)),
        cooling_power_kw=np.zeros((3, 2)),
    )
    transitions = load_transitions(path)
    assert transitions.indoor_temperature_c.shape == (3, 2)
    assert transitions.next_indoor_temperature_c.shape == (3, 2)
