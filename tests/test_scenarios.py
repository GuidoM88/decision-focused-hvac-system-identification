import numpy as np

from dfl_hvac.scenarios import (
    build_hot_year,
    build_representative_day_split,
    fixed_medoid_kmedoids,
)


def profiles():
    rng = np.random.default_rng(7)
    phase = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    groups = []
    for cluster in range(10):
        amplitude = 4.0 if cluster == 5 else 1.0 + 0.05 * cluster
        base = -5.0 + 4.0 * cluster + np.sin(phase) * amplitude
        groups.extend(base + rng.normal(0, 0.08, (8, 24)))
    return np.asarray(groups)


def test_fixed_extremes_are_medoids():
    x = profiles()
    labels, medoids = fixed_medoid_kmedoids(x, n_clusters=10, n_restarts=8, seed=4)
    means = x.mean(axis=1)
    variances = x.var(axis=1)
    assert int(np.argmin(means)) in medoids
    assert int(np.argmax(means)) in medoids
    assert int(np.argmax(variances)) in medoids
    assert labels.shape == (x.shape[0],)


def test_split_has_disjoint_training_validation_and_test_days():
    split = build_representative_day_split(profiles(), n_restarts=8, seed=2)
    assert len(np.unique(split.medoid_indices)) == 10
    assert len(np.unique(split.validation_indices)) == 10
    assert len(np.unique(split.test_indices)) == 10
    assert not set(split.medoid_indices) & set(split.validation_indices)
    assert not set(split.medoid_indices) & set(split.test_indices)


def test_hot_year_adds_two_degrees_to_hottest_selected_cluster():
    x = profiles()
    split = build_representative_day_split(x, n_restarts=8, seed=2)
    hot = build_hot_year(x, split.labels)
    assert hot.shape == (10, 24)
    boosted_mean = hot.mean(axis=1).max()
    original_hottest_cluster_day = max(
        x[np.flatnonzero(split.labels == c)].mean(axis=1).max()
        for c in np.unique(split.labels)
    )
    assert np.isclose(boosted_mean, original_hottest_cluster_day + 2.0)
