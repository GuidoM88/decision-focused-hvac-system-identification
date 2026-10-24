"""Representative-day selection for the Paper 1 experimental protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RepresentativeDaySplit:
    labels: np.ndarray
    medoid_indices: np.ndarray
    training_order: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def _validate_daily_profiles(ambient_daily_c: np.ndarray) -> np.ndarray:
    x = np.asarray(ambient_daily_c, dtype=float)
    if x.ndim != 2 or x.shape[0] < 10 or x.shape[1] < 2:
        raise ValueError("ambient_daily_c must have shape (days, steps_per_day)")
    if not np.isfinite(x).all():
        raise ValueError("ambient_daily_c must contain only finite values")
    return x


def _unique_extreme_indices(x: np.ndarray) -> list[int]:
    mean = x.mean(axis=1)
    variance = x.var(axis=1)
    rankings = [np.argsort(mean), np.argsort(-mean), np.argsort(-variance)]

    selected: list[int] = []
    for ranking in rankings:
        for index in ranking:
            index = int(index)
            if index not in selected:
                selected.append(index)
                break
    return selected


def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def fixed_medoid_kmedoids(
    ambient_daily_c,
    *,
    n_clusters: int = 10,
    n_restarts: int = 100,
    seed: int = 0,
    min_cluster_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run fixed-medoid k-medoids subject to holdout-compatible clusters."""
    x = _validate_daily_profiles(ambient_daily_c)
    if not 3 <= n_clusters <= x.shape[0]:
        raise ValueError("n_clusters must be between 3 and the number of days")
    if n_restarts < 1:
        raise ValueError("n_restarts must be positive")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be positive")

    fixed = np.asarray(_unique_extreme_indices(x), dtype=int)
    remaining = np.setdiff1d(np.arange(x.shape[0]), fixed)
    distances = _pairwise_euclidean(x)
    rng = np.random.default_rng(seed)

    best_objective = np.inf
    best_labels = None
    best_medoids = None

    for _ in range(n_restarts):
        extra = rng.choice(remaining, size=n_clusters - len(fixed), replace=False)
        medoids = np.concatenate([fixed, extra]).astype(int)

        for _ in range(200):
            labels = np.argmin(distances[:, medoids], axis=1)
            updated = medoids.copy()

            for cluster in range(len(fixed), n_clusters):
                members = np.flatnonzero(labels == cluster)
                if members.size == 0:
                    candidates = np.setdiff1d(remaining, updated)
                    if candidates.size:
                        nearest = distances[candidates][:, updated].min(axis=1)
                        updated[cluster] = int(candidates[np.argmax(nearest)])
                    continue
                within = distances[np.ix_(members, members)]
                updated[cluster] = int(members[np.argmin(within.sum(axis=1))])

            if np.array_equal(updated, medoids):
                break
            medoids = updated

        labels = np.argmin(distances[:, medoids], axis=1)
        counts = np.bincount(labels, minlength=n_clusters)
        if np.any(counts < min_cluster_size):
            continue
        objective = float(distances[np.arange(x.shape[0]), medoids[labels]].sum())
        if objective < best_objective:
            best_objective = objective
            best_labels = labels.copy()
            best_medoids = medoids.copy()

    if best_labels is None or best_medoids is None:
        raise RuntimeError(
            "k-medoids did not produce a partition compatible with the requested "
            "minimum cluster size"
        )

    return best_labels, best_medoids


def smooth_training_order(ambient_daily_c, medoid_indices) -> np.ndarray:
    """Order medoids by a deterministic nearest-neighbour path."""
    x = _validate_daily_profiles(ambient_daily_c)
    medoids = np.asarray(medoid_indices, dtype=int)
    features = np.column_stack([x[medoids].mean(axis=1), x[medoids].std(axis=1)])
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    features = (features - features.mean(axis=0)) / scale

    start = int(np.argmin(x[medoids].mean(axis=1)))
    order = [start]
    remaining = set(range(len(medoids))) - {start}

    while remaining:
        current = order[-1]
        nxt = min(
            remaining,
            key=lambda j: float(np.linalg.norm(features[current] - features[j])),
        )
        order.append(nxt)
        remaining.remove(nxt)

    return medoids[np.asarray(order, dtype=int)]


def sample_validation_test(labels, medoid_indices, *, seed: int = 42):
    """Sample one validation and one test day from each cluster."""
    labels = np.asarray(labels, dtype=int)
    medoids = np.asarray(medoid_indices, dtype=int)
    rng = np.random.default_rng(seed)

    validation = []
    test = []
    for cluster, medoid in enumerate(medoids):
        candidates = np.flatnonzero(labels == cluster)
        candidates = candidates[candidates != medoid]
        if candidates.size < 2:
            raise ValueError(
                f"cluster {cluster} requires at least two non-medoid days"
            )
        chosen = rng.choice(candidates, size=2, replace=False)
        validation.append(int(chosen[0]))
        test.append(int(chosen[1]))

    return np.asarray(validation, dtype=int), np.asarray(test, dtype=int)


def build_representative_day_split(
    ambient_daily_c,
    *,
    n_clusters: int = 10,
    n_restarts: int = 100,
    seed: int = 42,
) -> RepresentativeDaySplit:
    x = _validate_daily_profiles(ambient_daily_c)
    labels, medoids = fixed_medoid_kmedoids(
        x,
        n_clusters=n_clusters,
        n_restarts=n_restarts,
        seed=seed,
        min_cluster_size=3,
    )
    validation, test = sample_validation_test(labels, medoids, seed=seed)
    return RepresentativeDaySplit(
        labels=labels,
        medoid_indices=medoids,
        training_order=smooth_training_order(x, medoids),
        validation_indices=validation,
        test_indices=test,
    )


def build_hot_year(ambient_daily_c, labels, *, hottest_cluster_boost_c: float = 2.0):
    """Construct the ten-day hot-year stress set described in Section 3.6."""
    x = _validate_daily_profiles(ambient_daily_c)
    labels = np.asarray(labels, dtype=int)

    selected = []
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        hottest = int(members[np.argmax(x[members].mean(axis=1))])
        selected.append(x[hottest].copy())

    hot_year = np.asarray(selected, dtype=float)
    hottest = int(np.argmax(hot_year.mean(axis=1)))
    hot_year[hottest] += float(hottest_cluster_boost_c)
    return hot_year
