import pytest
import matplotlib.pyplot as plt
import numpy as np
from shapely import affinity
from shapely.geometry import MultiPoint

from locan import make_Thomas
from locan import max_distance, compute_inertia_moments


def test_max_distance_2d(locdata_2d):
    mdist = max_distance(locdata=locdata_2d)
    assert (mdist == {'max_distance': 5.656854249492381})


def test_max_distance_3d(locdata_3d):
    mdist = max_distance(locdata=locdata_3d)
    assert (mdist == {'max_distance': 6.164414002968976})


@pytest.mark.skip('Test needs visual inspection.')
def test_compute_inertia_moments_visual():
    points, _ = make_Thomas(n_samples=100, centers=([0, 0], [10, 0]), cluster_std=1, seed=1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for angle in np.linspace(0, 180, 20):
        rotated_points = affinity.rotate(MultiPoint(points), angle, origin=[0, 0], use_radians=False)
        axes[0].scatter(*np.array(rotated_points).T)
        inertia_moments = compute_inertia_moments(np.array(rotated_points))
        axes[1].scatter(angle, inertia_moments.orientation)
    plt.show()


def test_compute_inertia_moments():
    points, _ = make_Thomas(n_samples=100, centers=([0, 0], [10, 0]), cluster_std=1, seed=1)
    for angle in np.linspace(0, 180, 5):
        rotated_points = affinity.rotate(MultiPoint(points), angle, origin=[0, 0], use_radians=False)
        inertia_moments = compute_inertia_moments(np.array(rotated_points))
        assert len(inertia_moments.eigenvalues) == 2
        assert len(inertia_moments.eigenvectors) == 2
        assert len(inertia_moments.variance_explained) == 2
        assert (inertia_moments.orientation+180) % 180 == pytest.approx(angle, abs=0.5) or \
               (inertia_moments.orientation+180) % 180 == pytest.approx(angle + 180, abs=0.5)
        assert inertia_moments.excentricity == pytest.approx(0.98, abs=0.2)


def test_compute_inertia_moments_3d():
    points, _ = make_Thomas(n_samples=100, centers=([0, 0, 0], [10, 0, 0]), cluster_std=1, n_features=3, seed=1)
    inertia_moments = compute_inertia_moments(np.array(points))
    assert len(inertia_moments.eigenvalues) == 3
    assert np.isnan(inertia_moments.orientation)
    assert np.isnan(inertia_moments.excentricity)
