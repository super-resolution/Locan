import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import MultiPoint

from locan import (
    Ellipse,
    Rectangle,
    RoiRegion,
    distance_to_region,
    distance_to_region_boundary,
    inertia_moments,
    max_distance,
    regions_union,
    scatter_2d_mpl,
)
from locan.simulation import make_cluster, make_uniform


def test_distance_to_region_RoiRegion(locdata_2d):
    region = RoiRegion(region_type="rectangle", region_specs=((1, 1), 3, 3, 0))
    # visualize
    ax = scatter_2d_mpl(locdata_2d, index=True, marker="o", color="g")
    ax.add_patch(region.as_artist(fill=False))
    # plt.show()
    distances = distance_to_region(locdata_2d, region)
    assert np.array_equal(distances[:-1], np.array([0, 1, 0, 2, 0]))

    region_2 = RoiRegion(region_type="rectangle", region_specs=((3, 3), 3, 3, 0))
    distances = distance_to_region(locdata_2d, regions_union([region, region_2]))
    assert np.array_equal(distances[:-1], np.array([0, 1, 0, 0, 0]))

    plt.close("all")


def test_distance_to_region(locdata_2d):
    region = Rectangle((1, 1), 3, 3, 0)
    # visualize
    ax = scatter_2d_mpl(locdata_2d, index=True, marker="o", color="g")
    ax.add_patch(region.as_artist(fill=False))
    # plt.show()
    distances = distance_to_region(locdata_2d, region)
    assert np.array_equal(distances[:-1], np.array([0, 1, 0, 2, 0]))

    region_2 = Rectangle((3, 3), 3, 3, 0)
    distances = distance_to_region(locdata_2d, regions_union([region, region_2]))
    assert np.array_equal(distances[:-1], np.array([0, 1, 0, 0, 0]))

    plt.close("all")


def test_distance_to_region_boundary_RoiRegion(locdata_2d):
    region = RoiRegion(region_type="rectangle", region_specs=((1, 1), 3, 3, 0))
    # visualize
    # ax = scatter_2d_mpl(locdata_2d, index=True, marker='o', color='g')
    # ax.add_patch(region.as_artist(fill=False))
    # plt.show()
    distances = distance_to_region_boundary(locdata_2d, region)
    assert np.array_equal(distances[:-1], np.array([0, 1, 1, 2, 0]))

    region_2 = RoiRegion(region_type="rectangle", region_specs=((3.5, 3.5), 3, 3, 0))
    distances = distance_to_region(locdata_2d, regions_union([region, region_2]))
    assert np.array_equal(distances[:-1], np.array([0, 1, 0, 0.5, 0]))

    plt.close("all")


def test_distance_to_region_boundary(locdata_2d):
    region = Rectangle((1, 1), 3, 3, 0)
    # visualize
    # ax = scatter_2d_mpl(locdata_2d, index=True, marker='o', color='g')
    # ax.add_patch(region.as_artist(fill=False))
    # plt.show()
    distances = distance_to_region_boundary(locdata_2d, region)
    assert np.array_equal(distances[:-1], np.array([0, 1, 1, 2, 0]))

    region_2 = RoiRegion(region_type="rectangle", region_specs=((3.5, 3.5), 3, 3, 0))
    distances = distance_to_region(locdata_2d, regions_union([region, region_2]))
    assert np.array_equal(distances[:-1], np.array([0, 1, 0, 0.5, 0]))

    plt.close("all")


def test_max_distance_2d(locdata_2d):
    locdata_2d = deepcopy(locdata_2d)
    mdist = max_distance(locdata=locdata_2d)
    assert mdist == {"max_distance": 5.656854249492381}


def test_max_distance_3d(locdata_3d):
    locdata_3d = deepcopy(locdata_3d)
    mdist = max_distance(locdata=locdata_3d)
    assert mdist == {"max_distance": 6.164414002968976}


@pytest.mark.visual
def test_compute_inertia_moments_visual():
    rng = np.random.default_rng(seed=1)
    offspring = [make_uniform(n_samples=10, region=Ellipse((0, 0), 1, 1), seed=rng)] * 2
    points, _, _, _ = make_cluster(
        centers=([0, 0], [10, 0]),
        region=((0, 10), (0, 10)),
        offspring=offspring,
        clip=False,
        seed=rng,
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for angle in np.linspace(0, 180, 20):
        rotated_points = affinity.rotate(
            MultiPoint(points), angle, origin=[0, 0], use_radians=False
        )
        rotated_points = np.array([rpts.coords[0] for rpts in rotated_points.geoms])
        axes[0].scatter(*rotated_points.T)
        inertia_moments_ = inertia_moments(rotated_points)
        axes[1].scatter(angle, inertia_moments_.orientation)
    plt.show()


def test_compute_inertia_moments():
    rng = np.random.default_rng(seed=1)
    offspring = [
        make_uniform(n_samples=100, region=Ellipse((0, 0), 1, 1), seed=rng)
    ] * 2
    points, _, _, _ = make_cluster(
        centers=([0, 0], [10, 0]),
        region=((0, 10), (0, 10)),
        offspring=offspring,
        clip=False,
        seed=rng,
    )
    for angle in np.linspace(0, 180, 5):
        rotated_points = affinity.rotate(
            MultiPoint(points), angle, origin=[0, 0], use_radians=False
        )
        rotated_points = np.array([rpts.coords[0] for rpts in rotated_points.geoms])
        inertia_moments_ = inertia_moments(rotated_points)
        assert len(inertia_moments_.eigenvalues) == 2
        assert len(inertia_moments_.eigenvectors) == 2
        assert len(inertia_moments_.variance_explained) == 2

        assert (inertia_moments_.orientation + 180) % 180 == pytest.approx(
            (angle + 180.000001) % 180, abs=0.01
        )
        assert (inertia_moments_.orientation + 180) % 180 == pytest.approx(
            (angle + 180.000001) % 180, abs=0.01
        )
        assert inertia_moments_.eccentricity == pytest.approx(0.98, abs=0.1)


def test_compute_inertia_moments_3d(caplog):
    rng = np.random.default_rng(seed=1)
    offspring = [rng.normal(size=(100, 3), scale=1)] * 2
    points, _, _, _ = make_cluster(
        centers=([0, 0, 0], [10, 0, 0]),
        region=((0, 10), (0, 10), (0, 10)),
        offspring=offspring,
        clip=False,
        seed=rng,
    )

    inertia_moments_ = inertia_moments(np.array(points))
    assert len(inertia_moments_.eigenvalues) == 3
    assert np.isnan(inertia_moments_.orientation)
    assert np.isnan(inertia_moments_.eccentricity)
    assert caplog.record_tuples == [
        (
            "locan.data.properties.misc",
            logging.WARNING,
            "Orientation and eccentricity have not yet been implemented for 3D.",
        )
    ]
