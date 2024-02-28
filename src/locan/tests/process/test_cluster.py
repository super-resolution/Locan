from copy import deepcopy

import numpy as np
import pytest

from locan import (
    Bins,
    cluster_by_bin,
    cluster_dbscan,
    cluster_hdbscan,
    serial_clustering,
)
from locan.dependencies import HAS_DEPENDENCY


# tests hdbscan
@pytest.mark.skipif(not HAS_DEPENDENCY["hdbscan"], reason="Test requires hdbscan.")
def test_cluster_hdbscan_2d(locdata_two_cluster_2d):
    noise, clust = cluster_hdbscan(
        locdata_two_cluster_2d, min_cluster_size=2, allow_single_cluster=False
    )
    assert noise.data.empty
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_hdbscan(
        locdata_two_cluster_2d, min_cluster_size=5, allow_single_cluster=True
    )
    assert len(clust) == 1
    assert clust.data.localization_count[0] == 6
    assert all(np.in1d(clust.references[0].data.cluster_label, [1, 2]))
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_hdbscan(
        locdata_two_cluster_2d,
        loc_properties=["position_x"],
        min_cluster_size=2,
        allow_single_cluster=False,
    )
    assert noise.data.empty
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_2d.region


@pytest.mark.skipif(not HAS_DEPENDENCY["hdbscan"], reason="Test requires hdbscan.")
def test_cluster_hdbscan_2d_with_noise(locdata_two_cluster_with_noise_2d):
    noise, clust = cluster_hdbscan(
        locdata_two_cluster_with_noise_2d,
        min_cluster_size=2,
        allow_single_cluster=False,
    )
    assert len(noise) == 1
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region

    noise, clust = cluster_hdbscan(
        locdata_two_cluster_with_noise_2d, min_cluster_size=5, allow_single_cluster=True
    )
    assert len(noise) == 1
    assert len(clust) == 1
    assert clust.data.localization_count[0] == 6
    assert all(np.in1d(clust.references[0].data.cluster_label, [1, 2]))
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region


@pytest.mark.skipif(not HAS_DEPENDENCY["hdbscan"], reason="Test requires hdbscan.")
def test_cluster_hdbscan_with_shuffled_index(locdata_two_cluster_with_noise_2d):
    locdata = deepcopy(locdata_two_cluster_with_noise_2d)
    new_index = list(locdata.data.index)
    rng = np.random.default_rng()
    rng.shuffle(new_index)
    locdata.data.index = new_index
    noise, clust = cluster_hdbscan(
        locdata, min_cluster_size=2, allow_single_cluster=False
    )
    assert len(noise) == 1
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)


@pytest.mark.skipif(not HAS_DEPENDENCY["hdbscan"], reason="Test requires hdbscan.")
@pytest.mark.parametrize(
    "fixture_name, expected",
    [("locdata_empty", (0, 0)), ("locdata_single_localization", (1, 0))],
)
def test_cluster_hdbscan_empty_locdata(
    locdata_empty, locdata_single_localization, fixture_name, expected
):
    locdata = eval(fixture_name)
    noise, clust = cluster_hdbscan(
        locdata, min_cluster_size=2, allow_single_cluster=False
    )
    assert len(noise) == expected[0]
    assert len(clust) == expected[1]


# tests dbscan


def test_cluster_dbscan_2d(locdata_two_cluster_2d):
    noise, clust = cluster_dbscan(locdata_two_cluster_2d, eps=2, min_samples=1)
    assert noise.data.empty
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_dbscan(locdata_two_cluster_2d, eps=20, min_samples=1)
    assert len(clust) == 1
    assert clust.data.localization_count[0] == 6
    assert all(np.in1d(clust.references[0].data.cluster_label, [1, 2]))
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_dbscan(
        locdata_two_cluster_2d, loc_properties=["position_x"], eps=2, min_samples=1
    )
    assert noise.data.empty
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_dbscan(locdata_two_cluster_2d, eps=0.2, min_samples=10)
    assert len(noise) == len(locdata_two_cluster_2d)
    assert len(clust) == 0
    assert noise.region == locdata_two_cluster_2d.region
    assert clust.region is None


def test_cluster_dbscan_2d_with_noise(locdata_two_cluster_with_noise_2d):
    noise, clust = cluster_dbscan(
        locdata_two_cluster_with_noise_2d, eps=2, min_samples=1
    )
    assert noise.data.empty
    assert len(clust) == 3
    assert all(clust.data.localization_count == [3, 3, 1])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_with_noise_2d.region

    noise, clust = cluster_dbscan(
        locdata_two_cluster_with_noise_2d, eps=2, min_samples=2
    )
    assert len(noise) == 1
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region

    noise, clust = cluster_dbscan(
        locdata_two_cluster_with_noise_2d, eps=20, min_samples=2
    )
    assert len(noise) == 1
    assert len(clust) == 1
    assert clust.data.localization_count[0] == 6
    assert all(np.in1d(clust.references[0].data.cluster_label, [1, 2]))
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region


def test_cluster_dbscan_with_shuffled_index(locdata_two_cluster_with_noise_2d):
    locdata = deepcopy(locdata_two_cluster_with_noise_2d)
    new_index = list(locdata.data.index)
    rng = np.random.default_rng()
    rng.shuffle(new_index)
    locdata.data.index = new_index
    noise, clust = cluster_dbscan(locdata, eps=2, min_samples=2)
    assert len(noise) == 1
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)  #
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("locdata_empty", (0, 0)), ("locdata_single_localization", (1, 0))],
)
def test_cluster_dbscan_empty_locdata(
    locdata_empty, locdata_single_localization, fixture_name, expected
):
    locdata = eval(fixture_name)
    noise, clust = cluster_dbscan(locdata, eps=2, min_samples=2)
    assert len(noise) == expected[0]
    assert len(clust) == expected[1]


# tests serial_clustering


def test_serial_clustering(locdata_two_cluster_with_noise_2d):
    noise, clust = serial_clustering(
        locdata_two_cluster_with_noise_2d,
        cluster_dbscan,
        parameter_lists=dict(eps=[2, 20], min_samples=[2, 5]),
    )
    assert len(noise) == 4
    assert len(clust) == 4


# tests cluster_by_bin


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("locdata_empty", (np.array([]), 0, np.array([]))),
        ("locdata_single_localization", (np.array([[1, 1]]), 1, np.array([1]))),
    ],
)
def test_cluster_by_bin_empty_locdata(
    locdata_empty, locdata_single_localization, fixture_name, expected
):
    locdata = eval(fixture_name)
    bins, bin_indices, collection, counts = cluster_by_bin(
        locdata,
        loc_properties=["position_x", "position_y"],
        bin_size=2,
        return_counts=True,
    )
    assert bins is None or isinstance(bins, Bins)
    assert np.array_equal(bin_indices, expected[0])
    assert len(bin_indices) == len(collection)
    assert len(collection) == expected[1]
    assert np.array_equal(counts, expected[2])


def test_cluster_by_bin(locdata_2d):
    bins, bin_indices, collection, counts = cluster_by_bin(
        locdata_2d, loc_properties=["position_x", "position_y"], bin_size=5
    )
    assert bins.bin_size == (4, 5)
    assert np.array_equal(bin_indices, [[1, 1], [1, 2], [2, 1]])
    assert len(bin_indices) == len(collection)
    assert counts is None

    # test with min_samples
    bins, bin_indices, collection, counts = cluster_by_bin(
        locdata_2d,
        loc_properties=["position_x", "position_y"],
        bin_size=5,
        min_samples=2,
        return_counts=True,
    )
    assert bins.bin_size == (4, 5)
    assert np.array_equal(bin_indices, [[1, 1]])
    assert len(bin_indices) == len(collection)
    assert np.array_equal(counts, [4])
