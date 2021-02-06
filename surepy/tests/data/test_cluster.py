from copy import deepcopy

import pytest
import numpy as np

from surepy.data.cluster.clustering import cluster_hdbscan, cluster_dbscan
from surepy.data.cluster.utils import serial_clustering


# tests hdbscan

def test_cluster_hdbscan_2d(locdata_two_cluster_2d):
    noise, clust = cluster_hdbscan(locdata_two_cluster_2d, min_cluster_size=2, allow_single_cluster=False)
    assert noise is None
    assert (len(clust) == 2)
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)

    noise, clust = cluster_hdbscan(locdata_two_cluster_2d, min_cluster_size=5, allow_single_cluster=True)
    assert (len(clust) == 1)
    assert clust.data.localization_count[0] == 6
    assert all(np.in1d(clust.references[0].data.cluster_label, [1, 2]))

    noise, clust = cluster_hdbscan(locdata_two_cluster_2d, loc_properties=['position_x'],
                                   min_cluster_size=2, allow_single_cluster=False)
    assert noise is None
    assert (len(clust) == 2)
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)


def test_cluster_hdbscan_2d_with_noise(locdata_two_cluster_with_noise_2d):
    noise, clust = cluster_hdbscan(locdata_two_cluster_with_noise_2d, min_cluster_size=2, allow_single_cluster=False)
    assert len(noise) == 1
    assert (len(clust) == 2)
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)

    noise, clust = cluster_hdbscan(locdata_two_cluster_with_noise_2d, min_cluster_size=5, allow_single_cluster=True)
    assert len(noise) == 1
    assert (len(clust) == 1)
    assert clust.data.localization_count[0] == 6
    assert all(np.in1d(clust.references[0].data.cluster_label, [1, 2]))


def test_cluster_hdbscan_with_shuffled_index(locdata_two_cluster_with_noise_2d):
    locdata = deepcopy(locdata_two_cluster_with_noise_2d)
    new_index = list(locdata.data.index)
    np.random.shuffle(new_index)
    locdata.data.index = new_index
    noise, clust = cluster_hdbscan(locdata, min_cluster_size=2, allow_single_cluster=False)
    assert len(noise) == 1
    assert (len(clust) == 2)
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)

@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', (0, 0)),
    ('locdata_single_localization', (1, 0))
])
def test_cluster_hdbscan_empty_locdata(
        locdata_empty, locdata_single_localization,
        fixture_name, expected):
    locdata = eval(fixture_name)
    noise, clust = cluster_hdbscan(locdata, min_cluster_size=2, allow_single_cluster=False)
    assert len(noise) == expected[0]
    assert len(clust) == expected[1]


# tests dbscan

def test_cluster_dbscan_2d(locdata_two_cluster_2d):
    noise, clust = cluster_dbscan(locdata_two_cluster_2d, eps=2, min_samples=1)
    assert noise is None
    assert (len(clust) == 2)
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)

    noise, clust = cluster_dbscan(locdata_two_cluster_2d, eps=20, min_samples=1)
    assert (len(clust) == 1)
    assert clust.data.localization_count[0] == 6
    assert all(np.in1d(clust.references[0].data.cluster_label, [1, 2]))

    noise, clust = cluster_dbscan(locdata_two_cluster_2d, loc_properties=['position_x'], eps=2, min_samples=1)
    assert noise is None
    assert (len(clust) == 2)
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)


def test_cluster_dbscan_2d_with_noise(locdata_two_cluster_with_noise_2d):
    noise, clust = cluster_dbscan(locdata_two_cluster_with_noise_2d, eps=2, min_samples=1)
    assert noise is None
    assert (len(clust) == 3)
    assert all(clust.data.localization_count == [3, 3, 1])
    assert all(clust.references[0].data.cluster_label == 1)

    noise, clust = cluster_dbscan(locdata_two_cluster_with_noise_2d, eps=2, min_samples=2)
    assert len(noise) == 1
    assert (len(clust) == 2)
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)

    noise, clust = cluster_dbscan(locdata_two_cluster_with_noise_2d, eps=20, min_samples=2)
    assert len(noise) == 1
    assert (len(clust) == 1)
    assert clust.data.localization_count[0] == 6
    assert all(np.in1d(clust.references[0].data.cluster_label, [1, 2]))


def test_cluster_dbscan_with_shuffled_index(locdata_two_cluster_with_noise_2d):
    locdata = deepcopy(locdata_two_cluster_with_noise_2d)
    new_index = list(locdata.data.index)
    np.random.shuffle(new_index)
    locdata.data.index = new_index
    noise, clust = cluster_dbscan(locdata, eps=2, min_samples=2)
    assert len(noise) == 1
    assert (len(clust) == 2)
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)


@pytest.mark.parametrize('fixture_name, expected', [
    ('locdata_empty', (0, 0)),
    ('locdata_single_localization', (1, 0))
])
def test_cluster_dbscan_empty_locdata(
        locdata_empty, locdata_single_localization,
        fixture_name, expected):
    locdata = eval(fixture_name)
    noise, clust = cluster_dbscan(locdata, eps=2, min_samples=2)
    assert len(noise) == expected[0]
    assert len(clust) == expected[1]


# tests serial_clustering

def test_serial_clustering(locdata_two_cluster_with_noise_2d):
    noise, clust = serial_clustering(locdata_two_cluster_with_noise_2d, cluster_dbscan,
                                     parameter_lists=dict(eps=[2, 20], min_samples=[2, 5]))
    assert (len(noise) == 4)
    assert (len(clust) == 4)
