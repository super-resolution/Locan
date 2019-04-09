import pytest
from pathlib import Path

import pandas as pd
import numpy as np

from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file
from surepy.data.cluster.clustering import cluster_hdbscan, cluster_dbscan
from surepy.data.cluster.utils import serial_clustering


# fixtures

@pytest.fixture()
def locdata():
    path = Path(ROOT_DIR + '/tests/Test_data/five_blobs.txt')
    dat = load_txt_file(path)
    return dat


@pytest.fixture()
def locdata_3d():
    path = Path(ROOT_DIR + '/tests/Test_data/five_blobs_3D.txt')
    dat = load_txt_file(path)
    return dat


# tests hdbscan

def test_cluster_hdbscan(locdata):
    # print(locdata.data.head())
    clust = cluster_hdbscan(locdata, min_cluster_size=5, allow_single_cluster=False)
    # print(clust.data.head())
    assert (len(clust) == 5)
    assert all(clust.references[0].data.cluster_label == 2)

    assert clust.data['subregion_measure_bb'].name == 'subregion_measure_bb'
    assert isinstance(clust.data['localization_count'], pd.Series)

    clust.dataframe = clust.dataframe.assign(test_column=range(5))
    assert isinstance(clust.data['test_column'], pd.Series)


def test_cluster_hdbscan_with_shuffled_index(locdata):
    new_index = list(locdata.data.index)
    np.random.shuffle(new_index)
    locdata.data.index = new_index
    clust = cluster_hdbscan(locdata, min_cluster_size=5, allow_single_cluster=False)
    assert (len(clust) == 5)
    assert all(clust.references[0].data.cluster_label == 2)


def test_cluster_hdbscan_with_noise(locdata):
    # print(locdata.data.head())
    clust = cluster_hdbscan(locdata, min_cluster_size=5, allow_single_cluster=False, noise=True)
    # print(clust[0].data)
    assert (len(clust) == 2)
    assert all(clust[0].data.cluster_label == 2)


def test_cluster_hdbscan_3d(locdata_3d):
    # print(locdata.data.head())
    clust = cluster_hdbscan(locdata_3d, min_cluster_size=5, allow_single_cluster=False)
    # print(clust.data.head())
    assert (len(clust) == 5)
    # clust.print_meta()


# tests dbscan

def test_cluster_dbscan(locdata):
    # print(locdata.data.head())
    clust = cluster_dbscan(locdata, eps=100, min_samples=4)
    # print(clust.data.head())
    assert (len(clust) == 5)
    assert all(clust.references[0].data.cluster_label == 3)


def test_cluster_dbscan_with_shuffled_index(locdata):
    new_index = list(locdata.data.index)
    np.random.shuffle(new_index)
    locdata.data.index = new_index
    clust = cluster_dbscan(locdata, eps=100, min_samples=4)
    assert (len(clust) == 5)
    assert all(clust.references[0].data.cluster_label == 3)


def test_cluster_dbscan_with_noise(locdata):
    # print(locdata.data.head())
    clust = cluster_dbscan(locdata, eps=100, min_samples=4, noise=True)
    assert (len(clust) == 2)
    assert all(clust[0].data.cluster_label == 3)


def test_cluster_dbscan_3d(locdata_3d):
    # print(locdata.data.head())
    clust = cluster_dbscan(locdata_3d, eps=100, min_samples=5)
    # print(clust.data.head())
    assert (len(clust) == 5)
    # clust.print_meta()


# tests serial_clustering

def test_serial_clustering(locdata):
    # print(locdata.data.head())
    noise, clust = serial_clustering(locdata, cluster_dbscan,
                                     parameter_lists=dict(eps=[10, 20], min_samples=[4, 5]))
    assert (noise is None)
    assert (len(clust) == 4)
    noise, clust = serial_clustering(locdata, cluster_dbscan,
                                     parameter_lists=dict(eps=[10, 20], min_samples=[4, 5]),
                                     noise=True)
    # print(noise.data.head())
    # print(clust.data.head())
    # print(clust.meta)
    assert (len(noise) == 4)
    assert (len(clust) == 4)
