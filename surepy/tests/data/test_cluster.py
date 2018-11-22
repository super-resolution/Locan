import pytest
from pathlib import Path

from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file
from surepy.data.cluster.clustering import clustering_hdbscan, clustering_dbscan
from surepy.data.cluster.serial_clustering import serial_clustering


# fixtures

@pytest.fixture()
def locdata():
    path = Path(ROOT_DIR + '/tests/Test_data/five_blobs.txt')
    dat = load_txt_file(path)
    return dat

@pytest.fixture()
def locdata_3D():
    path = Path(ROOT_DIR + '/tests/Test_data/five_blobs_3D.txt')
    dat = load_txt_file(path)
    return dat

# tests hdbscan

def test_clustering_hdbscan(locdata):
    #print(locdata.data.head())
    clust = clustering_hdbscan(locdata, min_cluster_size = 5, allow_single_cluster = False)
    #print(clust.data.head())
    assert (len(clust) == 5)

def test_clustering_hdbscan_with_noise(locdata):
    #print(locdata.data.head())
    clust = clustering_hdbscan(locdata, min_cluster_size = 5, allow_single_cluster = False, noise=True)
    #print(clust.data.head())
    assert (len(clust) == 2)

def test_clustering_hdbscan_3D(locdata_3D):
    #print(locdata.data.head())
    clust = clustering_hdbscan(locdata_3D, min_cluster_size = 5, allow_single_cluster = False)
    #print(clust.data.head())
    assert (len(clust) == 5)
    # clust.print_meta()

# tests dbscan

def test_clustering_dbscan(locdata):
    #print(locdata.data.head())
    clust = clustering_dbscan(locdata, eps = 100, min_samples=4)
    #print(clust.data.head())
    assert (len(clust) == 5)

def test_clustering_dbscan_with_noise(locdata):
    #print(locdata.data.head())
    clust = clustering_dbscan(locdata, eps = 100, min_samples=4, noise=True)
    #print(clust.data.head())
    assert (len(clust) == 2)
    #print(clust[0].data.head())

def test_clustering_dbscan_3D(locdata_3D):
    #print(locdata.data.head())
    clust = clustering_dbscan(locdata_3D, eps = 100, min_samples=5)
    #print(clust.data.head())
    assert (len(clust) == 5)
    # clust.print_meta()

# tests serial_clustering

def test_serial_clustering(locdata):
    #print(locdata.data.head())
    noise, clust = serial_clustering(locdata, clustering_dbscan,
                                     parameter_lists=dict(eps=[10,20], min_samples=[4,5]))
    assert (noise is None)
    assert (len(clust) == 4)
    noise, clust = serial_clustering(locdata, clustering_dbscan,
                                     parameter_lists=dict(eps=[10, 20], min_samples=[4, 5]),
                                     noise=True)
    #print(noise.data.head())
    #print(clust.data.head())
    #print(clust.meta)
    assert (len(noise) == 4)
    assert (len(clust) == 4)