import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from surepy import LocData
from surepy.constants import ROOT_DIR
from surepy.io.io_locdata import load_txt_file
from surepy.data.clustering import clustering_hdbscan


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

# tests

def test_clustering_hdbscan(locdata):
    #print(locdata.data.head())
    clust = clustering_hdbscan(locdata, min_cluster_size = 5, allow_single_cluster = False)
    #print(clust.data.head())
    assert (len(clust) == 5)

def test_clustering_hdbscan_3D(locdata_3D):
    #print(locdata.data.head())
    clust = clustering_hdbscan(locdata_3D, min_cluster_size = 5, allow_single_cluster = False)
    #print(clust.data.head())
    assert (len(clust) == 5)
