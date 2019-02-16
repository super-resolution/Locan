import pytest
import numpy as np
import pandas as pd

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
# from surepy.analysis import ClusterCheck
from surepy.analysis.density_based_cluster_check import _check_cluster, _analyze, DensityBasedClusterCheck

@pytest.fixture()
def locdata_blobs():
    return io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs.txt')

@pytest.fixture()
def locdata_blobs_3D():
    return io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs_3D.txt')


def test_analyze(locdata_blobs):
    res = _analyze(locdata_blobs, hull='bb')
    assert len(res) == 3
    # print(res)
    res = _analyze(locdata_blobs, hull='ch')
    assert len(res) == 3
    # print(res)

def test_check_cluster(locdata_blobs):
    results = _check_cluster(locdata_blobs, bins=3, hull='bb')
    assert all(results.columns == ['localization_density', 'eta', 'rho'])
    print(results)

    results = _check_cluster(locdata_blobs, bins=3, hull='ch')
    assert all(results.columns == ['localization_density', 'eta', 'rho'])
    print(results)

    results = _check_cluster(locdata_blobs, bins=3, hull='bb', divide='sequential')
    assert all(results.columns == ['localization_density', 'eta', 'rho'])
    print(results)

def test_DensityBasedClusterCheck(locdata_blobs):
    dbcc = DensityBasedClusterCheck(locdata_blobs, algo_parameter=dict(min_cluster_size=3, allow_single_cluster=False),
                                    bins=3, hull='bb').compute()
    print(dbcc.results)
    dbcc.plot()
