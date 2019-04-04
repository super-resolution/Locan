import pytest
import numpy as np
import pandas as pd

from surepy import LocData
import surepy.constants
import surepy.io.io_locdata as io
import surepy.tests.test_data
from surepy.analysis import AccumulationClusterCheck
from surepy.analysis.accumulation_analysis import _accumulation_cluster_check, \
    _accumulation_cluster_check_for_single_dataset

@pytest.fixture()
def locdata_blobs():
    return io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs.txt')

@pytest.fixture()
def locdata_blobs_3D():
    return io.load_txt_file(path=surepy.constants.ROOT_DIR + '/tests/test_data/five_blobs_3D.txt')


def test___accumulation_cluster_check_for_single_dataset(locdata_blobs):
    res = _accumulation_cluster_check_for_single_dataset(locdata_blobs,
                                                         region_measure=locdata_blobs.bounding_box.region_measure,
                                                         hull='bb')
    assert len(res) == 3
    res = _accumulation_cluster_check_for_single_dataset(locdata_blobs,
                                                         region_measure=locdata_blobs.bounding_box.region_measure,
                                                         hull='ch')
    assert len(res) == 3


def test__accumulation_cluster_check(locdata_blobs):
    locdata_blobs.dataframe = locdata_blobs.data.drop(columns='index')
    results = _accumulation_cluster_check(locdata_blobs)
    assert len(results.columns) == 5

    results = _accumulation_cluster_check(locdata_blobs, n_loc=3, hull='bb', divide='sequential')
    assert len(results.columns) == 5

    results = _accumulation_cluster_check(locdata_blobs, n_loc=3, hull='ch', divide='sequential')
    assert len(results.columns) == 5

    results = _accumulation_cluster_check(locdata_blobs, region_measure=100, n_loc=3, divide='sequential')
    assert len(results.columns) == 5

    results = _accumulation_cluster_check(locdata_blobs, n_loc=[20, 30], divide='sequential')
    assert len(results.columns) == 5


def test_AccumulationClusterCheck(locdata_blobs):
    acc = AccumulationClusterCheck(algo_parameter=dict(min_cluster_size=3, allow_single_cluster=False),
                                    n_loc=5).compute(locdata_blobs)
    assert len(acc.results.columns) == 5
    acc = AccumulationClusterCheck(algo_parameter=dict(min_cluster_size=3, allow_single_cluster=False),
                                   n_loc=5, n_extrapolate=2).compute(locdata_blobs)
    assert len(acc.results.columns) == 5

    acc.plot(show=False)
