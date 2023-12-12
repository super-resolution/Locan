import pytest

import locan.constants
import locan.locan_io.locdata.io_locdata as io
from locan import LocData
from locan.analysis import AccumulationClusterCheck
from locan.analysis.accumulation_analysis import (
    _accumulation_cluster_check,
    _accumulation_cluster_check_for_single_dataset,
)
from locan.dependencies import HAS_DEPENDENCY


@pytest.fixture()
def locdata_blobs():
    return io.load_txt_file(path=locan.ROOT_DIR / "tests/test_data/five_blobs.txt")


@pytest.fixture()
def locdata_blobs_3D():
    return io.load_txt_file(path=locan.ROOT_DIR / "tests/test_data/five_blobs_3D.txt")


@pytest.mark.skipif(not HAS_DEPENDENCY["hdbscan"], reason="Test requires hdbscan.")
def test___accumulation_cluster_check_for_single_dataset(locdata_blobs):
    res = _accumulation_cluster_check_for_single_dataset(
        locdata_blobs,
        region_measure=locdata_blobs.bounding_box.region_measure,
        hull="bb",
    )
    assert len(res) == 3
    res = _accumulation_cluster_check_for_single_dataset(
        locdata_blobs,
        region_measure=locdata_blobs.bounding_box.region_measure,
        hull="ch",
    )
    assert len(res) == 3


@pytest.mark.skipif(not HAS_DEPENDENCY["hdbscan"], reason="Test requires hdbscan.")
def test__accumulation_cluster_check(locdata_blobs):
    results = _accumulation_cluster_check(locdata_blobs)
    assert len(results.columns) == 5

    results = _accumulation_cluster_check(
        locdata_blobs, n_loc=3, hull="bb", divide="sequential"
    )
    assert len(results.columns) == 5

    results = _accumulation_cluster_check(
        locdata_blobs, n_loc=3, hull="ch", divide="sequential"
    )
    assert len(results.columns) == 5

    results = _accumulation_cluster_check(
        locdata_blobs, region_measure=100, n_loc=3, divide="sequential"
    )
    assert len(results.columns) == 5

    results = _accumulation_cluster_check(
        locdata_blobs, n_loc=[20, 30], divide="sequential"
    )
    assert len(results.columns) == 5


@pytest.mark.skipif(not HAS_DEPENDENCY["hdbscan"], reason="Test requires hdbscan.")
def test_AccumulationClusterCheck_empty(caplog):
    acc = AccumulationClusterCheck().compute(LocData())
    acc.plot()
    assert caplog.record_tuples == [
        ("locan.analysis.accumulation_analysis", 30, "Locdata is empty.")
    ]


@pytest.mark.skipif(not HAS_DEPENDENCY["hdbscan"], reason="Test requires hdbscan.")
def test_AccumulationClusterCheck(locdata_blobs):
    acc = AccumulationClusterCheck(
        algo_parameter=dict(min_cluster_size=3, allow_single_cluster=False), n_loc=5
    ).compute(locdata_blobs)
    assert len(acc.results.columns) == 5
    acc = AccumulationClusterCheck(
        algo_parameter=dict(min_cluster_size=3, allow_single_cluster=False),
        n_loc=5,
        n_extrapolate=2,
    ).compute(locdata_blobs)
    assert len(acc.results.columns) == 5

    acc.plot()
