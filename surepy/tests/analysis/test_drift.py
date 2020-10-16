import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import matplotlib.pyplot as plt

from surepy import LocData
from surepy.constants import _has_open3d
from surepy.io.io_locdata import load_rapidSTORM_file
from surepy.constants import ROOT_DIR
from surepy.analysis.drift import Drift


@pytest.mark.skipif(not _has_open3d, reason="Test requires open3d.")
def test_Drift():
    locdata = load_rapidSTORM_file(path=ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt')

    drift = Drift(chunk_size=200, target='first').compute(locdata)
    assert isinstance(drift.collection, LocData)
    assert drift.results[0]._fields == ('matrix', 'offset')
    assert len(drift.results) == 4

    drift = Drift(chunk_size=200, target='previous').compute(locdata)
    assert isinstance(drift.collection, LocData)
    assert drift.results[0]._fields == ('matrix', 'offset')
    assert len(drift.results) == 4

    drift.plot(results_field='matrix', element=0)
    drift.plot(results_field='offset', element=None)
    # plt.show()
