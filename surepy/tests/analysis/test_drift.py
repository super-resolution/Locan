import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import matplotlib.pyplot as plt

from surepy import LocData
from surepy.io.io_locdata import load_rapidSTORM_file
from surepy.constants import ROOT_DIR
from surepy.data.drift import drift_correction
from surepy.analysis.drift import Drift


def test_Drift():
    locdata = load_rapidSTORM_file(path=ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt')
    columns = ['frame', 'intensity', 'chi_square', 'local_background']

    drift = Drift(chunk_size=200, target='first').compute(locdata)
    assert isinstance(drift.collection, LocData)
    assert drift.results._fields == ('matrices', 'offsets')
    assert len(drift.results.matrices) == 4

    drift = Drift(chunk_size=200, target='previous').compute(locdata)
    assert isinstance(drift.collection, LocData)
    assert drift.results._fields == ('matrices', 'offsets')
    assert len(drift.results.matrices) == 4
