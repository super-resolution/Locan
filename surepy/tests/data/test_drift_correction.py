import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import matplotlib.pyplot as plt

from surepy import LocData
from surepy.io.io_locdata import load_rapidSTORM_file
from surepy.constants import ROOT_DIR
from surepy.data.drift import drift_correction


def test_drift_correction():
    locdata = load_rapidSTORM_file(path=ROOT_DIR / 'tests/test_data/rapidSTORM_dstorm_data.txt')
    columns = ['frame', 'intensity', 'chi_square', 'local_background']

    locdata_corrected = drift_correction(locdata, chunk_size=200, target='first')
    assert len(locdata_corrected) == len(locdata) == 999
    assert_frame_equal(locdata_corrected.data[columns], locdata.data[columns])

    locdata_corrected = drift_correction(locdata, chunk_size=200, target='previous')
    assert len(locdata_corrected) == len(locdata)
    assert_frame_equal(locdata_corrected.data[columns], locdata.data[columns])
