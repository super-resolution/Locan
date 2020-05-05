import pytest
import numpy as np
import pandas as pd

from surepy import LocData
from surepy.constants import _has_trackpy
if _has_trackpy:
    from surepy.data.tracking import link_locdata, track

pytestmark = pytest.mark.skipif(not _has_trackpy, reason="requires trackpy")


@pytest.fixture()
def locdata_simple():
    dict = {
        'position_x': [0, 1, 2, 10, 20, 21, 30, 4],
        'position_y': [0, 1, 2, 10, 20, 21, 30, 4],
        'position_z': [0, 1, 2, 10, 20, 21, 30, 4],
        'frame': np.arange(8),

    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict))

def test_link_locdata(locdata_simple):
    track_series = link_locdata(locdata_simple, search_range=5, memory=0)
    assert (len(track_series) == 8)
    assert(track_series.name=='track')


def test_track(locdata_simple):
    locdata_new, track_series = track(locdata_simple, search_range=5)
    #print(locdata_new.data)
    assert (len(locdata_new) == 5)
    locdata_new, track_series = track(locdata_simple, search_range=5, memory=5)
    #print(locdata_new.data)
    assert (len(locdata_new) == 4)
