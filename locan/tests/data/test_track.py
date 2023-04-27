import numpy as np
import pandas as pd
import pytest

from locan import LocData
from locan.dependencies import HAS_DEPENDENCY

if HAS_DEPENDENCY["trackpy"]:
    from trackpy import quiet as tp_quiet

    from locan.data.tracking import link_locdata, track


pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCY["trackpy"], reason="requires trackpy"
)

if HAS_DEPENDENCY["trackpy"]:
    tp_quiet()  # same as: trackpy.logger.setLevel(logging.WARN)


@pytest.fixture()
def locdata_simple():
    dict_ = {
        "position_x": [0, 1, 2, 10, 20, 21, 30, 4],
        "position_y": [0, 1, 2, 10, 20, 21, 30, 4],
        "position_z": [0, 1, 2, 10, 20, 21, 30, 4],
        "frame": np.arange(8),
    }
    return LocData(dataframe=pd.DataFrame.from_dict(dict_))


def test_link_locdata(locdata_simple):
    track_series = link_locdata(locdata_simple, search_range=5, memory=0)
    assert len(track_series) == 8
    assert track_series.name == "track"


def test_track(locdata_simple):
    locdata_new, track_series = track(locdata_simple, search_range=5)
    expected = {
        "localization_count": 3,
        "position_x": 1.0,
        "uncertainty_x": 0.5773502691896257,
        "position_y": 1.0,
        "uncertainty_y": 0.5773502691896257,
        "position_z": 1.0,
        "uncertainty_z": 0.5773502691896257,
        "frame": 0,
        "region_measure_bb": 8,
        "localization_density_bb": 0.375,
        "subregion_measure_bb": 12,
    }
    for value, value_expected, values_in_data in zip(
        locdata_new.references[0].properties.values(),
        expected.values(),
        locdata_new.data.iloc[0].to_numpy(),
    ):
        assert value == value_expected == values_in_data

    expected = {
        "localization_count": 1,
        "position_x": 10,
        "uncertainty_x": 0,
        "position_y": 10,
        "uncertainty_y": 0,
        "position_z": 10,
        "uncertainty_z": 0,
        "frame": 3,
        "region_measure_bb": 0,
    }
    for value, value_expected, values_in_data in zip(
        locdata_new.references[1].properties.values(),
        expected.values(),
        locdata_new.data.iloc[1].to_numpy(),
    ):
        assert value == value_expected == values_in_data

    assert "frame" in locdata_new.data.columns
    assert len(locdata_new) == 5

    locdata_new, track_series = track(locdata_simple, search_range=5, memory=5)
    assert len(locdata_new) == 4
