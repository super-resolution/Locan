import pytest
import numpy as np

from surepy.render.utilities import _coordinate_ranges


def test__ranges(locdata_blobs_2d):
    hull_ranges = locdata_blobs_2d.bounding_box.hull
    assert np.array_equal(_coordinate_ranges(locdata_blobs_2d, range=None), hull_ranges.T)
    hull_ranges[0] = (0, 0)
    assert np.array_equal(_coordinate_ranges(locdata_blobs_2d, range='zero'), hull_ranges.T)
    with pytest.raises(ValueError):
        _coordinate_ranges(locdata_blobs_2d, range='test')
    assert np.array_equal(_coordinate_ranges(locdata_blobs_2d, range=((10, 100), (20, 200))), ((10, 100), (20, 200)))
    with pytest.raises(TypeError):
        _coordinate_ranges(locdata_blobs_2d, range=(10, 100))
