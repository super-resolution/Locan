import pytest
import numpy as np
import matplotlib.pyplot as plt

from surepy.render.render2d import _coordinate_ranges
from surepy.render import render_2d


def test__ranges(locdata_blobs_2d):
    hull_ranges = locdata_blobs_2d.bounding_box.hull
    assert np.array_equal(_coordinate_ranges(locdata_blobs_2d, ranges='auto'), hull_ranges.T)
    hull_ranges[0] = (0, 0)
    assert np.array_equal(_coordinate_ranges(locdata_blobs_2d, ranges='zero'), hull_ranges.T)
    with pytest.raises(ValueError):
        _coordinate_ranges(locdata_blobs_2d, ranges='test')
    assert np.array_equal(_coordinate_ranges(locdata_blobs_2d, ranges=((10, 100), (20, 200))), ((10, 100), (20, 200)))
    with pytest.raises(TypeError):
        _coordinate_ranges(locdata_blobs_2d, ranges=(10, 100))






def test_simple_rendering_2D(locdata_blobs_2d):
    # render_2d(locdata_blobs_2d)
    # render_2d(locdata_blobs_2d, bin_size=100, range=[[500, 1000], [500, 1000]], cbar=False)

    render_2d(locdata_blobs_2d, bin_size=100, ranges='auto', rescale=(0, 100))
    # render_2d(locdata_blobs_2d, bin_size=100, range='zero', rescale=(0, 100))
    # render_2d(locdata_blobs_2d, bin_size=100, range='auto', rescale='equal')
    # render_2d(locdata_blobs_2d, bin_size=100, range='auto', rescale=None)
    #
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # render_2d(locdata_blobs_2d, ax=ax[0])
    # render_2d(locdata_blobs_2d, range='zero', ax=ax[1])
    #
    # render_2d(locdata_blobs_2d, ax=ax[0], colorbar_kws=dict(ax=ax[0]))
    # render_2d(locdata_blobs_2d, range='zero', ax=ax[1])

    # plt.show()
