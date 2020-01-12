import pytest
import numpy as np
import matplotlib.pyplot as plt

from surepy.constants import RenderEngine
from surepy.render.render2d import _coordinate_ranges, _bin_number, render_2d_mpl, render_2d_scatter_density
from surepy.render import adjust_contrast, histogram, render_2d


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


def test__bin_number():
    assert _bin_number(range=(10, 100), bin_size=10) == 9
    assert np.array_equal(_bin_number(range=(10, 100), bin_size=(10, 5)), (9, 18))
    assert np.array_equal(_bin_number(range=((10, 100), (0, 100)), bin_size=10), (9, 10))
    assert np.array_equal(_bin_number(range=((10, 100), (0, 100)), bin_size=(10, 5)), (9, 20))


def test_adjust_contrast():
    img = np.array((1, 2, 3, 9), dtype=np.uint8)
    new_img = adjust_contrast(img, rescale=None)
    assert np.array_equal(img, new_img)
    new_img = adjust_contrast(img, rescale=False)
    assert np.array_equal(img, new_img)
    new_img = adjust_contrast(img, rescale='equal')
    assert max(new_img) == 1
    new_img = adjust_contrast(img, rescale=True)
    assert np.array_equal(new_img, np.array((0, 31, 63, 255)))
    new_img = adjust_contrast(img, rescale=(0, 50))
    assert np.array_equal(new_img, np.array((0, 170, 255, 255)))
    new_img = adjust_contrast(img, out_range=(0, 10))
    assert np.array_equal(new_img, np.array((0, 1, 2, 10)))
    new_img = adjust_contrast(img * 1., rescale=True)
    assert np.array_equal(new_img, np.array((0, 0.125, 0.25, 1.)))
    new_img = adjust_contrast(img, rescale='unity')
    assert np.array_equal(new_img, np.array((0, 0.125, 0.25, 1.)))


def test_histogram(locdata_blobs_2d):
    hist, ranges, bins, labels = histogram(locdata_blobs_2d)
    assert 'counts' in labels

    hist, ranges, bins, labels = histogram(locdata_blobs_2d, bins=10, bin_size=None)
    assert 'counts' in labels
    assert np.max(hist) == 9

    hist, range, bins, labels = histogram(locdata_blobs_2d, range='zero', rescale=True)
    assert np.max(hist) == 1
    assert 'counts' in labels

    hist, range, bins, labels = histogram(locdata_blobs_2d, loc_properties='position_x')
    assert 'counts' in labels
    assert hist.shape == (101,)

    hist, range, bins, labels = histogram(locdata_blobs_2d, loc_properties=('position_x', 'position_y'))
    assert 'counts' in labels
    assert hist.shape == (102, 101)

    hist, range, bins, labels = histogram(locdata_blobs_2d, other_property='position_y')
    assert 'position_y' in labels
    assert hist.shape == (102, 101)


def test_render_2d_mpl(locdata_blobs_2d):
    # render_2d_mpl(locdata_blobs_2d)
    # render_2d_mpl(locdata_blobs_2d, bin_size=100, range=[[500, 1000], [500, 1000]], cbar=False)

    render_2d_mpl(locdata_blobs_2d, bin_size=100, range=None, rescale=(0, 100))
    # render_2d_mpl(locdata_blobs_2d, bin_size=100, range='zero', rescale=(0, 100))
    # render_2d_mpl(locdata_blobs_2d, bin_size=100, range=None, rescale='equal')
    # render_2d_mpl(locdata_blobs_2d, bin_size=100, range=None, rescale=None)
    #
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # render_2d_mpl(locdata_blobs_2d, ax=ax[0])
    # render_2d_mpl(locdata_blobs_2d, range='zero', ax=ax[1])
    #
    # render_2d_mpl(locdata_blobs_2d, ax=ax[0], colorbar_kws=dict(ax=ax[0]))
    # render_2d_mpl(locdata_blobs_2d, range='zero', ax=ax[1])

    # plt.show()

def test_render_2d_scatter_density(locdata_blobs_2d):
    render_2d_scatter_density(locdata_blobs_2d)
    # render_2d_scatter_density(locdata_blobs_2d, range=[[500, 1000], [500, 1000]], cbar=False)

    # render_2d_scatter_density(locdata_blobs_2d, range=None, vmin=0, vmax=1)

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # render_2d_scatter_density(locdata_blobs_2d, ax=ax[0])
    # render_2d_scatter_density(locdata_blobs_2d, range='zero', ax=ax[1])
    #
    # render_2d_scatter_density(locdata_blobs_2d, ax=ax[0], colorbar_kws=dict(ax=ax[0]))
    # render_2d_scatter_density(locdata_blobs_2d, range='zero', ax=ax[1])

    # render_2d_scatter_density(locdata_blobs_2d, other_property='position_x')

    # plt.show()


def test_render_2d(locdata_blobs_2d):
    render_2d(locdata_blobs_2d)
    render_2d(locdata_blobs_2d, render_engine=RenderEngine.MPL_SCATTER_DENSITY)
    # plt.show()
