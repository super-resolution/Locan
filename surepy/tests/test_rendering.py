import pytest
import numpy as np
import matplotlib.pyplot as plt  # this import is needed for interactive tests

from surepy.constants import RenderEngine  # this import is needed for interactive tests
from surepy.constants import _has_mpl_scatter_density, _has_napari
if _has_napari: import napari
from surepy.render.render2d import _coordinate_ranges, _bin_edges, _bin_edges_from_size, _bin_edges_to_number, \
    render_2d_mpl, render_2d_scatter_density, render_2d_napari
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


def test__bin_edges():
    bin_edges = _bin_edges(10, (10, 20))
    assert bin_edges.shape == (11,)

    bin_edges = _bin_edges(10, [10, 20])
    assert bin_edges.shape == (11,)

    bin_edges = _bin_edges((5, 10), (10, 20))
    assert bin_edges.shape == (2,)
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (11,)

    bin_edges = _bin_edges(5, ((10, 20), (20, 30)))
    assert bin_edges.shape == (2, 6)

    bin_edges = _bin_edges((5, 10), ((10, 20), (20, 30)))
    assert bin_edges.shape == (2,)
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (11,)

    bin_edges = _bin_edges((3, (5, 10)), ((1, 5), ((10, 20), (20, 30))))
    assert bin_edges.shape == (2,)
    assert bin_edges[0].shape == (4,)
    assert bin_edges[1].shape == (2,)
    assert bin_edges[1][0].shape == (6,)
    assert bin_edges[1][1].shape == (11,)


def test__bin_edges_from_size():
    bin_edges = _bin_edges_from_size(1, (10, 20))
    assert bin_edges.shape == (11,)

    bin_edges = _bin_edges_from_size(5, (0, 17))
    assert bin_edges.shape == (5,)
    assert bin_edges[-1] > 17

    bin_edges = _bin_edges_from_size(5, (0, 17), extend_range=False)
    assert bin_edges.shape == (5,)
    assert bin_edges[-1] == 17

    bin_edges = _bin_edges_from_size(5, (0, 17), extend_range=None)
    assert bin_edges.shape == (4,)
    assert bin_edges[-1] < 17

    bin_edges = _bin_edges_from_size(1, [10, 20])
    assert bin_edges.shape == (11,)

    bin_edges = _bin_edges_from_size((2, 1), (10, 20))
    assert bin_edges.shape == (2,)
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (11,)

    bin_edges = _bin_edges_from_size(2, ((10, 20), (20, 30)))
    assert bin_edges.shape == (2, 6)

    bin_edges = _bin_edges_from_size((2, 1), ((10, 20), (20, 30)))
    assert bin_edges.shape == (2,)
    assert bin_edges[0].shape == (6,)
    assert bin_edges[1].shape == (11,)

    bin_edges = _bin_edges_from_size((2, (2, 1)), ((1, 5), ((10, 20), (20, 30))))
    assert bin_edges.shape == (2,)
    assert bin_edges[0].shape == (3,)
    assert bin_edges[1].shape == (2,)
    assert bin_edges[1][0].shape == (6,)
    assert bin_edges[1][1].shape == (11,)


def test__bin_edges_to_number():
    n_bins = _bin_edges_to_number([1, 3, 5])
    assert n_bins == 2

    with pytest.warns(UserWarning):
        n_bins = _bin_edges_to_number([1, 2, 4])
    assert n_bins is None

    with pytest.warns(UserWarning):
        n_bins = _bin_edges_to_number(((1, 3, 5), (1, 2, 4)))
    assert np.array_equal(n_bins, (2, None))

    n_bins = _bin_edges_to_number([[1, 3, 5], [1, 2, 3, 4]])
    assert np.array_equal(n_bins, (2, 3))


def test_adjust_contrast():
    img = np.array((1, 2, 3, 9), dtype=np.float64)
    new_img = adjust_contrast(img, rescale=(50, 100))
    assert np.array_equal(new_img, np.array((0, 0, 0, 1)))

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
    assert np.array_equal(new_img, np.array((0, 63, 127, 255)))

    new_img = adjust_contrast(img, out_range=(0, 10))
    assert np.array_equal(new_img, np.array((0, 1, 2, 10)))

    new_img = adjust_contrast(img * 1., rescale=True)
    assert np.array_equal(new_img, np.array((0, 0.125, 0.25, 1.)))

    new_img = adjust_contrast(img, rescale='unity')
    assert np.array_equal(new_img, np.array((0, 0.125, 0.25, 1.)))


def test_histogram(locdata_blobs_2d):
    # to do: fix bug
    # This test is ok when run by itself also with un-commented assert statements.
    # In the module sequence however, range is taken as 'zero'. Why?
    hist, ranges, bins, labels = histogram(locdata_blobs_2d)
    assert 'counts' in labels
    # assert locdata_blobs_2d.data['position_x'].min() == ranges[0][0]
    assert hist.dtype == 'float64'

    hist, ranges, bins, labels = histogram(locdata_blobs_2d, bins=10, bin_size=None)
    # print(locdata_blobs_2d.data['position_x'].min())
    assert 'counts' in labels
    # assert locdata_blobs_2d.data['position_x'].min() == ranges[0][0]
    # assert np.max(hist) == 7

    hist, ranges, bins, labels = histogram(locdata_blobs_2d, range='zero', rescale=True)
    assert np.max(hist) == 1
    assert 'counts' in labels

    hist, ranges, bins, labels = histogram(locdata_blobs_2d, loc_properties='position_x')
    assert 'counts' in labels
    assert hist.shape == (101,)

    hist, ranges, bins, labels = histogram(locdata_blobs_2d, loc_properties=('position_x', 'position_y'))
    assert 'counts' in labels
    assert hist.shape == (102, 101)

    hist, ranges, bins, labels = histogram(locdata_blobs_2d, other_property='position_y')
    assert 'position_y' in labels
    assert hist.shape == (102, 101)


def test_render_2d_mpl(locdata_blobs_2d):
    # render_2d_mpl(locdata_blobs_2d)
    # render_2d_mpl(locdata_blobs_2d, bin_size=100, range=[[500, 1000], [500, 1000]], cbar=False)

    render_2d_mpl(locdata_blobs_2d, bin_size=100, range=None, rescale=None)
    # render_2d_mpl(locdata_blobs_2d, bin_size=100, range=None, rescale=True)
    # render_2d_mpl(locdata_blobs_2d, bin_size=100, range=None, rescale='unity')
    # render_2d_mpl(locdata_blobs_2d, bin_size=100, range=None, rescale='equal')
    #render_2d_mpl(locdata_blobs_2d, bin_size=100, range=None, rescale=(0, 50))
    # render_2d_mpl(locdata_blobs_2d, bin_size=100, range='zero')
    #
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # render_2d_mpl(locdata_blobs_2d, ax=ax[0])
    # render_2d_mpl(locdata_blobs_2d, range='zero', ax=ax[1])
    #
    # render_2d_mpl(locdata_blobs_2d, ax=ax[0], colorbar_kws=dict(ax=ax[0]))
    # render_2d_mpl(locdata_blobs_2d, range='zero', ax=ax[1])

    # plt.show()


@pytest.mark.skipif(not _has_mpl_scatter_density, reason="requires mpl_scatter_density")
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


@pytest.mark.parametrize("test_input, expected", [
    ("RenderEngine.MPL", 0),
    ("RenderEngine.MPL_SCATTER_DENSITY", 0)
])
def test_render_2d(locdata_blobs_2d, test_input, expected):
    render_2d(locdata_blobs_2d, render_engine=test_input)
    # plt.show()


@pytest.mark.skip('GUI tests are skipped because they would need user interaction.')
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_render_2d_napari(locdata_blobs_2d):
    with napari.gui_qt():
        render_2d_napari(locdata_blobs_2d, bin_size=100, cmap='viridis', gamma=0.1)

    # with napari.gui_qt():
    #     viewer = render_2d_napari(locdata_blobs_2d, bin_size=50, cmap='magenta', gamma=0.1)
    #     render_2d_napari(locdata_blobs_2d, viewer=viewer, bin_size=100, cmap='cyan', gamma=0.1, scale=(2, 2),
    #                      blending='additive')
    #
    # with napari.gui_qt():
    #     render_2d(locdata_blobs_2d, render_engine=RenderEngine.NAPARI)
