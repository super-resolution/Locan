import pytest
import numpy as np
import matplotlib.pyplot as plt  # this import is needed for interactive tests

from locan import RenderEngine  # this import is needed for interactive tests
from locan.dependencies import HAS_DEPENDENCY
from locan import render_2d_mpl, render_2d_scatter_density, scatter_2d_mpl, render_2d_rgb_mpl
from locan import render_2d, apply_window
from locan import cluster_dbscan, transform_affine


def test_render_2d_mpl_empty(locdata_empty):
    render_2d_mpl(locdata_empty, bin_size=5)
    # plt.show()

    plt.close('all')


def test_render_2d_mpl_single(locdata_single_localization, caplog):
    render_2d_mpl(locdata_single_localization, bin_size=0.5)
    assert caplog.record_tuples == [('locan.render.render2d', 30, 'Locdata carries a single localization.')]
    # plt.show()

    plt.close('all')


def test_render_2d_mpl(locdata_blobs_2d):
    render_2d_mpl(locdata_blobs_2d)
    render_2d_mpl(locdata_blobs_2d, bin_size=100, bin_range=[[500, 1000], [500, 1000]], cbar=False)

    render_2d_mpl(locdata_blobs_2d, bin_size=100, bin_range=None, rescale=None)
    render_2d_mpl(locdata_blobs_2d, bin_size=100, bin_range=None, rescale=True)
    render_2d_mpl(locdata_blobs_2d, bin_size=100, bin_range=None, rescale='unity')
    render_2d_mpl(locdata_blobs_2d, bin_size=100, bin_range=None, rescale='equal')
    render_2d_mpl(locdata_blobs_2d, bin_size=100, bin_range=None, rescale=(0, 50))
    render_2d_mpl(locdata_blobs_2d, bin_size=100, bin_range='zero')

    fig, ax = plt.subplots(nrows=1, ncols=2)
    render_2d_mpl(locdata_blobs_2d, ax=ax[0])
    render_2d_mpl(locdata_blobs_2d, bin_range='zero', ax=ax[1])

    render_2d_mpl(locdata_blobs_2d, ax=ax[0], colorbar_kws=dict(ax=ax[0]))
    render_2d_mpl(locdata_blobs_2d, bin_range='zero', ax=ax[1])

    plt.close('all')


@pytest.mark.visual
# this is to check overlay of rendered image and single localization points
def test_render_2d_mpl_show(locdata_blobs_2d):
    # print(locdata_blobs_2d.coordinates)
    render_2d_mpl(locdata_blobs_2d, bin_size=10, bin_range=None, rescale=None, other_property='position_y')
    # plt.plot(locdata_blobs_2d.coordinates[:, 0], locdata_blobs_2d.coordinates[:, 1], 'o')
    plt.show()

    plt.close('all')


@pytest.mark.skipif(not HAS_DEPENDENCY["mpl_scatter_density"], reason="requires mpl_scatter_density")
def test_render_2d_scatter_density(locdata_blobs_2d):
    render_2d_scatter_density(locdata_blobs_2d)
    # render_2d_scatter_density(locdata_blobs_2d, bin_range=[[500, 1000], [500, 1000]], cbar=False)

    # render_2d_scatter_density(locdata_blobs_2d, bin_range=None, vmin=0, vmax=1)

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # render_2d_scatter_density(locdata_blobs_2d, ax=ax[0])
    # render_2d_scatter_density(locdata_blobs_2d, bin_range='zero', ax=ax[1])
    #
    # render_2d_scatter_density(locdata_blobs_2d, ax=ax[0], colorbar_kws=dict(ax=ax[0]))
    # render_2d_scatter_density(locdata_blobs_2d, bin_range='zero', ax=ax[1])

    # render_2d_scatter_density(locdata_blobs_2d, other_property='position_x')

    # plt.show()

    plt.close('all')


@pytest.mark.skipif(not HAS_DEPENDENCY["mpl_scatter_density"], reason="requires mpl_scatter_density")
def test_render_2d_scatter_density_empty(locdata_empty):
    render_2d_scatter_density(locdata_empty)
    # plt.show()

    plt.close('all')


@pytest.mark.skipif(not HAS_DEPENDENCY["mpl_scatter_density"], reason="requires mpl_scatter_density")
def test_render_2d_scatter_density_single(locdata_single_localization, caplog):
    render_2d_scatter_density(locdata_single_localization)
    assert caplog.record_tuples == [('locan.render.render2d', 30, 'Locdata carries a single localization.')]
    # plt.show()
    plt.close('all')


@pytest.mark.gui
@pytest.mark.parametrize("test_input, expected", list((member, 0) for member in list(RenderEngine)))
def test_render_2d(locdata_blobs_2d, test_input, expected):
    if HAS_DEPENDENCY["napari"] and test_input == RenderEngine.NAPARI:
        render_2d(locdata_blobs_2d, render_engine=test_input)
        # napari.run()
    else:
        render_2d(locdata_blobs_2d, render_engine=test_input)
    # plt.show()

    plt.close('all')


def test_scatter_2d_mpl(locdata_2d):
    scatter_2d_mpl(locdata_2d, text_kwargs=dict(color='r'), color='r')
    # plt.show()

    plt.close('all')

def test_scatter_2d_mpl_empty(locdata_empty):
    scatter_2d_mpl(locdata_empty)
    # plt.show()

    plt.close('all')


def test_scatter_2d_mpl_single(locdata_single_localization, caplog):
    scatter_2d_mpl(locdata_single_localization)
    assert caplog.record_tuples == [('locan.render.render2d', 30, 'Locdata carries a single localization.')]
    # plt.show()

    plt.close('all')


@pytest.mark.visual  # Visual check repeating previously checked functionality
def test_scatter_2d_mpl_2(locdata_blobs_2d):
    _, collection = cluster_dbscan(locdata_blobs_2d, eps=20, min_samples=3)
    render_2d_mpl(locdata_blobs_2d)
    scatter_2d_mpl(collection)
    plt.show()

    plt.close('all')


def test_apply_window():
    img = np.ones((10, 10))
    img_filtered = apply_window(image=img, window_function='tukey', alpha=0.4)
    assert np.array_equal(img_filtered[0, :], np.zeros(10))
    assert np.array_equal(img_filtered[:, 0], np.zeros(10))


def test_render_2d_rgb_mpl_empty(locdata_empty):
    render_2d_rgb_mpl([locdata_empty, locdata_empty], bin_size=1)
    # plt.show()

    plt.close('all')


def test_render_2d_rgb_mpl_single(locdata_empty, locdata_single_localization, caplog):
    render_2d_rgb_mpl([locdata_empty, locdata_single_localization], bin_size=1)
    assert caplog.record_tuples == [('locan.render.render2d', 30, 'Locdata carries a single localization.')]
    # plt.show()

    plt.close('all')


def test_render_2d_rgb_mpl(locdata_2d):
    render_2d_rgb_mpl([locdata_2d, locdata_2d], bin_size=1)
    # plt.show()

    plt.close('all')


@pytest.mark.visual  # Visual check repeating previously checked functionality
def test_render_2d_rgb_mpl_2(locdata_blobs_2d):
    locdata_0 = locdata_blobs_2d
    locdata_1 = transform_affine(locdata_blobs_2d, offset=(20, 0))
    render_2d_rgb_mpl([locdata_0, locdata_1], bin_size=20)
    plt.show()

    plt.close('all')


@pytest.mark.visual  # Visual check repeating previously checked functionality
def test_render_2d_rgb_mpl_3(locdata_blobs_2d):
    """Check intensity normalization."""
    locdata_0 = locdata_blobs_2d
    locdata_1 = transform_affine(locdata_blobs_2d, offset=(20, 0))

    render_2d_rgb_mpl([locdata_0, locdata_1], bin_size=20)
    plt.show()
    render_2d_rgb_mpl([locdata_0, locdata_1], bin_size=20, rescale=False)
    plt.show()

    render_2d_rgb_mpl([locdata_0, locdata_1], bin_size=100)
    plt.show()
    render_2d_rgb_mpl([locdata_0, locdata_1], bin_size=100, rescale=True)
    plt.show()
    render_2d_rgb_mpl([locdata_0, locdata_1], bin_size=100, rescale=False)
    plt.show()

    plt.close('all')
