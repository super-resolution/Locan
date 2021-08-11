import pytest
import numpy as np
import matplotlib.pyplot as plt  # this import is needed for interactive tests

from locan.constants import RenderEngine  # this import is needed for interactive tests
from locan.constants import _has_mpl_scatter_density, _has_napari
if _has_napari: import napari
from locan import render_2d_mpl, render_2d_scatter_density, render_2d_napari, scatter_2d_mpl, select_by_drawing_napari
from locan.render.render2d import _napari_shape_to_region
from locan import render_2d, apply_window
from locan import cluster_dbscan


# flag to skip some of the following tests
skip_tests = True


def test_render_2d_mpl_empty(locdata_empty):
    render_2d_mpl(locdata_empty, bin_size=5)
    # plt.show()

    plt.close('all')


def test_render_2d_mpl_single(locdata_single_localization):
    render_2d_mpl(locdata_single_localization, bin_size=5)
    render_2d_mpl(locdata_single_localization, bin_size=0.5)
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


@pytest.mark.skip('Test for visual inspection.')
# this is to check overlay of rendered image and single localization points
def test_render_2d_mpl_show(locdata_blobs_2d):
    print(locdata_blobs_2d.coordinates)
    render_2d_mpl(locdata_blobs_2d, bin_size=10, bin_range=None, rescale=None)
    plt.plot(locdata_blobs_2d.coordinates[:, 0], locdata_blobs_2d.coordinates[:, 1], 'o')
    plt.show()

    plt.close('all')


@pytest.mark.skipif(not _has_mpl_scatter_density, reason="requires mpl_scatter_density")
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


@pytest.mark.skipif(skip_tests, reason='GUI tests are skipped because they would need user interaction.')
@pytest.mark.parametrize("test_input, expected", list((member, 0) for member in list(RenderEngine)))
def test_render_2d(locdata_blobs_2d, test_input, expected):
    if _has_napari and test_input == RenderEngine.NAPARI:
        render_2d(locdata_blobs_2d, render_engine=test_input)
        # napari.run()
    else:
        render_2d(locdata_blobs_2d, render_engine=test_input)
    # plt.show()

    plt.close('all')


@pytest.mark.skip('Test for visual inspection.')
# this is to check overlay of rendered image and single localization points
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_render_2d_napari_coordinates(locdata_blobs_2d):
    render_2d_mpl(locdata_blobs_2d, bin_size=10, cmap='viridis')
    plt.show()

    viewer, histogram = render_2d_napari(locdata_blobs_2d, bin_size=10, cmap='viridis', gamma=0.1)
    viewer.add_points((locdata_blobs_2d.coordinates - np.array(histogram.bins.bin_range)[:, 0]) / 10 )
    napari.run()

    vertices = viewer.layers['Shapes'].data
    print(vertices)

    plt.close('all')


@pytest.mark.skipif(skip_tests, reason='GUI tests are skipped because they would need user interaction.')
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_render_2d_napari(locdata_blobs_2d):
    render_2d_mpl(locdata_blobs_2d, bin_size=100, cmap='viridis')
    plt.show()

    render_2d_napari(locdata_blobs_2d, bin_size=100, cmap='viridis', gamma=0.1)
    napari.run()

    viewer, _ = render_2d_napari(locdata_blobs_2d, bin_size=50, cmap='magenta', gamma=0.1)
    render_2d_napari(locdata_blobs_2d, viewer=viewer, bin_size=100, cmap='cyan', gamma=0.1, scale=(2, 2),
                     blending='additive')
    # napari.run()

    render_2d(locdata_blobs_2d, render_engine=RenderEngine.NAPARI)
    # napari.run()

    plt.close('all')


def test__napari_shape_to_region():
    # rectangle
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    bin_edges = np.array([[0, 10, 20], [2, 3, 4, 5]], dtype=object)
    region = _napari_shape_to_region(vertices, bin_edges, 'rectangle')
    assert repr(region) == 'Rectangle((0.0, 2.0), 31.0, 2.5, 0)'

    # ellipse
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    bin_edges = np.array([[0, 10, 20], [2, 3, 4, 5]], dtype=object)
    region = _napari_shape_to_region(vertices, bin_edges, 'ellipse')
    assert repr(region) == 'Ellipse((15.5, 3.25), 31.0, 2.5, 0)'

    # polygon
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    bin_edges = np.array([[0, 10, 20], [2, 3, 4, 5]], dtype=object)
    region = _napari_shape_to_region(vertices, bin_edges, 'polygon')
    assert repr(region) == 'Polygon([[0.0, 2.0], [0.0, 4.5], [31.0, 4.5], [31.0, 2.0], [0.0, 2.0]])'


@pytest.mark.skipif(skip_tests, reason='GUI tests are skipped because they would need user interaction.')
@pytest.mark.skipif(not _has_napari, reason="Test requires napari.")
def test_select_by_drawing_napari(locdata_blobs_2d):
    viewer = napari.Viewer()
    viewer.add_shapes(data=((1, 10), (10, 20)), shape_type='rectangle')

    rois = select_by_drawing_napari(locdata_blobs_2d, viewer=viewer, bin_size=10, cmap='viridis', gamma=0.1)
    # No need for napari.run() since it is called inside select_by_drawing_napari.
    print(rois)
    assert len(rois) == 1
    assert repr(rois[0].region) == 'Rectangle((122.0, 565.0), 90.0, 100.0, 0)'


@pytest.mark.skipif(skip_tests, reason='GUI tests are skipped because they would need user interaction.')
def test_select_by_drawing_napari_2(locdata_blobs_2d):
    roi_list = select_by_drawing_napari(locdata_blobs_2d)
    print(roi_list)


def test_scatter_2d_mpl(locdata_2d):
    scatter_2d_mpl(locdata_2d, text_kwargs=dict(color='r'), color='r')
    # plt.show()

    plt.close('all')


@pytest.mark.skip('Visual check repeating previously checked functionality.')
def test_scatter_2d_mpl_2(locdata_blobs_2d):
    _, collection = cluster_dbscan(locdata_blobs_2d, eps=20, min_samples=3, noise=True)
    render_2d_mpl(locdata_blobs_2d)
    scatter_2d_mpl(collection)
    plt.show()

    plt.close('all')


def test_apply_window():
    img = np.ones((10, 10))
    img_filtered = apply_window(image=img, window_function='tukey', alpha=0.4)
    assert np.array_equal(img_filtered[0, :], np.zeros(10))
    assert np.array_equal(img_filtered[:, 0], np.zeros(10))
