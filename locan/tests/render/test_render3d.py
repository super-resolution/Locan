import matplotlib.pyplot as plt  # this import is needed for interactive tests
import numpy as np
import pytest

from locan import RenderEngine  # this import is needed for interactive tests
from locan import (
    render_3d,
    render_3d_napari,
    render_3d_rgb_napari,
    scatter_3d_mpl,
    transform_affine,
)
from locan.dependencies import HAS_DEPENDENCY

if HAS_DEPENDENCY["napari"]:
    import napari  # noqa: F401


@pytest.mark.visual
# this is to check overlay of rendered image and single localization points
@pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
def test_render_3d_napari_coordinates(locdata_blobs_3d):
    viewer, histogram = render_3d_napari(
        locdata_blobs_3d, bin_size=10, cmap="viridis", gamma=0.1
    )
    viewer.add_points(
        (locdata_blobs_3d.coordinates - np.array(histogram.bins.bin_range)[:, 0]) / 10
    )
    napari.run()


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
def test_render_3d_napari(locdata_blobs_3d):
    render_3d_napari(locdata_blobs_3d, bin_size=100, cmap="viridis", gamma=0.1)
    # napari.run()

    viewer = render_3d_napari(locdata_blobs_3d, bin_size=50, cmap="magenta", gamma=0.1)
    render_3d_napari(
        locdata_blobs_3d,
        viewer=viewer,
        bin_size=100,
        cmap="cyan",
        gamma=0.1,
        scale=(2, 2),
        blending="additive",
    )
    # napari.run()

    render_3d(locdata_blobs_3d, render_engine=RenderEngine.NAPARI)
    # napari.run()


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
def test_render_3d_napari_empty(locdata_empty):
    render_3d_napari(locdata_empty, bin_size=100, cmap="viridis", gamma=0.1)
    napari.run()


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
def test_render_3d_napari_single(locdata_single_localization, caplog):
    render_3d_napari(
        locdata_single_localization, bin_size=100, cmap="viridis", gamma=0.1
    )
    assert caplog.record_tuples[1] == (
        "locan.render.render3d",
        30,
        "Locdata carries a single localization.",
    )
    napari.run()


def test_scatter_3d_mpl(locdata_3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter_3d_mpl(locdata_3d, ax=ax, text_kwargs=dict(color="r"), color="r")
    # plt.show()

    plt.close("all")


def test_scatter_3d_mpl_empty(locdata_empty):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter_3d_mpl(locdata_empty, ax=ax)
    # plt.show()

    plt.close("all")


def test_scatter_3d_mpl_single(locdata_single_localization, caplog):
    fig = plt.figure()
    fig.add_subplot(projection="3d")
    scatter_3d_mpl(locdata_single_localization)
    assert caplog.record_tuples == [
        ("locan.render.render3d", 30, "Locdata carries a single localization.")
    ]
    # plt.show()

    plt.close("all")


@pytest.mark.gui
@pytest.mark.parametrize(
    "test_input, expected", list((member, 0) for member in list(RenderEngine))
)
def test_render_3d(locdata_blobs_3d, test_input, expected):
    if HAS_DEPENDENCY["napari"] and test_input == RenderEngine.NAPARI:
        render_3d(locdata_blobs_3d, render_engine=test_input)
        # napari.run()
    else:
        with pytest.raises(NotImplementedError):
            render_3d(locdata_blobs_3d, render_engine=test_input)
    # plt.show()

    plt.close("all")


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
def test_render_3d_rgb_napari(locdata_blobs_3d):
    locdata_0 = locdata_blobs_3d
    locdata_1 = transform_affine(locdata_blobs_3d, offset=(20, 0, 0))
    render_3d_rgb_napari([locdata_0, locdata_1], bin_size=20)

    napari.run()
