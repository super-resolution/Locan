import numpy as np
import pytest

from locan import render_3d_napari, render_3d_rgb_napari, transform_affine
from locan.dependencies import HAS_DEPENDENCY

if HAS_DEPENDENCY["napari"]:
    import napari


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
    napari.run()


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


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
def test_render_3d_rgb_napari(locdata_blobs_3d):
    locdata_0 = locdata_blobs_3d
    locdata_1 = transform_affine(locdata_blobs_3d, offset=(20, 0, 0))
    render_3d_rgb_napari([locdata_0, locdata_1], bin_size=20)

    napari.run()
