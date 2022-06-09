import pytest
import numpy as np
import matplotlib.pyplot as plt  # this import is needed for interactive tests

from locan import RenderEngine  # this import is needed for interactive tests
from locan.dependencies import HAS_DEPENDENCY
from locan import (
    render_2d_mpl,
    render_2d_napari,
    select_by_drawing_napari,
    render_2d_rgb_napari,
)
from locan import render_2d
from locan import transform_affine

if HAS_DEPENDENCY["napari"]:
    import napari


pytestmark = pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="requires napari")

HAS_NAPARI_AND_PYTESTQT = HAS_DEPENDENCY["napari"] and HAS_DEPENDENCY["pytestqt"]
# pytestqt is not a requested or extra dependency.
# If napari and pytest-qt is installed, all tests run.
# Tests in docker or GitHub actions on linux require xvfb for tests with pytest-qt to run.


@pytest.mark.gui
@pytest.mark.parametrize(
    "test_input, expected", list((member, 0) for member in list(RenderEngine))
)
def test_render_2d_gui(locdata_blobs_2d, test_input, expected):
    if test_input == RenderEngine.NAPARI:
        render_2d(locdata_blobs_2d, render_engine=test_input)
        # napari.run()
    else:
        render_2d(locdata_blobs_2d, render_engine=test_input)
    # plt.show()

    plt.close("all")


@pytest.mark.visual
# this is to check overlay of rendered image and single localization points
def test_render_2d_napari_coordinates(locdata_blobs_2d):
    render_2d_mpl(locdata_blobs_2d, bin_size=10, cmap="viridis")
    plt.show()

    viewer, bins = render_2d_napari(
        locdata_blobs_2d, bin_size=10, cmap="viridis", gamma=0.1
    )
    viewer.add_points(
        (locdata_blobs_2d.coordinates - np.array(bins.bin_range)[:, 0]) / 10
    )
    napari.run()

    plt.close("all")


@pytest.mark.gui
def test_render_2d_napari_gui(locdata_blobs_2d):
    render_2d_mpl(locdata_blobs_2d, bin_size=100, cmap="viridis")
    plt.show()

    render_2d_napari(locdata_blobs_2d, bin_size=100, cmap="viridis", gamma=0.1)
    napari.run()

    viewer, _ = render_2d_napari(
        locdata_blobs_2d, bin_size=50, cmap="magenta", gamma=0.1
    )
    render_2d_napari(
        locdata_blobs_2d,
        viewer=viewer,
        bin_size=100,
        cmap="cyan",
        gamma=0.1,
        scale=(2, 2),
        blending="additive",
    )
    # napari.run()

    render_2d(locdata_blobs_2d, render_engine=RenderEngine.NAPARI)
    # napari.run()

    plt.close("all")


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_render_2d_napari(make_napari_viewer, locdata_blobs_2d):
    viewer = make_napari_viewer()
    render_2d_napari(
        locdata_blobs_2d, viewer=viewer, bin_size=100, cmap="viridis", gamma=0.1
    )
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "LocData 0"
    viewer.close()


@pytest.mark.gui
def test_render_2d_napari_empty_gui(locdata_empty):
    render_2d_napari(locdata_empty, bin_size=100, cmap="viridis", gamma=0.1)
    napari.run()


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_render_2d_napari_empty(make_napari_viewer, locdata_empty):
    viewer = make_napari_viewer()
    render_2d_napari(
        locdata_empty, viewer=viewer, bin_size=100, cmap="viridis", gamma=0.1
    )
    assert viewer.layers == []
    viewer.close()


@pytest.mark.gui
def test_render_2d_napari_single_gui(locdata_single_localization, caplog):
    render_2d_napari(
        locdata_single_localization, bin_size=100, cmap="viridis", gamma=0.1
    )
    assert caplog.record_tuples[1] == (
        "locan.render.render2d",
        30,
        "Locdata carries a single localization.",
    )
    napari.run()


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_render_2d_napari_single(
    make_napari_viewer, locdata_single_localization, caplog
):
    viewer = make_napari_viewer()
    render_2d_napari(
        locdata_single_localization,
        viewer=viewer,
        bin_size=100,
        cmap="viridis",
        gamma=0.1,
    )
    viewer.close()
    assert caplog.record_tuples[0] == (
        "locan.render.render2d",
        30,
        "Locdata carries a single localization.",
    )


@pytest.mark.gui
def test_select_by_drawing_napari_gui(locdata_blobs_2d):
    viewer = napari.Viewer()
    viewer.add_shapes(data=np.array([(1, 10), (10, 20)]), shape_type="rectangle")

    rois = select_by_drawing_napari(
        locdata_blobs_2d, viewer=viewer, bin_size=10, cmap="viridis", gamma=0.1
    )
    # No need for napari.run() since it is called inside select_by_drawing_napari.
    print(rois)
    print(viewer.layers["Shapes"].data)
    assert len(rois) == 1
    assert repr(rois[0].region) == "Rectangle((122.0, 565.0), 90.0, 100.0, 0)"


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_select_by_drawing_napari(make_napari_viewer, locdata_blobs_2d):
    viewer = make_napari_viewer()
    viewer.add_shapes(data=np.array([(1, 10), (10, 20)]), shape_type="rectangle")

    rois = select_by_drawing_napari(
        locdata_blobs_2d,
        viewer=viewer,
        napari_run=False,
        bin_size=10,
        cmap="viridis",
        gamma=0.1,
    )
    assert len(rois) == 1
    assert repr(rois[0].region) == "Rectangle((122.0, 565.0), 90.0, 100.0, 0)"
    viewer.close()


@pytest.mark.gui
def test_select_by_drawing_napari_2(locdata_blobs_2d):
    roi_list = select_by_drawing_napari(locdata_blobs_2d)
    print(roi_list)


@pytest.mark.gui
def test_render_2d_rgb_napari_gui(locdata_blobs_2d):
    locdata_0 = locdata_blobs_2d
    locdata_1 = transform_affine(locdata_blobs_2d, offset=(20, 0))
    render_2d_rgb_napari([locdata_0, locdata_1], bin_size=20)

    napari.run()


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_render_2d_rgb_napari(make_napari_viewer, locdata_blobs_2d):
    viewer = make_napari_viewer()
    locdata_0 = locdata_blobs_2d
    locdata_1 = transform_affine(locdata_blobs_2d, offset=(20, 0))
    render_2d_rgb_napari([locdata_0, locdata_1], viewer=viewer, bin_size=20)
    viewer.close()
