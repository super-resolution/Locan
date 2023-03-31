import matplotlib.pyplot as plt  # noqa: F401  # this import is needed for interactive tests
import numpy as np
import pytest

import locan
from locan import (  # noqa: F401  # this import is needed for interactive tests
    RenderEngine,
    render_2d_mpl,
    render_2d_napari,
    render_2d_rgb_napari,
    transform_affine,
)
from locan.dependencies import HAS_DEPENDENCY

if HAS_DEPENDENCY["napari"]:
    import napari

pytestmark = [
    pytest.mark.qt,
    pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="requires napari"),
]

HAS_NAPARI_AND_PYTESTQT = HAS_DEPENDENCY["napari"] and HAS_DEPENDENCY["pytestqt"]
# pytestqt is not a requested or extra dependency.
# If napari and pytest-qt is installed, all tests run.
# Tests in docker or GitHub actions on linux require xvfb
# for tests with pytest-qt to run.


@pytest.mark.visual
# this is to check overlay of rendered image and single localization points
def test_render_2d_napari_new_visual(locdata_blobs_2d):
    print(locdata_blobs_2d.data[locdata_blobs_2d.coordinate_properties].describe())

    render_2d_mpl(locdata_blobs_2d, bin_size=10, cmap="viridis")
    plt.show()

    viewer = napari.Viewer()

    bins = locan.Bins(
        bin_size=(50, 100),
        bin_range=(
            (0, locdata_blobs_2d.coordinates.max(axis=0)[0]),
            (0, locdata_blobs_2d.coordinates.max(axis=0)[1]),
        ),
        extend_range=True,
    )

    render_2d_napari(locdata_blobs_2d, bins=bins, cmap="viridis", viewer=viewer)

    render_2d_napari(locdata_blobs_2d, bin_size=10, cmap="viridis", viewer=viewer)

    viewer.add_points(locdata_blobs_2d.coordinates, size=10, opacity=0.2)

    print(viewer.layers[0].corner_pixels)
    print(viewer.layers[0].data_to_world(viewer.layers[0].corner_pixels))

    napari.run()
    plt.close("all")


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_render_2d_napari(make_napari_viewer, locdata_blobs_2d):
    viewer = make_napari_viewer()

    with pytest.raises(AssertionError):
        render_2d_napari(
            locdata_blobs_2d, viewer=viewer, bin_edges=((0, 10, 100), (0, 10, 100))
        )

    viewer_ = render_2d_napari(
        locdata_blobs_2d, viewer=viewer, bin_size=100, cmap="viridis", gamma=0.1
    )
    assert viewer_ is viewer
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "LocData 0"
    assert np.array_equal(viewer.layers[0].corner_pixels, [[0, 0], [8, 5]])
    assert np.array_equal(
        viewer.layers[0].data_to_world(viewer.layers[0].corner_pixels),
        [[112.0, 465.0], [912.0, 965.0]],
    )
    assert viewer.scale_bar.unit is None or len(viewer.scale_bar.unit) != 0
    assert viewer.layers[0].metadata["message"]
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
        "locan.visualize.napari.render2d",
        30,
        "Locdata carries a single localization.",
    )


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
