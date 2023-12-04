import pytest

from locan import (
    render_3d_napari,
    render_3d_napari_image,
    render_3d_rgb_napari,
    transform_affine,
)
from locan.dependencies import HAS_DEPENDENCY

napari = pytest.importorskip("napari")

HAS_NAPARI_AND_PYTESTQT = HAS_DEPENDENCY["napari"] and HAS_DEPENDENCY["pytestqt"]
# pytestqt is not a requested or extra dependency.
# If napari and pytest-qt is installed, all tests run.
# Tests in docker or GitHub actions on linux require xvfb
# for tests with pytest-qt to run.


@pytest.mark.visual
# this is to check overlay of rendered image and single localization points
def test_render_3d_napari_coordinates(locdata_blobs_3d):
    viewer = render_3d_napari(locdata_blobs_3d, bin_size=10, cmap="viridis", gamma=0.1)
    viewer.add_points(locdata_blobs_3d.coordinates)
    napari.run()


def test_render_3d_napari_image(locdata_blobs_3d):
    with pytest.raises(ValueError):
        render_3d_napari_image(
            locdata_blobs_3d, bin_edges=((0, 10, 100), (0, 10, 100), (0, 10, 100))
        )

    data, image_kwargs, layer_type = render_3d_napari_image(
        locdata_blobs_3d, bin_size=100, cmap="viridis", gamma=0.1
    )
    assert layer_type == "image"
    assert all(
        key in ["name", "colormap", "scale", "translate", "metadata", "gamma"]
        for key in image_kwargs
    )
    assert data.shape == (7, 6, 8)


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_render_3d_napari(make_napari_viewer, locdata_blobs_3d):
    viewer = make_napari_viewer()
    viewer_ = render_3d_napari(
        locdata_blobs_3d,
        viewer=viewer,
        bin_size=100,
        cmap="cyan",
        gamma=0.1,
        scale=(2, 2),
        blending="additive",
    )
    assert viewer_ is viewer
    viewer.close()


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_render_3d_napari_empty(make_napari_viewer, locdata_empty):
    viewer = make_napari_viewer()
    render_3d_napari(
        locdata_empty, viewer=viewer, bin_size=100, cmap="viridis", gamma=0.1
    )
    assert viewer.layers == []
    viewer.close()


@pytest.mark.gui
def test_render_3d_napari_single_gui(locdata_single_localization_3d, caplog):
    render_3d_napari(
        locdata_single_localization_3d, bin_size=100, cmap="viridis", gamma=0.1
    )
    try:
        record_tuples_ = caplog.record_tuples[1]
    except IndexError:
        record_tuples_ = caplog.record_tuples[0]
    assert record_tuples_ == (
        "locan.visualize.napari.render3d",
        30,
        "Locdata carries a single localization.",
    )
    napari.run()


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_render_3d_napari_single(
    make_napari_viewer, locdata_single_localization_3d, caplog
):
    viewer = make_napari_viewer()
    render_3d_napari(
        locdata_single_localization_3d,
        viewer=viewer,
        bin_size=100,
        cmap="viridis",
        gamma=0.1,
    )
    viewer.close()
    assert caplog.record_tuples[0] == (
        "locan.visualize.napari.render3d",
        30,
        "Locdata carries a single localization.",
    )


@pytest.mark.gui
def test_render_3d_rgb_napari_gui(locdata_blobs_3d):
    locdata_0 = locdata_blobs_3d
    locdata_1 = transform_affine(locdata_blobs_3d, offset=(20, 0, 0))
    render_3d_rgb_napari([locdata_0, locdata_1], bin_size=20)

    napari.run()


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_render_3d_rgb_napari(make_napari_viewer, locdata_blobs_3d):
    viewer = make_napari_viewer()
    locdata_0 = locdata_blobs_3d
    locdata_1 = transform_affine(locdata_blobs_3d, offset=(20, 0, 0))
    render_3d_rgb_napari([locdata_0, locdata_1], viewer=viewer, bin_size=20)
    viewer.close()
