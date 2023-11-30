import numpy as np
import pytest

from locan import Roi, get_rois, save_rois, select_by_drawing_napari
from locan.dependencies import HAS_DEPENDENCY
from locan.visualize.render_napari.utilities import _shape_to_region, _shapes_to_regions

napari = pytest.importorskip("napari")

HAS_NAPARI_AND_PYTESTQT = HAS_DEPENDENCY["napari"] and HAS_DEPENDENCY["pytestqt"]
# pytestqt is not a requested or extra dependency.
# If napari and pytest-qt is installed, all tests run.
# Tests in docker or GitHub actions on linux require xvfb
# for tests with pytest-qt to run.


def clean_directory(directory):
    for item in directory.glob("*"):
        if item.is_dir():
            item.rmdir()
        else:
            item.unlink()


def test__shape_to_region():
    # rectangle
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    region = _shape_to_region(vertices, "rectangle")
    assert repr(region) == "Rectangle((0.0, 0.0), 3.1, 2.5, 0)"

    # ellipse
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    region = _shape_to_region(vertices, "ellipse")
    assert repr(region) == "Ellipse((1.55, 1.25), 3.1, 2.5, 0)"

    # polygon
    vertices = np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]])
    region = _shape_to_region(vertices, "polygon")
    assert (
        repr(region)
        == "Polygon([[0.0, 0.0], [0.0, 2.5], [3.1, 2.5], [3.1, 0.0], [0.0, 0.0]])"
    )


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test__shapes_to_regions(make_napari_viewer, locdata_blobs_2d):
    viewer = make_napari_viewer()
    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "rectangle")
    layer_shapes = viewer.add_shapes(shape_data)

    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "ellipse")
    layer_shapes.add(shape_data)

    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "polygon")
    layer_shapes.add(shape_data)

    shapes_data = layer_shapes.as_layer_data_tuple()
    regions = _shapes_to_regions(shapes_data)

    expected_regions = [
        "Rectangle((0.0, 0.0), 3.1, 2.5, 0)",
        "Ellipse((1.55, 1.25), 3.1, 2.5, 0)",
        "Polygon([[0.0, 0.0], [0.0, 2.5], [3.1, 2.5], [3.1, 0.0], [0.0, 0.0]])",
    ]

    for region, expected in zip(regions, expected_regions):
        assert repr(region) == expected


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_get_rois(make_napari_viewer, locdata_blobs_2d):
    viewer = make_napari_viewer()
    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "rectangle")
    layer_shapes = viewer.add_shapes(shape_data)

    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "ellipse")
    layer_shapes.add(shape_data)

    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "polygon")
    layer_shapes.add(shape_data)

    rois = get_rois(
        shapes_layer=layer_shapes,
        reference=locdata_blobs_2d,
        loc_properties=locdata_blobs_2d.coordinate_keys,
    )
    for roi in rois:
        assert isinstance(roi, Roi)
        assert roi.loc_properties == ["position_x", "position_y"]


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_save_rois(make_napari_viewer, locdata_blobs_2d, tmp_path):
    viewer = make_napari_viewer()
    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "rectangle")
    layer_shapes = viewer.add_shapes(shape_data)

    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "ellipse")
    layer_shapes.add(shape_data)

    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "polygon")
    layer_shapes.add(shape_data)

    rois = get_rois(
        shapes_layer=layer_shapes,
        reference=locdata_blobs_2d,
        loc_properties=locdata_blobs_2d.coordinate_keys,
    )

    # files_path is existing directory
    roi_path_list = save_rois(rois=rois, file_path=tmp_path)
    saved_path_list = list(tmp_path.glob("*.yaml"))
    for roi_path, saved_path in zip(roi_path_list, saved_path_list):
        assert roi_path == saved_path
        assert saved_path.suffix == ".yaml"
        assert saved_path.stem[:-2] == "my_roi"
    clean_directory(tmp_path)

    # file_path is in non-existing directory
    with pytest.raises(FileNotFoundError):
        save_rois(rois=rois, file_path=tmp_path / "name" / "name.txt")
    clean_directory(tmp_path)

    # file_path is some name in existing directory
    roi_path_list = save_rois(rois=rois, file_path=tmp_path / "some.txt")
    saved_path_list = list(tmp_path.glob("*.yaml"))
    for roi_path, saved_path in zip(roi_path_list, saved_path_list):
        assert roi_path == saved_path
        assert saved_path.suffix == ".yaml"
        assert saved_path.stem[:-2] == "some_roi"
    clean_directory(tmp_path)

    # file_path is roi_reference
    with pytest.raises(AttributeError):
        save_rois(rois=rois, file_path="roi_reference")
    clean_directory(tmp_path)

    # file_path is roi_reference
    modified_rois = []
    for roi in rois:
        roi_ = Roi(
            region=roi.region,
            reference=dict(file_path=tmp_path / "reference.txt", file_type=0),
            loc_properties=roi.loc_properties,
        )
        modified_rois.append(roi_)

    roi_path_list = save_rois(rois=modified_rois, file_path="roi_reference")
    saved_path_list = list(tmp_path.glob("*.yaml"))
    for roi_path, saved_path in zip(roi_path_list[:1], saved_path_list):
        assert roi_path == saved_path
        assert saved_path.suffix == ".yaml"
        assert saved_path.stem[:-2] == "reference_roi"
    clean_directory(tmp_path)


@pytest.mark.gui
@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_save_rois_with_dialog(make_napari_viewer, locdata_blobs_2d):
    viewer = make_napari_viewer()
    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "rectangle")
    layer_shapes = viewer.add_shapes(shape_data)

    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "ellipse")
    layer_shapes.add(shape_data)

    shape_data = (np.array([[0, 0], [0, 2.5], [3.1, 2.5], [3.1, 0]]), "polygon")
    layer_shapes.add(shape_data)

    rois = get_rois(
        shapes_layer=layer_shapes,
        reference=locdata_blobs_2d,
        loc_properties=locdata_blobs_2d.coordinate_keys,
    )

    # files_path is None
    roi_path_list = save_rois(rois=rois, file_path=None)
    print(roi_path_list)


@pytest.mark.gui
def test_select_by_drawing_napari_simple_check(locdata_blobs_2d):
    rois = select_by_drawing_napari(locdata_blobs_2d, bin_size=50)
    print(rois)
    assert len(rois) != 0


@pytest.mark.gui
def test_select_by_drawing_napari_gui(locdata_blobs_2d):
    viewer = napari.Viewer()
    viewer.add_shapes(
        name="Rois", data=np.array([(100, 500), (600, 700)]), shape_type="rectangle"
    )

    rois = select_by_drawing_napari(
        locdata_blobs_2d, viewer=viewer, bin_size=10, cmap="viridis", gamma=0.1
    )
    # No need for napari.run() since it is called inside select_by_drawing_napari.
    print(rois)
    print(viewer.layers["Rois"].data)
    assert len(rois) == 1
    assert repr(rois[0].region) == "Rectangle((100.0, 500.0), 500.0, 200.0, 0)"


@pytest.mark.skipif(
    not HAS_NAPARI_AND_PYTESTQT, reason="Test requires napari and pytest-qt."
)
def test_select_by_drawing_napari(make_napari_viewer, locdata_blobs_2d):
    viewer = make_napari_viewer()
    viewer.add_shapes(
        name="Rois", data=np.array([(100, 500), (600, 700)]), shape_type="rectangle"
    )

    rois = select_by_drawing_napari(
        locdata_blobs_2d,
        viewer=viewer,
        napari_run=False,
        bin_size=10,
        cmap="viridis",
        gamma=0.1,
    )
    assert len(rois) == 1
    assert repr(rois[0].region) == "Rectangle((100.0, 500.0), 500.0, 200.0, 0)"
    viewer.close()
