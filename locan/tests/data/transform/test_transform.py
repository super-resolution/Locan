import matplotlib.pyplot as plt
import numpy as np
import pytest

import locan.constants
from locan import (
    bunwarp,
    overlay,
    render_2d_mpl,
    render_2d_rgb_mpl,
    standardize,
    transform_affine,
)
from locan.data.cluster import cluster_dbscan
from locan.data.transform.bunwarpj import _read_matrix, _unwarp
from locan.data.transform.spatial_transformation import _homogeneous_matrix
from locan.dependencies import HAS_DEPENDENCY
from locan.locan_io.locdata.io_locdata import load_asdf_file


def test_bunwarp_raw_transformation():
    matrix_path = (
        locan.ROOT_DIR
        / "tests/test_data/transform/BunwarpJ_transformation_raw_green.txt"
    )
    dat_green = load_asdf_file(
        path=locan.ROOT_DIR / "tests/test_data/transform/rapidSTORM_beads_green.asdf"
    )

    matrix_x, matrix_y = _read_matrix(path=matrix_path)
    assert np.array_equal(matrix_x.shape, [140, 140])

    dat_green_flipped = transform_affine(
        dat_green, matrix=[[-1, 0], [0, 1]], offset=[1400, 0]
    )
    new_points = _unwarp(
        dat_green_flipped.coordinates, matrix_x, matrix_y, pixel_size=(10, 10)
    )
    assert len(new_points) == len(dat_green)
    assert new_points[0].tolist() == pytest.approx([719.42151268, 744.8724311])
    assert new_points[-1].tolist() == pytest.approx([710.75245448, 751.69111475])

    dat_green_transformed = bunwarp(
        locdata=dat_green, matrix_path=matrix_path, pixel_size=(10, 10), flip=True
    )
    assert len(dat_green_transformed) == len(dat_green)
    assert np.array_equal(dat_green_transformed.coordinates, new_points)
    assert dat_green_transformed.meta.history[-1].name == "bunwarp"

    # for visual inspection
    dat_red = load_asdf_file(
        path=locan.ROOT_DIR / "tests/test_data/transform/rapidSTORM_beads_red.asdf"
    )

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    render_2d_mpl(
        dat_red,
        ax=ax,
        bin_size=5,
        bin_range=((0, 1000), (0, 1400)),
        rescale=True,
        cmap="Reds",
    )
    render_2d_mpl(
        dat_green,
        ax=ax,
        bin_size=5,
        bin_range=((0, 1000), (0, 1400)),
        rescale=True,
        cmap="Greens",
    )
    render_2d_mpl(
        dat_green_transformed,
        ax=ax,
        bin_size=5,
        bin_range=((0, 1000), (0, 1400)),
        rescale=True,
        cmap="Blues",
        alpha=0.5,
    )
    # plt.show()

    render_2d_rgb_mpl(
        [dat_red, dat_green_transformed], bin_size=5, bin_range=((0, 1000), (0, 1400))
    )
    # plt.show()

    plt.close("all")


def test_homogeneous_matrix():
    matrix_out = _homogeneous_matrix()
    result = np.identity(4)
    assert np.array_equal(matrix_out, result)

    matrix = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    offset = (10, 20, 30)
    matrix_out = _homogeneous_matrix(matrix, offset)
    result = np.array(((1, 2, 3, 10), (4, 5, 6, 20), (7, 8, 9, 30), (0, 0, 0, 1)))
    assert np.array_equal(matrix_out, result)

    matrix = ((1, 2), (3, 4))
    offset = (10, 20)
    matrix_out = _homogeneous_matrix(matrix, offset)
    result = np.array(((1, 2, 10), (3, 4, 20), (0, 0, 1)))
    assert np.array_equal(matrix_out, result)


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("locdata_empty", 0),
        ("locdata_single_localization", 4),
        ("locdata_2d", 4),
        ("locdata_3d", 5),
        ("locdata_non_standard_index", 4),
    ],
)
def test_standard_locdata_objects(
    locdata_empty,
    locdata_single_localization,
    locdata_2d,
    locdata_3d,
    locdata_non_standard_index,
    fixture_name,
    expected,
):
    locdata = eval(fixture_name)
    new_locdata = transform_affine(locdata)
    assert len(new_locdata.data.columns) == expected


@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("locdata_empty", 0),
        ("locdata_single_localization", 4),
        ("locdata_2d", 4),
        ("locdata_3d", 5),
        ("locdata_non_standard_index", 4),
    ],
)
def test_standard_locdata_objects_open3d(
    locdata_empty,
    locdata_single_localization,
    locdata_2d,
    locdata_3d,
    locdata_non_standard_index,
    fixture_name,
    expected,
):
    locdata = eval(fixture_name)
    new_locdata = transform_affine(locdata, method="open3d")
    assert len(new_locdata.data.columns) == expected


def test_transformation_affine_2d(locdata_2d):
    new_locdata = transform_affine(locdata_2d)
    assert np.array_equal(new_locdata.coordinates, locdata_2d.coordinates)
    assert len(new_locdata.data.columns) == 4
    assert new_locdata.meta.history[-1].name == "transform_affine"
    assert (
        "'matrix': None, 'offset': None, 'pre_translation': None, 'method': 'numpy'"
        in new_locdata.meta.history[-1].parameter
    )

    matrix = ((-1, 0), (0, -1))
    offset = (10, 10)
    pre_translation = (100, 100)

    new_locdata = transform_affine(locdata_2d, matrix, offset)
    points_target = ((9, 9), (9, 5), (8, 7), (7, 4), (6, 8), (5, 5))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 4

    new_locdata = transform_affine(
        locdata_2d, offset=offset, pre_translation=pre_translation
    )
    points_target = ((11, 11), (11, 15), (12, 13), (13, 16), (14, 12), (15, 15))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 4


@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
def test_transformation_affine_2d_open3d(locdata_2d):
    new_locdata = transform_affine(locdata_2d, method="open3d")
    assert np.array_equal(new_locdata.coordinates, locdata_2d.coordinates)
    assert len(new_locdata.data.columns) == 4

    matrix = ((-1, 0), (0, -1))
    offset = (10, 10)
    pre_translation = (100, 100)

    new_locdata = transform_affine(locdata_2d, matrix, offset, method="open3d")
    points_target = ((9, 9), (9, 5), (8, 7), (7, 4), (6, 8), (5, 5))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 4

    new_locdata = transform_affine(
        locdata_2d, offset=offset, pre_translation=pre_translation, method="open3d"
    )
    points_target = ((11, 11), (11, 15), (12, 13), (13, 16), (14, 12), (15, 15))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 4


def test_transformation_affine_3d(locdata_3d):
    new_locdata = transform_affine(locdata_3d)
    assert np.array_equal(new_locdata.coordinates, locdata_3d.coordinates)
    assert len(new_locdata.data.columns) == 5
    assert new_locdata.meta.history[-1].name == "transform_affine"
    assert (
        "'matrix': None, 'offset': None, 'pre_translation': None, 'method': 'numpy'"
        in new_locdata.meta.history[-1].parameter
    )

    matrix = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
    offset = (10, 10, 10)
    pre_translation = (100, 100, 100)

    new_locdata = transform_affine(locdata_3d, matrix, offset)
    points_target = ((9, 9, 9), (9, 5, 8), (8, 7, 5), (7, 4, 6), (6, 8, 7), (5, 5, 8))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 5

    new_locdata = transform_affine(
        locdata_3d, offset=offset, pre_translation=pre_translation
    )
    points_target = (
        (11, 11, 11),
        (11, 15, 12),
        (12, 13, 15),
        (13, 16, 14),
        (14, 12, 13),
        (15, 15, 12),
    )
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 5


@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
def test_transformation_affine_3d_open3d(locdata_3d):
    new_locdata = transform_affine(locdata_3d, method="open3d")
    assert np.array_equal(new_locdata.coordinates, locdata_3d.coordinates)
    assert len(new_locdata.data.columns) == 5

    matrix = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
    offset = (10, 10, 10)
    pre_translation = (100, 100, 100)

    new_locdata = transform_affine(locdata_3d, matrix, offset, method="open3d")
    points_target = ((9, 9, 9), (9, 5, 8), (8, 7, 5), (7, 4, 6), (6, 8, 7), (5, 5, 8))
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 5

    new_locdata = transform_affine(
        locdata_3d, offset=offset, pre_translation=pre_translation, method="open3d"
    )
    points_target = (
        (11, 11, 11),
        (11, 15, 12),
        (12, 13, 15),
        (13, 16, 14),
        (14, 12, 13),
        (15, 15, 12),
    )
    assert np.array_equal(new_locdata.coordinates, points_target)
    assert len(new_locdata.data.columns) == 5


def test_standardize(locdata_2d):
    locdata_standardized = standardize(locdata_2d)
    assert locdata_standardized.coordinates.mean() == pytest.approx(0)
    assert locdata_standardized.coordinates.var(ddof=0) == pytest.approx(1)

    locdata_standardized = standardize(locdata_2d, with_std=False)
    assert locdata_standardized.coordinates.mean() == pytest.approx(0)
    assert locdata_standardized.coordinates.var() != pytest.approx(
        locdata_2d.coordinates.var(ddof=0)
    )

    locdata_standardized = standardize(
        locdata_2d, loc_properties=["intensity"], with_mean=True
    )
    assert locdata_standardized.data.intensity.mean() == pytest.approx(0)
    assert locdata_standardized.data.intensity.var(ddof=0) == pytest.approx(1)


def test_overlay(locdata_two_cluster_2d):
    _, clust = cluster_dbscan(locdata_two_cluster_2d, eps=2, min_samples=1)
    new_locdata = overlay(clust.references)
    assert new_locdata.meta.history[0].name == "overlay"
    assert len(new_locdata) == 2
    assert 0 == pytest.approx(new_locdata.coordinates, abs=0.1)

    # various translations
    new_locdata = overlay(clust.references, centers=None)
    assert len(new_locdata) == 2
    assert np.array_equal(new_locdata.coordinates, clust.coordinates)

    centers = [ref.centroid for ref in clust.references]
    new_locdata = overlay(clust.references, centers=centers)
    assert len(new_locdata) == 2
    assert 0 == pytest.approx(new_locdata.coordinates, abs=0.1)

    new_locdata = overlay(clust.references, centers="bb")
    assert len(new_locdata) == 2
    assert 0 == pytest.approx(new_locdata.coordinates, abs=0.1)

    new_locdata = overlay(clust.references, centers="obb")
    assert len(new_locdata) == 2
    assert 0 == pytest.approx(new_locdata.coordinates, abs=0.3)

    new_locdata = overlay(clust.references, centers="ch")
    assert len(new_locdata) == 2
    assert 0 == pytest.approx(new_locdata.coordinates, abs=0.1)

    for reference in clust.references:
        reference.region = reference.bounding_box.region
    new_locdata = overlay(clust.references, centers="region")
    assert len(new_locdata) == 2
    assert 0 == pytest.approx(new_locdata.coordinates, abs=0.1)

    with pytest.raises(ValueError):
        overlay(clust.references, centers="undefined")
    with pytest.raises(ValueError):
        overlay(clust.references, centers=[(1, 1)])

    # various rotations
    orientations = [ref.oriented_bounding_box.angle for ref in clust.references]
    new_locdata = overlay(clust.references, orientations=orientations)
    assert len(new_locdata) == 2
    assert 0 == pytest.approx(new_locdata.coordinates)

    new_locdata = overlay(clust.references, orientations="orientation_obb")
    assert len(new_locdata) == 2
    assert 0 == pytest.approx(new_locdata.coordinates)

    new_locdata = overlay(clust.references, orientations="orientation_im")
    assert len(new_locdata) == 2
    assert 0 == pytest.approx(new_locdata.coordinates)

    with pytest.raises(ValueError):
        overlay(clust.references, orientations="undefined")
    with pytest.raises(ValueError):
        overlay(clust.references, orientations=[(180,)])

    with pytest.raises(TypeError):
        overlay([])
    with pytest.raises(TypeError):
        overlay(clust)
