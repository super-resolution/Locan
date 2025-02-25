from importlib.metadata import version

import numpy as np
import pytest

from locan.dependencies import HAS_DEPENDENCY
from locan.process.register import (
    _register_cc_picasso,
    _register_cc_skimage,
    register_cc,
    register_icp,
)
from locan.process.transform import transform_affine
from locan.tests.conftest import get_open3d_version


@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
@pytest.mark.skipif(
    HAS_DEPENDENCY["open3d"]
    and get_open3d_version().startswith("0.18")
    and version("numpy").startswith("2"),
    reason="Test requires open3d>0.18 or numpy<2.",
)
def test_register_icp_2d(locdata_blobs_2d):
    locdata_2d_transformed = transform_affine(locdata_blobs_2d, offset=(100, 100))
    offset_target = np.array([100.0, 100.0])
    matrix_target = np.array([[1, 0], [0, 1]])

    matrix, offset = register_icp(
        locdata_blobs_2d, locdata_2d_transformed, verbose=False
    )
    assert np.allclose(np.array(offset), offset_target)
    assert np.allclose(matrix, matrix_target)

    matrix, offset = register_icp(
        locdata_blobs_2d,
        locdata_2d_transformed,
        pre_translation=(-90, -90),
        verbose=False,
    )
    offset_target = np.array([10.0, 10.0])
    assert np.allclose(np.array(offset), offset_target)
    assert np.allclose(matrix, matrix_target)


@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
@pytest.mark.skipif(
    HAS_DEPENDENCY["open3d"]
    and get_open3d_version().startswith("0.18")
    and version("numpy").startswith("2"),
    reason="Test requires open3d>0.18 or numpy<2.",
)
def test_register_icp_3d(locdata_blobs_3d):
    locdata_3d_transformed = transform_affine(locdata_blobs_3d, offset=(100, 100, 100))
    offset_target = np.array([100.0, 100.0, 100.0])
    matrix_target = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    matrix, offset = register_icp(
        locdata_blobs_3d, locdata_3d_transformed, verbose=False
    )
    assert np.allclose(np.array(offset), offset_target)
    assert np.allclose(matrix, matrix_target)

    matrix, offset = register_icp(
        locdata_blobs_3d,
        locdata_3d_transformed,
        pre_translation=(-90, -90, -90),
        verbose=False,
    )
    offset_target = np.array([10.0, 10.0, 10.0])
    assert np.allclose(np.array(offset), offset_target)
    assert np.allclose(matrix, matrix_target)


def test__register_cc_picasso(locdata_blobs_2d):
    locdata_2d_transformed = transform_affine(locdata_blobs_2d, offset=(100, 50))

    offset_target = np.array([100.0, 50.0])
    matrix_target = np.array([[1, 0], [0, 1]])

    matrix, offset = _register_cc_picasso(
        locdata_blobs_2d, locdata_2d_transformed, bin_size=50, verbose=False
    )
    assert np.allclose(np.array(offset), offset_target, atol=5)
    assert np.allclose(matrix, matrix_target)

    matrix, offset = _register_cc_picasso(
        locdata_blobs_2d, locdata_2d_transformed, bin_size=(10, 50), verbose=False
    )
    assert np.allclose(np.array(offset), offset_target, atol=5)
    assert np.allclose(matrix, matrix_target)


def test__register_cc_skimage(locdata_blobs_2d):
    locdata_2d_transformed = transform_affine(locdata_blobs_2d, offset=(100, 50))

    offset_target = np.array([100.0, 50.0])
    matrix_target = np.array([[1, 0], [0, 1]])

    matrix, offset = _register_cc_skimage(
        locdata_blobs_2d, locdata_2d_transformed, bin_size=50
    )
    assert np.allclose(np.array(offset), offset_target, atol=5)
    assert np.allclose(matrix, matrix_target)

    matrix, offset = _register_cc_skimage(
        locdata_blobs_2d, locdata_2d_transformed, bin_size=(10, 50)
    )
    assert np.allclose(np.array(offset), offset_target, atol=5)
    assert np.allclose(matrix, matrix_target)


def test_register_cc(locdata_blobs_2d):
    locdata_2d_transformed = transform_affine(locdata_blobs_2d, offset=(100, 50))

    offset_target = np.array([100.0, 50.0])
    matrix_target = np.array([[1, 0], [0, 1]])

    matrix, offset = register_cc(locdata_blobs_2d, locdata_2d_transformed, bin_size=50)
    assert np.allclose(np.array(offset), offset_target, atol=5)
    assert np.allclose(matrix, matrix_target)

    matrix, offset = register_cc(
        locdata_blobs_2d, locdata_2d_transformed, bin_size=(10, 50)
    )
    assert np.allclose(np.array(offset), offset_target, atol=5)
    assert np.allclose(matrix, matrix_target)
