import numpy as np
import pytest

from locan.data.register import register_cc, register_icp
from locan.data.transform import transform_affine
from locan.dependencies import HAS_DEPENDENCY


@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
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


def test_register_cc(locdata_blobs_2d):
    locdata_2d_transformed = transform_affine(locdata_blobs_2d, offset=(100, 50))
    offset_target = np.array([100.0, 50.0])
    matrix_target = np.array([[1, 0], [0, 1]])

    matrix, offset = register_cc(
        locdata_blobs_2d, locdata_2d_transformed, bin_size=50, verbose=False
    )
    assert np.allclose(np.array(offset), offset_target, atol=5)
    assert np.allclose(matrix, matrix_target)

    matrix, offset = register_cc(
        locdata_blobs_2d, locdata_2d_transformed, bin_size=(10, 50), verbose=False
    )
    assert np.allclose(np.array(offset), offset_target, atol=5)
    assert np.allclose(matrix, matrix_target)
