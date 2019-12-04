import numpy as np

from surepy.data.transform import transform_affine
from surepy.data.register import register_icp


def test_register_icp_2d(locdata_blobs_2d):
    locdata_2d_transformed = transform_affine(locdata_blobs_2d, offset=(100, 100))
    offset_target = np.array([100., 100.])
    matrix_target = np.array([[1, 0], [0, 1]])

    matrix, offset = register_icp(locdata_blobs_2d, locdata_2d_transformed, verbose=False)
    assert np.allclose(np.array(offset), offset_target)
    assert np.allclose(matrix, matrix_target)

    matrix, offset = register_icp(locdata_blobs_2d, locdata_2d_transformed, pre_translation=(-90, -90), verbose=False)
    offset_target = np.array([10., 10.])
    assert np.allclose(np.array(offset), offset_target)
    assert np.allclose(matrix, matrix_target)


def test_register_icp_3d(locdata_blobs_3d):
    locdata_3d_transformed = transform_affine(locdata_blobs_3d, offset=(100, 100, 100))
    offset_target = np.array([100., 100., 100.])
    matrix_target = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    matrix, offset = register_icp(locdata_blobs_3d, locdata_3d_transformed, verbose=False)
    assert np.allclose(np.array(offset), offset_target)
    assert np.allclose(matrix, matrix_target)

    matrix, offset = register_icp(locdata_blobs_3d, locdata_3d_transformed, pre_translation=(-90, -90, -90),
                                  verbose=False)
    offset_target = np.array([10., 10., 10.])
    assert np.allclose(np.array(offset), offset_target)
    assert np.allclose(matrix, matrix_target)