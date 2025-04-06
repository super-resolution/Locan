import pytest

o3d = pytest.importorskip("open3d")

import numpy as np

from locan.data.adapter import open3d_point_cloud_to_2d_points, points_to_open3d


def test_points_to_open3d():
    points = np.asarray([1, 2, 3, 4])
    point_cloud_o3d = points_to_open3d(points=points)
    assert np.array(point_cloud_o3d.point.positions).shape == (4, 3)

    points = [[1], [2]]
    point_cloud_o3d = points_to_open3d(points=points)
    assert np.array(point_cloud_o3d.point.positions).shape == (2, 3)

    points = [[1, 2], [3, 4]]
    point_cloud_o3d = points_to_open3d(points=points)
    assert np.array(point_cloud_o3d.point.positions).shape == (2, 3)

    points = [[1, 2, 3], [4, 5, 6]]
    point_cloud_o3d = points_to_open3d(points=points)
    assert np.array(point_cloud_o3d.point.positions).shape == (2, 3)

    with pytest.raises(TypeError):
        points_to_open3d(points=[[1, 2, 3, 4], [1, 2, 3, 4]])


def test_open3d_point_cloud_to_2d_points():
    points_3d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    point_cloud_o3d = o3d.t.geometry.PointCloud(o3d.core.Tensor(points_3d))
    # legacy: point_cloud_o3d.points = o3d.utility.Vector3dVector(points_3d)
    points = open3d_point_cloud_to_2d_points(point_cloud=point_cloud_o3d)
    assert points.shape == (3, 2)
    np.array_equal(points[1], [4, 5])

    point_cloud_o3d = o3d.t.geometry.PointCloud()
    points = open3d_point_cloud_to_2d_points(point_cloud=point_cloud_o3d)
    assert points.shape == (0,)
