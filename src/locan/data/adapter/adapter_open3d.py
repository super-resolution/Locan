"""

Adapter functions for working with objects from open3d.

Note
----
We use the tensor modules in open3d.

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from locan.dependencies import HAS_DEPENDENCY, needs_package

if HAS_DEPENDENCY["open3d"]:
    import open3d as o3d

__all__: list[str] = ["points_to_open3d", "open3d_point_cloud_to_2d_points"]


@needs_package("open3d")
def points_to_open3d(points: npt.ArrayLike) -> o3d.t.geometry.PointCloud:
    """
    Convert 2 or 3-dimensional points into open3d point cloud.

    Parameters
    ----------
    points
        Coordinates of input points with shape (npoints, ndim).

    Returns
    -------
    open3d.t.geometry.PointCloud
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = np.expand_dims(points, axis=1)
        zeros = np.zeros(shape=(len(points), 2))
        points = np.concatenate((points, zeros), axis=1)
    elif points.shape[1] == 1:
        zeros = np.zeros(shape=(len(points), 2))
        points = np.concatenate((points, zeros), axis=1)
    elif points.shape[1] == 2:
        zeros = np.zeros(shape=(len(points), 1))
        points = np.concatenate((points, zeros), axis=1)
    elif points.shape[1] == 3:
        points = points
    else:
        raise TypeError("points must have a dimension <=3.")
    point_cloud_o3d = o3d.t.geometry.PointCloud()
    # legacy point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    point_cloud_o3d.point.positions = o3d.core.Tensor(points)
    return point_cloud_o3d


@needs_package("open3d")
def open3d_point_cloud_to_2d_points(
    point_cloud: o3d.t.geometry.PointCloud,
) -> npt.NDArray[np.float64]:
    """
    Convert an open3d point cloud into 2-dimensional points.

    Parameters
    ----------
    point_cloud
        Open3d point cloud.

    Returns
    -------
    npt.NDArray[np.float64]
    """
    if point_cloud.is_empty():
        points = np.array([])
    else:
        points = np.asarray(point_cloud.point.positions)[:, :2]
    return points
