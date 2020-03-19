import pytest
import numpy as np
from scipy.spatial import Delaunay

from surepy.data.hulls.alpha_shape_2d import _circumcircle, _half_distance


def test__circumcircle_2d(locdata_2d):
    points = np.array([(0, 0), (1, 1+np.sqrt(2)), (1+np.sqrt(2), 1)])
    center, radius = _circumcircle(points, [2, 1, 0])
    assert radius == np.sqrt(2)
    assert np.array_equal(center, [1, 1])

    points = locdata_2d.coordinates
    triangulation = Delaunay(points)
    center, radius = _circumcircle(points, triangulation.simplices[0])
    assert radius == 1.8210786221487993
    assert center[0] == 3.357142857142857


def test__half_distance():
    points = np.array([(0, 0), (0, 1)])
    radius = _half_distance(points)
    assert radius == 0.5
