import pickle
import tempfile
from pathlib import Path

import matplotlib.patches as mPatches
import matplotlib.pyplot as plt  # needed for visual inspection
import numpy as np
import pytest
from matplotlib.path import Path as mplPath
from shapely.geometry import LineString as shLine
from shapely.geometry import MultiPolygon as shMultiPolygon
from shapely.geometry import Polygon as shPolygon

from locan import (
    AxisOrientedCuboid,
    AxisOrientedHypercuboid,
    AxisOrientedRectangle,
    Cuboid,
    Ellipse,
    EmptyRegion,
    Interval,
    LineSegment2D,
    LineSegment3D,
    MultiPolygon,
    Polygon,
    Rectangle,
    Region,
    Region1D,
    Region2D,
    Region3D,
    RegionND,
    Rotation2D,
    get_region_from_open3d,
    get_region_from_shapely,
)
from locan.data.regions.region import _polygon_path
from locan.dependencies import HAS_DEPENDENCY

if HAS_DEPENDENCY["open3d"]:
    import open3d as o3d


class TestRegion:

    def test_init(self):
        with pytest.raises(TypeError):
            Region()


class TestRegion1D:

    def test_init(self):
        with pytest.raises(TypeError):
            Region1D()


class TestRegion2D:

    def test_init(self):
        with pytest.raises(TypeError):
            Region2D()


class TestRegion3D:

    def test_init(self):
        with pytest.raises(TypeError):
            Region3D()


class TestRegionND:

    def test_init(self):
        with pytest.raises(TypeError):
            RegionND()


class TestEmptyRegion:

    def test_init(self):
        region = EmptyRegion()
        assert isinstance(region, Region)
        assert repr(region) == "EmptyRegion()"
        assert str(region) == "EmptyRegion()"

    def test_from_shapely(self):
        points = ()
        shapely_object = shLine(points)
        assert shapely_object.is_empty
        region = EmptyRegion.from_shapely(shapely_object)
        assert isinstance(region, EmptyRegion)

    def test_attributes(self):
        region = EmptyRegion()
        assert region.dimension is None
        assert len(region.points) == 0
        assert len(region.vertices) == 0
        assert region.centroid is None
        assert np.isnan(region.max_distance)
        assert np.isnan(region.elongation)
        assert np.isnan(region.radial_distance)
        assert np.isnan(region.isoperimetric_quotient)
        assert region.region_measure == 0
        assert region.subregion_measure == 0
        assert region.bounds is None
        assert region.extent is None
        assert isinstance(region.bounding_box, EmptyRegion)

    def test_methods(self):
        region = EmptyRegion()
        other = Rectangle()
        assert isinstance(region.intersection(other), EmptyRegion)
        assert region.symmetric_difference(other) is other
        assert region.union(other) is other
        assert region.contains([[9, 8], [10.5, 10.5], [100, 100], [11, 12]]).size == 0
        assert region.contains([(10.5, 10)]).size == 0
        assert region.contains([(100, 100)]).size == 0
        assert region.contains([]).size == 0
        region.as_artist()
        assert isinstance(region.shapely_object, shPolygon)
        assert isinstance(region.buffer(1), EmptyRegion)


class TestInterval:

    def test_init(self):
        region = Interval()
        assert 2 not in region
        assert 0.5 in region
        assert isinstance(region, Region)
        assert repr(region) == "Interval(0, 1)"
        assert str(region) == "Interval(0, 1)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, Interval)
        with pytest.raises(AttributeError):
            region.lower_bound = None
            region.upper_bound = None

        region = Interval(1, 2)
        assert repr(region) == "Interval(1, 2)"

        region = Interval.from_intervals((0, 2))
        assert repr(region) == "Interval(0, 2)"
        region = Interval.from_intervals([0, 2])
        assert repr(region) == "Interval(0, 2)"
        region = Interval.from_intervals(np.array([0, 2]))
        assert repr(region) == "Interval(0, 2)"
        region = Interval.from_intervals([(0, 2)])
        assert repr(region) == "Interval(0, 2)"

    def test_attributes(self):
        region = Interval()
        assert region.dimension == 1
        assert np.array_equal(region.bounds, (0, 1))
        assert np.array_equal(region.intervals, (0, 1))
        assert region.extent == 1
        assert np.allclose(region.points.astype(float), (0, 1))
        assert np.allclose(region.vertices.astype(float), (0, 1))
        assert region.centroid == 0.5
        assert region.max_distance == 1
        assert np.isnan(region.elongation)
        assert region.region_measure == 1
        assert region.subregion_measure == 0
        assert region.radial_distance == 0.5

        region = Interval(0, -1)
        assert region.dimension == 1
        assert np.array_equal(region.bounds, (-1, 0))
        assert np.array_equal(region.intervals, (-1, 0))
        assert region.extent == 1
        assert np.allclose(region.vertices.astype(float), (0, -1))
        assert region.centroid == -0.5
        assert region.max_distance == 1
        assert np.isnan(region.elongation)
        assert region.region_measure == 1
        assert region.subregion_measure == 0
        assert region.radial_distance == 0.5

    def test_methods(self):
        region = Interval()
        assert np.array_equal(region.contains((0, 0.5, 1, 2)), (0, 1))
        with pytest.raises(NotImplementedError):
            other = Interval()
            assert isinstance(region.intersection(other), EmptyRegion)
            assert region.symmetric_difference(other) is other
            assert region.union(other) is other
        assert region.contains((0.5,)) == (0,)
        assert region.contains((2,)).size == 0
        assert region.contains([]).size == 0
        # assert isinstance(region.as_artist(), mPatches)
        assert np.array_equal(region.buffer(1).points, [-1, 2])
        assert np.array_equal(region.buffer(1).vertices, [-1, 2])
        assert repr(region.buffer(1)) == "Interval(-1, 2)"


class TestLineSegment2D:

    def test_init(self):
        region = LineSegment2D(points=((0, 0), (1, 1)), is_directed=False)
        assert np.array_equal(region.vertices, ((0, 0), (1, 1)))
        assert region.origin is None
        assert region.is_directed is False
        region = LineSegment2D(points=((0, 0), (1, 1)))
        assert region.is_directed is True
        assert np.array_equal(region.origin, [0, 0])
        assert isinstance(region, Region)
        assert isinstance(region, Region2D)
        assert repr(region) == "LineSegment2D([[0, 0], [1, 1]], True)"
        assert str(region) == "LineSegment2D([[0, 0], [1, 1]], True)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, LineSegment2D)
        with pytest.raises(AttributeError):
            region.vertices = None
            region.origin = None

        region = LineSegment2D.from_intervals(((0, 2), (1, 1)))
        assert repr(region) == "LineSegment2D([[0, 1], [2, 1]], True)"

    def test_from_shapely(self):
        points = ((2, 2), (2, 3))
        shapely_object = shLine(points)
        region = LineSegment2D.from_shapely(shapely_object)
        assert isinstance(region, LineSegment2D)

    def test_attributes(self):
        region = LineSegment2D(points=((0, 0), (1, -1)))
        assert region.dimension == 2
        assert region.bounds == pytest.approx((0, -1, 1, 0))
        assert np.array_equal(region.intervals, [(0, 1), (-1, 0)])
        assert region.extent == pytest.approx((1, 1))
        assert len(region.vertices) == 2
        assert np.allclose(
            region.vertices.astype(float),
            [[0.0, 0.0], [1.0, -1.0]],
        )
        assert np.array_equal(region.centroid, (0.5, -0.5))
        assert region.max_distance == pytest.approx(np.sqrt(2))
        assert region.elongation == pytest.approx(1)
        assert region.subregion_measure == region.max_distance
        assert region.region_measure == 0
        assert region.radial_distance == pytest.approx(0.7071067811865476)

    def test_methods(self):
        region = LineSegment2D(points=((0, 0), (1, 1)))
        assert (0.1, 0.1) in region
        assert (0, 1) not in region
        assert np.array_equal(
            region.contains([[0.5, 0.5], [-0.5, 0.5], [100, 100], [-1, 2]]), (0,)
        )
        assert region.contains([(0.5, 0.5)]) == (0,)
        assert region.contains([(10, 10)]).size == 0
        assert region.contains([]).size == 0

        other = LineSegment2D(points=((1, 2), (0, 0)))
        with pytest.raises(TypeError):
            assert isinstance(region.intersection(other), Polygon)
        with pytest.raises(TypeError):
            assert isinstance(region.symmetric_difference(other), MultiPolygon)
        with pytest.raises(TypeError):
            assert isinstance(region.union(other), Polygon)

        assert isinstance(region.as_artist(), mPatches.PathPatch)
        assert isinstance(region.shapely_object, shLine)
        assert region.region_measure == region.shapely_object.area
        assert isinstance(region.buffer(1), Polygon)
        assert np.array_equal(region.bounding_box.corner, (0.0, 0.0))
        assert region.bounding_box.width == pytest.approx(1)
        assert region.bounding_box.height == pytest.approx(1)

    @pytest.mark.visual
    def test_LineSegment2D_visual(self):
        region = LineSegment2D([(0, 2), (3, -1)])
        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*region.vertices.T, marker="o", color="Blue")
        ax.add_patch(region.as_artist(origin=(0, 0), alpha=0.2))
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)

        for angle in [30, 60, 90, 120, 150, 180, 275, 350, -10, -20, -30]:
            region_rotated = Rotation2D.from_angle(angle=angle, degrees=True).apply(
                region.vertices
            )
            region_rotated = LineSegment2D.from_intervals(region_rotated.T)
            ax.add_patch(region_rotated.as_artist(origin=(0, 0), alpha=0.2))
            ax.plot(*region_rotated.centroid, "*", color="Green")
        plt.show()
        plt.close("all")


class TestAxisOrientedRectangle:

    def test_init(self):
        region = AxisOrientedRectangle((0, 0), 2, 1)
        assert (10, 1) not in region
        assert (0.1, 0.1) in region
        assert (0.5, 0.5) in region
        assert isinstance(region, Region)
        assert repr(region) == "AxisOrientedRectangle((0, 0), 2, 1)"
        assert str(region) == "AxisOrientedRectangle((0, 0), 2, 1)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, AxisOrientedRectangle)
        with pytest.raises(AttributeError):
            region.corner = None
            region.width = None
            region.height = None

        region = AxisOrientedRectangle.from_intervals(((0, 2), (0, 1)))
        assert repr(region) == "AxisOrientedRectangle((0, 0), 2, 1)"
        region = AxisOrientedRectangle.from_intervals([(0, 2), (0, 1)])
        assert repr(region) == "AxisOrientedRectangle((0, 0), 2, 1)"
        region = AxisOrientedRectangle.from_intervals(np.array([(0, 2), (0, 1)]))
        assert repr(region) == "AxisOrientedRectangle((0, 0), 2, 1)"

    def test_attributes(self):
        region = AxisOrientedRectangle((0, 0), 2, 1)
        assert region.dimension == 2
        assert region.bounds == pytest.approx((0, 0, 2, 1))
        assert np.array_equal(region.intervals, [(0, pytest.approx(2)), (0, 1)])
        assert region.extent == pytest.approx((2, 1))
        assert len(region.points) == 5
        assert np.allclose(
            region.points.astype(float),
            [[0.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0], [0.0, 0.0]],
        )
        assert np.allclose(
            region.vertices.astype(float),
            [[0.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]],
        )
        assert np.array_equal(region.centroid, (1, 0.5))
        assert region.max_distance == np.sqrt(5)
        assert region.elongation == pytest.approx(0.5)
        assert region.region_measure == 2
        assert region.subregion_measure == 6
        assert region.radial_distance == pytest.approx(1.118033988749895)
        assert region.isoperimetric_quotient == pytest.approx(0.6981317007977318)

        region = AxisOrientedRectangle((0, 0), 2, -1)
        assert region.dimension == 2
        assert region.bounds == pytest.approx((0, -1, 2, 0))
        assert np.array_equal(region.intervals, [(0, pytest.approx(2)), (-1, 0)])
        assert region.extent == pytest.approx((2, 1))
        assert np.allclose(
            region.vertices.astype(float),
            [[0.0, 0.0], [0.0, -1.0], [2.0, -1.0], [2.0, 0.0]],
        )
        assert np.array_equal(region.centroid, (1, -0.5))
        assert region.max_distance == np.sqrt(5)
        assert region.elongation == pytest.approx(0.5)
        assert region.region_measure == 2
        assert region.subregion_measure == 6
        assert region.radial_distance == pytest.approx(1.118033988749895)
        assert region.isoperimetric_quotient == pytest.approx(0.6981317007977318)

    def test_methods(self):
        region = AxisOrientedRectangle((0, 0), 2, 1)
        assert np.array_equal(
            region.contains(
                [[0, 0], [0.5, 0.5], [1, 0.8], [1, 0.2], [100, 100], [-1, 2]]
            ),
            (1, 2, 3),
        )
        other = AxisOrientedRectangle((0, 0), 2, 1)
        assert isinstance(region.intersection(other), Polygon)
        assert isinstance(region.symmetric_difference(other), EmptyRegion)
        assert isinstance(region.union(other), Polygon)
        assert region.contains([(0.5, 1)]) == (0,)
        assert region.contains([(10, 10)]).size == 0
        assert region.contains([]).size == 0
        assert isinstance(region.as_artist(), mPatches.Rectangle)
        assert isinstance(region.shapely_object, shPolygon)
        assert region.region_measure == region.shapely_object.area
        assert isinstance(region.buffer(1), Polygon)
        assert np.array_equal(region.bounding_box.corner, (0.0, 0.0))
        assert region.bounding_box.width == pytest.approx(2)
        assert region.bounding_box.height == pytest.approx(1)


class TestRectangle:

    def test_init(self):
        region = Rectangle((0, 0), 2, 1, 90)
        assert isinstance(region, Region)
        assert repr(region) == "Rectangle((0, 0), 2, 1, 90)"
        assert str(region) == "Rectangle((0, 0), 2, 1, 90)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, Rectangle)
        with pytest.raises(AttributeError):
            region.corner = None
            region.width = None
            region.height = None
            region.angle = None

    def test_attributes(self):
        region = Rectangle((0, 0), 2, 1, 90)
        assert region.dimension == 2
        assert isinstance(region.rotation, Rotation2D)
        assert region.rotation.as_angle(degrees=True) == pytest.approx(90)
        assert region.bounds == pytest.approx((-1, 0, 0, 2))
        assert np.array_equal(region.intervals, [(-1, pytest.approx(0)), (0, 2)])
        assert region.extent == pytest.approx((1, 2))
        assert len(region.points) == 5
        assert len(region.vertices) == 4
        assert np.allclose(
            region.points.astype(float),
            [[0.0, 0.0], [-1.0, 0.0], [-1.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
        )
        assert np.allclose(
            region.vertices.astype(float),
            [[0.0, 0.0], [-1.0, 0.0], [-1.0, 2.0], [0.0, 2.0]],
        )
        assert np.array_equal(region.centroid, (-0.5, 1))
        assert region.max_distance == np.sqrt(5)
        assert region.elongation == pytest.approx(0.5)
        assert region.region_measure == 2
        assert region.subregion_measure == 6
        assert region.radial_distance == pytest.approx(1.118033988749895)
        assert region.isoperimetric_quotient == pytest.approx(0.6981317007977318)

        region = Rectangle((0, 0), 2, -1, 0)
        assert region.dimension == 2
        assert isinstance(region.rotation, Rotation2D)
        assert region.rotation.as_angle(degrees=True) == pytest.approx(0)
        assert region.bounds == pytest.approx((0, -1, 2, 0))
        assert np.allclose(region.intervals.ravel(), [0, 2, -1, 0])
        assert region.extent == pytest.approx((2, 1))
        assert len(region.points) == 5
        assert len(region.vertices) == 4
        assert np.allclose(
            region.points.astype(float),
            [[0.0, 0.0], [0.0, -1.0], [2.0, -1.0], [2.0, 0.0], [0.0, 0.0]],
        )
        assert np.allclose(
            region.vertices.astype(float),
            [[0.0, 0.0], [0.0, -1.0], [2.0, -1.0], [2.0, 0.0]],
        )
        assert np.array_equal(region.centroid, (1, -0.5))
        assert region.max_distance == np.sqrt(5)
        assert region.elongation == pytest.approx(0.5)
        assert region.region_measure == 2
        assert region.subregion_measure == 6
        assert region.radial_distance == pytest.approx(1.118033988749895)
        assert region.isoperimetric_quotient == pytest.approx(0.6981317007977318)

    def test_methods(self):
        region = Rectangle((0, 0), 2, 1, 90)
        assert np.array_equal(
            region.contains([[0, 0], [-0.5, 0.5], [100, 100], [-1, 2]]), (1,)
        )
        other = Rectangle((0, 0), 2, 1, 0)
        assert isinstance(region.intersection(other), Polygon)
        assert isinstance(region.symmetric_difference(other), MultiPolygon)
        assert isinstance(region.union(other), Polygon)
        assert (10, 1) not in region
        assert (-0.5, 0.5) in region
        assert region.contains([(-0.5, 1)]) == (0,)
        assert region.contains([(10, 10)]).size == 0
        assert region.contains([]).size == 0
        assert isinstance(region.as_artist(), mPatches.Rectangle)
        assert isinstance(region.shapely_object, shPolygon)
        assert region.region_measure == region.shapely_object.area
        assert isinstance(region.buffer(1), Polygon)
        assert np.array_equal(region.bounding_box.corner, (-1.0, 0.0))
        assert region.bounding_box.width == pytest.approx(1)
        assert region.bounding_box.height == pytest.approx(2)

        region = Rectangle.from_intervals(((0, 2), (0, 1)))
        assert repr(region) == "AxisOrientedRectangle((0, 0), 2, 1)"
        region = Rectangle.from_intervals([(0, 2), (0, 1)])
        assert repr(region) == "AxisOrientedRectangle((0, 0), 2, 1)"
        region = Rectangle.from_intervals(np.array([(0, 2), (0, 1)]))
        assert repr(region) == "AxisOrientedRectangle((0, 0), 2, 1)"

    @pytest.mark.visual
    def test_Rectangle_visual(self):
        region = Rectangle((0, 0), 2, 1, 90)
        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*region.points.T, marker="o", color="Blue")
        ax.plot(*region.vertices.T, marker="o", color="Blue")
        ax.add_patch(region.as_artist(origin=(0, 0), fill=True, alpha=0.2))
        ax.plot(
            *np.array(region.shapely_object.exterior.coords).T, marker=".", color="Red"
        )
        ax.plot(*region.centroid, "*", color="Green")
        ax.plot(
            *np.array(region.buffer(1).exterior.coords).T, marker=".", color="Yellow"
        )
        region.plot(color="Green", alpha=0.2)
        plt.show()
        plt.close("all")


class TestEllipse:

    def test_init(self):
        region = Ellipse((10, 10), 4, 2, 90)
        assert isinstance(region, Region)
        assert repr(region) == "Ellipse((10, 10), 4, 2, 90)"
        assert str(region) == "Ellipse((10, 10), 4, 2, 90)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, Ellipse)
        with pytest.raises(AttributeError):
            region.center = None
            region.width = None
            region.height = None
            region.angle = None

    def test_attributes(self):
        region = Ellipse((10, 10), 4, 2, 90)
        assert region.dimension == 2
        assert isinstance(region.rotation, Rotation2D)
        assert region.rotation.as_angle(degrees=True) == pytest.approx(90)
        assert region.bounds == pytest.approx((9, 8, 11, 12))
        assert region.extent == pytest.approx((2, 4))
        assert len(region.points) == 65 or len(region.points) == 66
        assert len(region.vertices) == 65 or len(region.vertices) == 66
        assert np.array_equal(region.centroid, (10, 10))
        assert region.max_distance == 4
        assert region.elongation == pytest.approx(0.5)
        assert region.region_measure == pytest.approx(6.283185307179586)
        assert region.subregion_measure == pytest.approx(9.688448216130086)
        assert region.radial_distance == pytest.approx(1.5490111263409625)
        assert region.isoperimetric_quotient == pytest.approx(0.8411651817734023)
        assert isinstance(region.major_axis, LineSegment2D)
        assert np.array_equal(region.major_axis.vertices, ((10, 8), (10, 12)))
        assert isinstance(region.minor_axis, LineSegment2D)
        assert np.array_equal(region.minor_axis.vertices, ((11, 10), (9, 10)))
        assert region.eccentricity == pytest.approx(0.8660254037844386)

    def test_methods(self):
        region = Ellipse((10, 10), 4, 2, 90)
        assert np.array_equal(
            region.contains([[9, 8], [10.5, 10.5], [100, 100], [11, 12]]), (1,)
        )
        other = Rectangle((10, 10), 10, 10, 0)
        assert isinstance(region.intersection(other), Polygon)
        assert isinstance(region.symmetric_difference(other), MultiPolygon)
        assert isinstance(region.union(other), Polygon)
        assert (1, 1) not in region
        assert (10, 10) in region
        assert region.contains([(10.5, 10)]) == (0,)
        assert region.contains([(100, 100)]).size == 0
        assert region.contains([]).size == 0
        assert isinstance(region.as_artist(), mPatches.Ellipse)
        assert isinstance(region.shapely_object, shPolygon)
        assert region.region_measure == pytest.approx(
            region.shapely_object.area, rel=10e-3
        )
        assert isinstance(region.buffer(1), Polygon)
        assert np.array_equal(region.bounding_box.corner, (9, 8))
        assert region.bounding_box.width == pytest.approx(2)
        assert region.bounding_box.height == pytest.approx(4)

    @pytest.mark.visual
    def test_Ellipse_visual(self):
        region = Ellipse((10, 10), 4, 2, 90)
        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*region.points.T, marker="o", color="Blue")
        ax.plot(*region.vertices.T, marker="o", color="Blue")
        ax.plot(
            *np.array(region.shapely_object.exterior.coords).T, marker=".", color="Red"
        )
        ax.add_patch(region.as_artist(origin=(0, 0), fill=True, alpha=0.2))
        ax.plot(
            *np.array(region.buffer(1).exterior.coords).T, marker=".", color="Yellow"
        )
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)
        plt.show()
        plt.close("all")


class TestPolygon:

    def test_init(self):
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        region = Polygon(points)
        assert isinstance(region, Region)
        assert np.array_equal(region.vertices, points[:-1])
        assert (
            repr(region) == "Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5]])"
        )
        assert str(region) == "Polygon(<self.vertices>, <self.holes>)"

        region = Polygon(points[:-1])
        assert np.array_equal(region.vertices, points[:-1])
        assert (
            repr(region) == "Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5]])"
        )
        assert str(region) == "Polygon(<self.vertices>, <self.holes>)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, Polygon)
        assert np.array_equal(region.vertices, points[:-1])

    def test_attributes(self):
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        region = Polygon(points)
        assert region.dimension == 2
        assert np.array_equal(region.vertices, points[:-1])
        assert np.array_equal(
            region.centroid,
            (
                pytest.approx(0.4444444444444444),
                pytest.approx(0.611111111111111),
            ),
        )
        assert region.max_distance == np.sqrt(2)
        assert region.region_measure == pytest.approx(0.75)
        assert region.subregion_measure == pytest.approx(3.618033988749895)
        assert region.radial_distance == pytest.approx(0.6477251522831856)
        assert region.isoperimetric_quotient == pytest.approx(0.719988968918596)

    def test_methods(self):
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        region = Polygon(points)
        assert np.array_equal(
            region.contains([[0, 0], [0.2, 0.8], [100, 100], [1, 0.5]]), (1,)
        )
        assert region.contains([(0.2, 0.8)]) == (0,)
        assert region.contains([(100, 100)]).size == 0
        assert region.contains([]).size == 0
        assert (10, 1) not in region
        assert (0.5, 0.5) in region
        assert isinstance(region.as_artist(), mPatches.PathPatch)
        assert isinstance(region.shapely_object, shPolygon)
        assert region.region_measure == pytest.approx(
            region.shapely_object.area, rel=10e-3
        )
        assert isinstance(region.buffer(1), Polygon)
        assert np.array_equal(region.bounding_box.corner, (0, 0))
        assert region.bounding_box.width == pytest.approx(1)
        assert region.bounding_box.height == pytest.approx(1)

    @pytest.mark.visual
    def test_Polygon_visual(self):
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        region = Polygon(points)

        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*region.vertices.T, marker=".", color="Blue")
        ax.plot(
            *np.array(region.shapely_object.exterior.coords).T, marker=".", color="Red"
        )
        ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        ax.plot(
            *np.array(region.buffer(1).exterior.coords).T, marker=".", color="Yellow"
        )
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)
        plt.show()

        # visualize points inside
        points = np.random.default_rng().random(size=(1000, 2))
        points_inside = points[region.contains(points)]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(*points.T, marker=".", color="Gray")
        ax.scatter(*points_inside.T, marker=".", color="Blue")
        ax.plot(
            *np.array(region.shapely_object.exterior.coords).T, marker=".", color="Red"
        )
        ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        plt.show()

        plt.close("all")

    def test_Polygon_with_holes(self):
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region = Polygon(points, holes)
        assert np.array_equal(region.vertices, points[:-1])
        assert (
            repr(region) == "Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5]], "
            "[[[0.2, 0.2], [0.2, 0.3], [0.3, 0.3], [0.3, 0.25]], "
            "[[0.4, 0.4], [0.4, 0.5], [0.5, 0.5], [0.5, 0.45]]])"
        )
        region = Polygon(points[:-1], holes)
        assert isinstance(region, Region)
        assert (
            repr(region) == "Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5]], "
            "[[[0.2, 0.2], [0.2, 0.3], [0.3, 0.3], [0.3, 0.25]], "
            "[[0.4, 0.4], [0.4, 0.5], [0.5, 0.5], [0.5, 0.45]]])"
        )
        assert str(region) == "Polygon(<self.vertices>, <self.holes>)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, Polygon)
        assert region.dimension == 2
        assert np.array_equal(region.points, points)
        assert np.array_equal(region.vertices, points[:-1])
        assert np.array_equal(
            region.centroid,
            (
                pytest.approx(0.446485260770975),
                pytest.approx(0.6162131519274376),
            ),
        )
        assert region.max_distance == np.sqrt(2)
        assert region.region_measure == pytest.approx(0.735)
        assert region.subregion_measure == pytest.approx(4.341640786499874)
        assert region.radial_distance == pytest.approx(0.6472153862069733)
        assert np.array_equal(
            region.contains([[0, 0], [0.2, 0.8], [100, 100], [1, 0.5]]), (1,)
        )
        assert region.contains([(0.2, 0.8)]) == (0,)
        assert region.contains([(100, 100)]).size == 0
        assert region.contains([]).size == 0
        assert isinstance(region.as_artist(), mPatches.PathPatch)
        assert isinstance(region.shapely_object, shPolygon)
        assert region.region_measure == pytest.approx(
            region.shapely_object.area, rel=10e-3
        )
        assert isinstance(region.buffer(1), Polygon)
        assert np.array_equal(region.bounding_box.corner, (0, 0))
        assert region.bounding_box.width == pytest.approx(1)
        assert region.bounding_box.height == pytest.approx(1)

    @pytest.mark.visual
    def test_Polygon_with_holes_visual(self):
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region = Polygon(points, holes)
        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*region.points.T, marker="o", color="Blue")
        ax.plot(*region.vertices.T, marker="o", color="Blue")
        ax.plot(
            *np.array(region.shapely_object.exterior.coords).T, marker=".", color="Red"
        )
        ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        ax.plot(
            *np.array(region.buffer(0.01).exterior.coords).T, marker=".", color="Yellow"
        )
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)
        plt.show()

        plt.close("all")

    def test_Polygon_from_shapely(self):
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        shapely_polygon = shPolygon(points, holes)
        region = Polygon.from_shapely(shapely_polygon)
        assert np.array_equal(region.points, points)
        assert np.array_equal(region.vertices, points[:-1])
        assert isinstance(region, Region)
        assert (
            repr(region) == "Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5]], "
            "[[[0.2, 0.2], [0.2, 0.3], [0.3, 0.3], [0.3, 0.25], [0.2, 0.2]], "
            "[[0.4, 0.4], [0.4, 0.5], [0.5, 0.5], [0.5, 0.45], [0.4, 0.4]]])"
        )
        assert str(region) == "Polygon(<self.vertices>, <self.holes>)"
        assert region.dimension == 2
        assert np.array_equal(region.points, points)
        assert np.array_equal(region.vertices, points[:-1])
        assert np.array_equal(
            region.centroid,
            (
                pytest.approx(0.446485260770975),
                pytest.approx(0.6162131519274376),
            ),
        )
        assert region.max_distance == np.sqrt(2)
        assert region.region_measure == pytest.approx(0.735)
        assert region.subregion_measure == pytest.approx(4.341640786499874)

        assert np.array_equal(
            region.contains([[0, 0], [0.2, 0.8], [100, 100], [1, 0.5]]), (1,)
        )
        assert region.contains([(0.2, 0.8)]) == (0,)
        assert region.contains([(100, 100)]).size == 0
        assert region.contains([]).size == 0
        assert isinstance(region.as_artist(), mPatches.PathPatch)
        assert isinstance(region.shapely_object, shPolygon)
        assert region.region_measure == pytest.approx(
            region.shapely_object.area, rel=10e-3
        )
        assert isinstance(region.buffer(1), Polygon)

    @pytest.mark.visual
    def test_Polygon_from_shapely_visual(self):
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        shapely_polygon = shPolygon(points, holes)
        region = Polygon.from_shapely(shapely_polygon)

        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*region.points.T, marker="o", color="Blue")
        ax.plot(*region.vertices.T, marker="o", color="Blue")
        ax.plot(
            *np.array(region.shapely_object.exterior.coords).T, marker=".", color="Red"
        )
        ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        ax.plot(
            *np.array(region.buffer(0.01).exterior.coords).T, marker=".", color="Yellow"
        )
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)
        plt.show()

        plt.close("all")


class TestMultiPolygon:

    def test_init(self):
        points = ((2, 2), (2, 3), (3, 3), (3, 2.5), (2, 2))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region_1 = Polygon(points, holes)
        region = MultiPolygon([region_0, region_1])

        assert (10, 1) not in region
        assert (0.2, 0.8) in region
        assert isinstance(region, Region)
        assert (
            repr(region)
            == "MultiPolygon([Polygon([[2.0, 2.0], [2.0, 3.0], [3.0, 3.0], [3.0, 2.5]]), Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5]], [[[0.2, 0.2], [0.2, 0.3], [0.3, 0.3], [0.3, 0.25]], [[0.4, 0.4], [0.4, 0.5], [0.5, 0.5], [0.5, 0.45]]])])"
        )
        assert str(region) == "MultiPolygon(<self.polygons>)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, MultiPolygon)

    def test_attributes(self):
        points = ((2, 2), (2, 3), (3, 3), (3, 2.5), (2, 2))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region_1 = Polygon(points, holes)
        region = MultiPolygon([region_0, region_1])

        assert region.dimension == 2
        assert len(region.polygons) == 2
        assert len(region.points) == 2
        assert len(region.vertices) == 2
        assert len(region.holes) == 2
        assert np.array_equal(
            region.centroid,
            (
                pytest.approx(1.4555555555555553),
                pytest.approx(1.6237373737373733),
            ),
        )
        assert region.max_distance == pytest.approx(4.242640687119285)
        with pytest.raises(NotImplementedError):
            region.elongation  # noqa: B018
        assert region.region_measure == pytest.approx(1.485)
        assert region.subregion_measure == pytest.approx(7.959674775249769)
        assert region.radial_distance == pytest.approx(1.4669234564865672)

    def test_methods(self):
        points = ((2, 2), (2, 3), (3, 3), (3, 2.5), (2, 2))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region_1 = Polygon(points, holes)
        region = MultiPolygon([region_0, region_1])

        assert np.array_equal(
            region.contains([[0, 0], [0.2, 0.8], [100, 100], [1, 0.5]]), (1,)
        )
        assert region.contains([(0.2, 0.8)]) == (0,)
        assert region.contains([(100, 100)]).size == 0
        assert region.contains([]).size == 0
        assert isinstance(region.as_artist(), mPatches.PathPatch)
        assert isinstance(region.shapely_object, shMultiPolygon)
        assert region.region_measure == pytest.approx(
            region.shapely_object.area, rel=10e-3
        )
        assert isinstance(region.buffer(1), Polygon)
        assert np.array_equal(region.bounds, (0, 0, 3, 3))
        assert np.array_equal(region.extent, (3, 3))
        assert np.array_equal(region.bounding_box.corner, (0, 0))
        assert region.bounding_box.width == pytest.approx(3)
        assert region.bounding_box.height == pytest.approx(3)

    @pytest.mark.visual
    def test_MultiPolygon_visual(self):
        points = ((2, 2), (2, 3), (3, 3), (3, 2.5), (2, 2))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region_1 = Polygon(points, holes)
        region = MultiPolygon([region_0, region_1])

        _fig, ax = plt.subplots(nrows=1, ncols=1)
        for polygon in region.polygons:
            ax.plot(*polygon.vertices.T, marker="o", color="Blue")
            ax.plot(
                *np.array(polygon.shapely_object.exterior.coords).T,
                marker=".",
                color="Red",
            )
        ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        ax.plot(
            *np.array(region.buffer(1).exterior.coords).T, marker=".", color="Yellow"
        )
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)
        plt.show()

        # visualize points inside
        # points = np.random.default_rng().random(size=(100, 2)) * 5
        # points_inside = points[region.contains(points)]
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.plot(*points.T, '.', markersize=1, color='Gray')
        # ax.plot(*points_inside.T, 'o', markersize=1, color='Blue')
        # for polygon in region.polygons:
        #     ax.plot(*np.array(polygon.shapely_object.exterior.coords).T, marker='.', color='Red')
        # ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        # plt.show()

        plt.close("all")


class TestPolygonOperations:

    def test_Polygon_union(self):
        points = ((2, 2), (2, 3), (3, 3), (3, 2.5), (2, 2))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region_1 = Polygon(points, holes)
        region = region_0.union(region_1)
        assert isinstance(region, Region)
        assert isinstance(region, Region2D)
        assert isinstance(region, MultiPolygon)
        assert region.dimension == 2
        assert len(region.polygons) == 2
        assert len(region.points) == 2
        assert len(region.vertices) == 2
        assert len(region.holes) == 2

    @pytest.mark.visual
    def test_Polygon_union_visual(self):
        points = ((2, 2), (2, 3), (3, 3), (3, 2.5), (2, 2))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region_1 = Polygon(points, holes)
        region = region_0.union(region_1)

        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.add_patch(region_0.as_artist(fill=True, alpha=0.2, color="Red"))
        ax.add_patch(region_1.as_artist(fill=True, alpha=0.2, color="Green"))
        ax.add_patch(region.as_artist(fill=True, alpha=0.2, color="Blue"))
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)
        plt.show()

        # visualize points inside
        points = np.random.default_rng().random(size=(1_000, 2)) * 10
        points_inside = points[region.contains(points)]
        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*points_inside.T, ".", markersize=1, color="Blue")
        ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        plt.show()

        plt.close("all")

    def test_Polygon_union_ragged(self):
        points = ((0, 0), (0, 10), (10, 10), (10, 0))
        holes = [((1, 1), (1, 9), (4, 9), (4, 1))]
        region_0 = Polygon(points, holes)
        points = ((2, 2), (2, 5), (3, 5), (3, 2))
        holes = [((2.5, 2.5), (2.5, 4), (2.6, 4), (2.6, 2.5))]
        region_1 = Polygon(points, holes)
        region = region_0.union(region_1)
        assert region.region_measure == pytest.approx(
            region_0.region_measure + region_1.region_measure
        )

    @pytest.mark.visual
    def test_Polygon_union_ragged_visual(self):
        points = ((0, 0), (0, 10), (10, 10), (10, 0))
        holes = [((1, 1), (1, 9), (4, 9), (4, 1))]
        region_0 = Polygon(points, holes)
        points = ((2, 2), (2, 5), (3, 5), (3, 2))
        holes = [((2.5, 2.5), (2.5, 4), (2.6, 4), (2.6, 2.5))]
        region_1 = Polygon(points, holes)
        region = region_0.union(region_1)

        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.add_patch(region_0.as_artist(fill=True, alpha=0.2, color="Red"))
        ax.add_patch(region_1.as_artist(fill=True, alpha=0.2, color="Green"))
        ax.add_patch(region.as_artist(fill=True, alpha=0.2, color="Blue"))
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)
        plt.show()

        # visualize points inside
        points = np.random.default_rng().random(size=(1_000, 2)) * 10
        points_inside = points[region.contains(points)]
        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*points_inside.T, ".", markersize=1, color="Blue")
        ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        plt.show()

        plt.close("all")

    def test_Polygon_intersection(self):
        points = ((0, 2), (0, 3), (3, 3), (3, 2.5), (2, 2))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region_1 = Polygon(points, holes)
        region = region_0.intersection(region_1)
        assert isinstance(region, EmptyRegion)

        points = ((0, 0), (0, 3), (3, 3), (3, 2), (2, 1.5))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region_1 = Polygon(points, holes)
        region = region_0.intersection(region_1)
        assert isinstance(region, Polygon)
        assert region.dimension == 2
        assert len(region.points) == 5
        assert len(region.vertices) == 4
        assert len(region.holes) == 2

    @pytest.mark.visual
    def test_Polygon_intersection_visual(self):
        points = ((0, 0), (0, 3), (3, 3), (3, 2), (2, 1.5))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        region_1 = Polygon(points, holes)
        region = region_0.intersection(region_1)

        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.add_patch(region_0.as_artist(fill=True, alpha=0.2, color="Red"))
        ax.add_patch(region_1.as_artist(fill=True, alpha=0.2, color="Green"))
        ax.add_patch(region.as_artist(fill=True, alpha=0.2, color="Blue"))
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)
        plt.show()

        # visualize points inside
        points = np.random.default_rng().random(size=(100_000, 2)) * 10
        points_inside = points[region.contains(points)]
        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*points_inside.T, ".", markersize=1, color="Blue")
        ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        plt.show()

        plt.close("all")

    def test_Polygon_difference(self):
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.4), (0.4, 0.4), (0.3, 0.25)),
            ((0.5, 0.5), (0.5, 0.8), (0.8, 0.8), (0.7, 0.45)),
        ]
        region_0 = Polygon(points, holes)
        other_region = Rectangle(corner=(0.5, 0.2), width=0.5, height=0.5, angle=0)
        region = region_0.symmetric_difference(other_region)

        points = ((0, 0), (0, 3), (3, 3), (3, 2), (2, 1.5))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        other_region = Polygon(points, holes)
        region = region_0.symmetric_difference(other_region)
        # print(repr(region))
        assert isinstance(region, MultiPolygon)
        assert region.dimension == 2
        assert len(region.polygons) == 4

    @pytest.mark.visual
    def test_Polygon_difference_visual(self):
        points = ((0, 0), (0, 3), (3, 3), (3, 2), (2, 1.5))
        region_0 = Polygon(points)
        points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
        holes = [
            ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
            ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
        ]
        other_region = Polygon(points, holes)
        region = region_0.symmetric_difference(other_region)

        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.add_patch(region_0.as_artist(fill=True, alpha=0.2, color="Red"))
        ax.add_patch(other_region.as_artist(fill=True, alpha=0.2, color="Green"))
        ax.add_patch(region.as_artist(fill=True, alpha=0.2, color="Blue"))
        ax.plot(*region.centroid, "*", color="Green")
        region.plot(color="Green", alpha=0.2)
        plt.show()

        # visualize points inside
        points = np.random.default_rng().random(size=(100_000, 2)) * 10
        points_inside = points[region.contains(points)]
        _fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(*points_inside.T, ".", markersize=1, color="Blue")
        ax.add_patch(region.as_artist(fill=True, alpha=0.2))
        plt.show()

        plt.close("all")


class TestLineSegment3D:

    def test_init(self):
        region = LineSegment3D(points=((0, 0, 0), (1, 1, 1)), is_directed=False)
        assert np.array_equal(region.vertices, ((0, 0, 0), (1, 1, 1)))
        assert region.origin is None
        assert region.is_directed is False
        region = LineSegment3D(points=((0, 0, 0), (1, 1, 1)))
        assert region.is_directed is True
        assert np.array_equal(region.origin, [0, 0, 0])
        assert isinstance(region, Region)
        assert isinstance(region, Region3D)
        assert repr(region) == "LineSegment3D([[0, 0, 0], [1, 1, 1]], True)"
        assert str(region) == "LineSegment3D([[0, 0, 0], [1, 1, 1]], True)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, LineSegment3D)
        with pytest.raises(AttributeError):
            region.vertices = None
            region.origin = None

        region = LineSegment3D.from_intervals(((0, 2), (1, 1), (3, 1)))
        assert repr(region) == "LineSegment3D([[0, 1, 3], [2, 1, 1]], True)"

    def test_attributes(self):
        region = LineSegment3D(points=((0, 0, 0), (1, 1, 1)))
        assert region.dimension == 3
        assert region.bounds == pytest.approx((0, 0, 0, 1, 1, 1))
        assert np.array_equal(region.intervals, [(0, 1), (0, 1), (0, 1)])
        assert region.extent == pytest.approx((1, 1, 1))
        assert len(region.vertices) == 2
        assert np.allclose(
            region.vertices.astype(float),
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        )
        assert np.array_equal(region.centroid, (0.5, 0.5, 0.5))
        assert region.max_distance == pytest.approx(np.sqrt(3))
        assert region.elongation == pytest.approx(1)
        assert region.subregion_measure == region.max_distance
        assert region.region_measure == 0
        assert region.radial_distance == pytest.approx(0.8660254037844386)

    def test_methods(self):
        region = LineSegment3D(points=((0, 0, 0), (1, 1, 1)))
        assert (0.1, 0.1, 0.1) in region
        assert (0, 1, 1) not in region
        assert np.array_equal(
            region.contains(
                [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [100, 100, 100], [-1, 2, 2]]
            ),
            (0,),
        )
        assert region.contains([(0.5, 0.5, 0.5)]) == (0,)
        assert region.contains([(10, 10, 10)]).size == 0
        assert region.contains([]).size == 0

        other = LineSegment3D(points=((1, 1, 1), (0, 0, 0)))
        with pytest.raises(NotImplementedError):
            assert isinstance(region.intersection(other), Polygon)
        with pytest.raises(NotImplementedError):
            assert isinstance(region.symmetric_difference(other), MultiPolygon)
        with pytest.raises(NotImplementedError):
            assert isinstance(region.union(other), Polygon)

        # assert isinstance(region.as_artist(), mPatches.PathPatch)
        # assert isinstance(region.buffer(1), Polygon)
        assert np.array_equal(region.bounding_box.corner, (0.0, 0.0, 0.0))
        assert region.bounding_box.width == pytest.approx(1)
        assert region.bounding_box.height == pytest.approx(1)
        assert region.bounding_box.length == pytest.approx(1)

    @pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
    def test_attributes_open3d(self):
        region = LineSegment3D(points=((0, 0, 0), (1, 1, 1)))
        assert isinstance(region.open3d_object, o3d.t.geometry.LineSet)
        assert np.array_equal(region.open3d_object.point.positions, region.vertices)

    @pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
    def test_from_open3d(self):
        lineset = o3d.t.geometry.LineSet()
        region = LineSegment3D.from_open3d(open3d_object=lineset)
        assert isinstance(region, EmptyRegion)

        lineset.point.positions = o3d.core.Tensor([[0, 0, 0], [1, 1, 1]])
        region = LineSegment3D.from_open3d(open3d_object=lineset)
        assert isinstance(region, LineSegment3D)
        assert np.array_equal(region.vertices, ((0, 0, 0), (1, 1, 1)))


class TestAxisOrientedCuboid:

    def test_init(self):
        region = AxisOrientedCuboid((1, 1, 1), 9, 19, 29)
        assert (-1, 1, 1) not in region
        assert (2, 2, 2) in region
        assert isinstance(region, Region)
        assert isinstance(region, Region3D)
        assert repr(region) == "AxisOrientedCuboid((1, 1, 1), 9, 19, 29)"
        assert str(region) == "AxisOrientedCuboid((1, 1, 1), 9, 19, 29)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, AxisOrientedCuboid)
        with pytest.raises(AttributeError):
            region.corner = None
            region.length = None
            region.width = None
            region.height = None

    def test_attributes(self):
        region = AxisOrientedCuboid((1, 1, 1), 9, 19, 29)
        assert region.dimension == 3
        assert region.bounds == pytest.approx((1, 1, 1, 10, 20, 30))
        assert np.array_equal(region.intervals, ((1, 10), (1, 20), (1, 30)))
        assert region.extent == pytest.approx((9, 19, 29))
        assert len(region.points) == 8
        assert len(region.vertices) == 8
        assert np.array_equal(region.centroid, (5.5, 10.5, 15.5))
        assert region.max_distance == np.sqrt(9**2 + 19**2 + 29**2)
        assert region.elongation == pytest.approx(1 - 9 / 29)
        assert region.region_measure == (9 * 19 * 29)
        assert region.subregion_measure == 2 * (9 * 19 + 19 * 29 + 29 * 9)
        assert region.radial_distance == pytest.approx(17.909494688572316)
        assert region.isoperimetric_quotient == pytest.approx(0.3660075545899819)

        assert np.array_equal(region.bounding_box.corner, (1, 1, 1))
        assert region.bounding_box.length == pytest.approx(9)
        assert region.bounding_box.width == pytest.approx(19)
        assert region.bounding_box.height == pytest.approx(29)

    @pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
    def test_attributes_open3d(self):
        region = AxisOrientedCuboid((1, 1, 1), 9, 19, 29)
        assert isinstance(region.open3d_object, o3d.t.geometry.AxisAlignedBoundingBox)
        assert region.region_measure == region.open3d_object.volume()

    def test_methods(self):
        region = AxisOrientedCuboid((1, 1, 1), 9, 19, 29)

        assert np.array_equal(
            region.contains(
                [[0, 0, 0], [1, 10, 10], [10, 10, 10], [5, 100, 10], [5, 10, 100]]
            ),
            (1,),
        )

        # other = Rectangle((0, 0), 2, 1, 0)
        # assert isinstance(region.intersection(other), Polygon)
        # assert isinstance(region.symmetric_difference(other), MultiPolygon)
        # assert isinstance(region.union(other), Polygon)

        assert region.contains([(2, 2, 2)]) == (0,)
        assert region.contains([(0, 0, 0)]).size == 0
        assert region.contains([]).size == 0

        # assert isinstance(region.as_artist(), mPatches.Rectangle)

        assert isinstance(region.buffer(1), AxisOrientedCuboid)
        assert repr(region.buffer(1)) == "AxisOrientedCuboid((0, 0, 0), 11, 21, 31)"

    def test_from_intervals(self):
        region = AxisOrientedCuboid.from_intervals(((1, 10), (1, 20), (1, 30)))
        assert repr(region) == "AxisOrientedCuboid((1, 1, 1), 9, 19, 29)"
        region = AxisOrientedCuboid.from_intervals([(1, 10), (1, 20), (1, 30)])
        assert repr(region) == "AxisOrientedCuboid((1, 1, 1), 9, 19, 29)"
        region = AxisOrientedCuboid.from_intervals(
            np.array([(1, 10), (1, 20), (1, 30)])
        )
        assert repr(region) == "AxisOrientedCuboid((1, 1, 1), 9, 19, 29)"

    @pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
    def test_from_open3d(self):
        open3d_object = o3d.t.geometry.AxisAlignedBoundingBox(
            min_bound=o3d.core.Tensor([1.0, 1.0, 1.0]),
            max_bound=o3d.core.Tensor([9.0, 19.0, 29.0]),
        )
        region = AxisOrientedCuboid.from_open3d(open3d_object=open3d_object)
        assert repr(region) == "AxisOrientedCuboid((1.0, 1.0, 1.0), 8.0, 18.0, 28.0)"


class TestCuboid:

    def test_init(self):
        region = Cuboid((1, 1, 1), 9, 19, 29, 45, 45, 45)
        assert isinstance(region, Region)
        assert isinstance(region, Region3D)
        assert repr(region) == "Cuboid((1, 1, 1), 9, 19, 29, 45, 45, 45)"
        assert str(region) == "Cuboid((1, 1, 1), 9, 19, 29, 45, 45, 45)"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, Cuboid)
        for attr in ["corner", "length"]:
            with pytest.raises(AttributeError):
                setattr(region, attr, None)

    def test_attributes(self):
        region = Cuboid((1, 1, 1), 9, 19, 29, 10.0, 20.0, 30.0)
        assert region.dimension == 3
        assert region.bounds == pytest.approx(
            (
                -8.24110838,
                -3.52826625,
                -0.80452893,
                17.9287348,
                21.73866441,
                33.93353878,
            )
        )
        assert region.intervals.shape == (3, 2)
        g, b, a = region.rotation.as_euler("zyx", degrees=True)
        assert region.alpha == pytest.approx(a)
        assert region.beta == pytest.approx(b)
        assert region.gamma == pytest.approx(g)
        assert region.bounds == pytest.approx(
            (
                -8.24110838,
                -3.52826625,
                -0.80452893,
                17.9287348,
                21.73866441,
                33.93353878,
            )
        )
        assert region.extent == pytest.approx((26.16984319, 25.26693065, 34.73806771))
        assert len(region.points) == 8
        assert len(region.vertices) == 8
        assert region.centroid == pytest.approx((4.84381321, 9.10519908, 16.56450492))
        assert region.max_distance == np.sqrt(9**2 + 19**2 + 29**2)
        assert region.elongation == pytest.approx(1 - 9 / 29)
        assert region.region_measure == (9 * 19 * 29)
        assert region.subregion_measure == 2 * (9 * 19 + 19 * 29 + 29 * 9)
        assert region.radial_distance == pytest.approx(17.909494688572316)
        assert region.isoperimetric_quotient == pytest.approx(0.3660075545899819)

        assert isinstance(region.bounding_box, AxisOrientedCuboid)
        assert region.bounding_box.corner == pytest.approx(
            (-8.24110838, -3.52826625, -0.80452893)
        )
        assert region.bounding_box.length == pytest.approx(26.169843186054884)
        assert region.bounding_box.width == pytest.approx(25.26693065443495)
        assert region.bounding_box.height == pytest.approx(34.738067706223326)

    @pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
    def test_methods(self):
        region = Cuboid((1, 1, 1), 9, 19, 29, 45, 45, 45)
        indices_in = region.contains(
            [
                [1.001, 1.001, 1.001],
                [1, 10, 10],
                [10, 10, 10],
                [5, 100, 10],
                [5, 10, 100],
            ]
        )
        assert np.array_equal(indices_in, (0,))

        assert region.contains([(2, 2, 2)]) == (0,)
        assert region.contains([(0, 0, 0)]).size == 0
        assert region.contains([]).size == 0

        # other = Rectangle((0, 0), 2, 1, 0)
        # assert isinstance(region.intersection(other), Polygon)
        # assert isinstance(region.symmetric_difference(other), MultiPolygon)
        # assert isinstance(region.union(other), Polygon)

        # assert isinstance(region.as_artist(), mPatches.Rectangle)

        with pytest.raises(NotImplementedError):
            region.buffer(1)
        # assert isinstance(region.buffer(1), Cuboid)
        # assert repr(region.buffer(1)) == "Cuboid((0, 0, 0), 11, 21, 31, 45, 45, 45))"

    @pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
    def test_attributes_open3d(self):
        region = Cuboid((1, 1, 1), 9, 19, 29, 45, 45, 45)
        assert isinstance(region.open3d_object, o3d.t.geometry.OrientedBoundingBox)
        assert len(region.bounds) == 6
        assert region.open3d_object.extent.numpy() == pytest.approx(
            [region.length, region.width, region.height]
        )
        assert region.rotation.as_matrix() == pytest.approx(
            region.open3d_object.rotation.numpy()
        )
        assert region.region_measure == pytest.approx(region.open3d_object.volume())

    @pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
    def test_from_open3d(self):
        # preparation
        corner = o3d.core.Tensor([1.0, 1.0, 1.0])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
            (np.pi / 4, 0, np.pi / 8)
        )
        aabb = o3d.t.geometry.AxisAlignedBoundingBox(
            min_bound=corner, max_bound=o3d.core.Tensor([9.0, 19.0, 29.0])
        )
        obb = o3d.t.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
            aabb
        )
        open3d_object = obb.rotate(rotation=rotation_matrix, center=corner)

        # tests
        region = Cuboid.from_open3d(open3d_object=open3d_object)
        assert isinstance(region, Cuboid)
        assert region.corner == pytest.approx((1.0, 1.0, 1.0))

        assert region.length == pytest.approx(8)
        assert region.width == pytest.approx(18)
        assert region.height == pytest.approx(28)

        assert region.alpha == pytest.approx(180 / 4)
        assert region.beta == pytest.approx(0)
        assert region.gamma == pytest.approx(180 / 8)


class TestAxisOrientedHypercuboid:

    def test_init(self):
        region = AxisOrientedHypercuboid((1, 1, 1), (9, 19, 29))
        assert (-1, 1, 1) not in region
        assert (2, 2, 2) in region
        assert isinstance(region, Region)
        assert repr(region) == "AxisOrientedHypercuboid((1, 1, 1), (9, 19, 29))"
        assert str(region) == "AxisOrientedHypercuboid((1, 1, 1), (9, 19, 29))"
        new_reg = eval(repr(region))
        assert isinstance(new_reg, AxisOrientedHypercuboid)
        with pytest.raises(AttributeError):
            region.corner = None
            region.length = None
            region.width = None
            region.height = None

    def test_attributes(self):
        region = AxisOrientedHypercuboid((1, 1, 1), (9, 19, 29))
        assert region.dimension == 3
        assert region.bounds == pytest.approx((1, 1, 1, 10, 20, 30))
        assert np.array_equal(region.intervals, ((1, 10), (1, 20), (1, 30)))
        assert region.extent == pytest.approx((9, 19, 29))
        assert len(region.points) == 8
        assert len(region.vertices) == 8
        assert region.centroid.tolist() == [5.5, 10.5, 15.5]
        assert region.max_distance == np.sqrt(9**2 + 19**2 + 29**2)
        assert region.elongation == pytest.approx(1 - 9 / 29)
        assert region.region_measure == (9 * 19 * 29)
        with pytest.raises(NotImplementedError):
            region.subregion_measure  # noqa B018
        assert region.radial_distance == pytest.approx(17.909494688572316)

    def test_methods(self):
        region = AxisOrientedHypercuboid((1, 1, 1), (9, 19, 29))
        assert np.array_equal(
            region.contains(
                [[0, 0, 0], [1, 10, 10], [10, 10, 10], [5, 100, 10], [5, 10, 100]]
            ),
            (1,),
        )
        # needs to be implemented:
        # other = Rectangle((0, 0), 2, 1, 0)
        # assert isinstance(region.intersection(other), Polygon)
        # assert isinstance(region.symmetric_difference(other), MultiPolygon)
        # assert isinstance(region.union(other), Polygon)

        assert region.contains([(2, 2, 2)]) == (0,)
        assert region.contains([(0, 0, 0)]).size == 0
        assert region.contains([]).size == 0
        # assert isinstance(region.as_artist(), mPatches.Rectangle)
        assert isinstance(region.buffer(1), AxisOrientedHypercuboid)
        assert (
            repr(region.buffer(1)) == "AxisOrientedHypercuboid((0, 0, 0), (11, 21, 31))"
        )
        assert region.bounding_box.corner.tolist() == [1, 1, 1]
        assert region.bounding_box.lengths.tolist() == [9, 19, 29]

        region = AxisOrientedHypercuboid.from_intervals(((1, 10), (1, 20), (1, 30)))
        assert repr(region) == "AxisOrientedHypercuboid((1, 1, 1), (9, 19, 29))"
        region = AxisOrientedHypercuboid.from_intervals([(1, 10), (1, 20), (1, 30)])
        assert repr(region) == "AxisOrientedHypercuboid((1, 1, 1), (9, 19, 29))"
        region = AxisOrientedHypercuboid.from_intervals(
            np.array([(1, 10), (1, 20), (1, 30)])
        )
        assert repr(region) == "AxisOrientedHypercuboid((1, 1, 1), (9, 19, 29))"


def test_get_region_from_shapely():
    points = ((2, 2), (2, 3))
    shapely_object = shLine(points)
    region = get_region_from_shapely(shapely_object)
    assert isinstance(region, LineSegment2D)

    points = ((2, 2), (2, 3), (3, 3), (3, 2.5), (2, 2))
    shapely_object = shPolygon(points)
    region = get_region_from_shapely(shapely_object)
    assert isinstance(region, Polygon)

    points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
    holes = [
        ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
        ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
    ]
    shapely_object = shMultiPolygon([shapely_object, shPolygon(points, holes)])
    region = get_region_from_shapely(shapely_object)
    assert isinstance(region, MultiPolygon)


@pytest.mark.skipif(not HAS_DEPENDENCY["open3d"], reason="Test requires open3d.")
def test_get_region_from_open3d():
    open3d_object = o3d.t.geometry.LineSet()
    open3d_object.point.positions = o3d.core.Tensor([[0, 0, 0], [1, 1, 1]])
    open3d_object.line.indices = o3d.core.Tensor([[0, 1]])

    region = get_region_from_open3d(open3d_object)
    assert isinstance(region, LineSegment3D)
    assert repr(region) == "LineSegment3D([[0, 0, 0], [1, 1, 1]], True)"


def test_pickling_Region():
    regions = [
        AxisOrientedCuboid(),
        AxisOrientedHypercuboid(),
        AxisOrientedRectangle(),
        Cuboid(),
        Ellipse(),
        EmptyRegion(),
        Interval(),
        LineSegment2D(points=((0, 0), (1, 1))),
        LineSegment3D(points=((0, 0, 0), (1, 1, 1))),
        MultiPolygon(
            [Polygon(((10, 10), (10, 11), (12, 12))), Polygon(((0, 0), (0, 1), (2, 2)))]
        ),
        Rectangle(),
        Polygon(((0, 0), (0, 1), (2, 2))),
    ]
    with tempfile.TemporaryDirectory() as tmp_directory:
        file_path = Path(tmp_directory) / "pickled_region.pickle"
        for region in regions:
            with open(file_path, "wb") as file:
                pickle.dump(region, file, pickle.HIGHEST_PROTOCOL)
            with open(file_path, "rb") as file:
                region_pickled = pickle.load(file)  # noqa S301
            assert type(region_pickled) is type(region)


def test__polygon_path():
    points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
    holes = [
        ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
        ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
    ]
    shapely_polygon = shPolygon(points, holes)
    path = _polygon_path(polygon=shapely_polygon)
    assert isinstance(path, mplPath)

    polygon = Polygon(((0, 0), (0, 1), (2, 2)))
    path = _polygon_path(polygon=polygon)
    assert isinstance(path, mplPath)
