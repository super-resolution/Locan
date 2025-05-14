import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.spatial
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon

from locan import BoundingBox, ConvexHull, OrientedBoundingBox, Rectangle
from locan.data.hulls.hull import (
    _ConvexHullScipy,
    _ConvexHullShapely,
    _OrientedBoundingBoxOpen3D,
    _OrientedBoundingBoxShapely,
)
from locan.dependencies import HAS_DEPENDENCY


class TestBoundingBox:

    def test_BoundingBox_2d(self, locdata_2d):
        true_hull = np.array([[1, 1], [5, 6]])
        hull = BoundingBox(locdata_2d.coordinates)
        assert hull.width.shape == (2,)
        assert locdata_2d.properties["region_measure_bb"] == hull.region_measure == 20
        assert hull.subregion_measure == 18
        np.testing.assert_array_equal(hull.hull, true_hull)
        assert hull.region.region_measure == 20

    def test_BoundingBox_3d(self, locdata_3d):
        true_hull = np.array([[1, 1, 1], [5, 6, 5]])
        hull = BoundingBox(locdata_3d.coordinates)
        assert hull.width.shape == (3,)
        assert hull.region_measure == 80
        assert hull.subregion_measure == 26
        np.testing.assert_array_equal(hull.hull, true_hull)
        assert hull.region.region_measure == 80

    @pytest.mark.parametrize(
        "fixture_name, expected",
        [
            ("locdata_empty", 0),
            ("locdata_single_localization", 0),
            ("locdata_non_standard_index", 20),
        ],
    )
    def test_BoundingBox_special(
        self,
        locdata_empty,
        locdata_single_localization,
        locdata_non_standard_index,
        fixture_name,
        expected,
    ):
        locdata = eval(fixture_name)
        hull = BoundingBox(locdata.coordinates)
        assert hull.region_measure == expected


class TestConvexHullScipy:

    def test_ConvexHullScipy_2d(self, locdata_2d):
        true_convex_hull_indices = np.array([5, 3, 1, 0, 4])
        hull = _ConvexHullScipy(locdata_2d.coordinates)
        assert np.array_equal(hull.vertex_indices, true_convex_hull_indices)
        assert np.array_equal(
            hull.vertices, locdata_2d.coordinates[true_convex_hull_indices]
        )
        assert hull.points_on_boundary == 5
        assert hull.region_measure == 14
        assert hull.region.region_measure == 14

    def test_ConvexHullScipy_3d(self, locdata_3d):
        true_convex_hull_indices = np.array([0, 1, 2, 3, 4, 5])
        hull = _ConvexHullScipy(locdata_3d.coordinates)
        assert np.array_equal(hull.vertex_indices, true_convex_hull_indices)
        assert np.array_equal(
            hull.vertices, locdata_3d.coordinates[true_convex_hull_indices]
        )
        assert hull.points_on_boundary == 6
        assert hull.region_measure == pytest.approx(19.833333333333332)
        with pytest.raises(NotImplementedError):
            assert hull.region.region_measure == pytest.approx(19.833333333333332)


class TestConvexHullShapely:

    def test_ConvexHullShapely_1d(self, locdata_1d):
        with pytest.raises(TypeError):
            _ConvexHullShapely(locdata_1d.coordinates)

    def test_ConvexHullShapely_2d(self, locdata_2d):
        true_convex_hull_indices = np.array([0, 1, 3, 5, 4])
        hull = _ConvexHullShapely(locdata_2d.coordinates)
        # assert np.array_equal(hull.vertex_indices, true_convex_hull_indices)
        assert np.array_equal(
            hull.vertices, locdata_2d.coordinates[true_convex_hull_indices]
        )
        assert hull.points_on_boundary == 5
        assert hull.region_measure == 14
        assert hull.region.region_measure == 14

    def test_ConvexHullShapely_3d(self, locdata_3d):
        with pytest.raises(TypeError):
            _ConvexHullShapely(locdata_3d.coordinates)


class TestConvexHull:

    def test_ConvexHull_2d(self, locdata_2d):
        hull = ConvexHull(locdata_2d.coordinates, method="scipy")
        assert isinstance(hull.hull, scipy.spatial._qhull.ConvexHull)

        hull = ConvexHull(locdata_2d.coordinates, method="shapely")
        assert isinstance(hull.hull, Polygon)

    def test_ConvexHull_3d(self, locdata_3d):
        hull = ConvexHull(locdata_3d.coordinates, method="scipy")
        assert isinstance(hull.hull, scipy.spatial._qhull.ConvexHull)

        with pytest.raises(TypeError):
            ConvexHull(locdata_3d.coordinates, method="shapely")

    @pytest.mark.parametrize(
        "fixture_name, expected",
        [("locdata_empty", 0), ("locdata_single_localization", 0)],
    )
    def test_ConvexHull(
        self, locdata_empty, locdata_single_localization, fixture_name, expected
    ):
        locdata = eval(fixture_name)
        hull = ConvexHull(locdata.coordinates)
        assert hull.hull is None
        assert hull.region_measure == 0


class TestOrientedBoundingBoxShapely:

    def test_OrientedBoundingBoxShapely_2d_points_0(self):
        points = np.array([[0, 0], [0.3, 0.6], [0, 2], [1, 2], [1, 0], [0, 0]])
        obb = _OrientedBoundingBoxShapely(points)
        assert round(obb.angle) in [0, 90, 180, -90]
        assert len(obb.vertices) == 5
        assert np.isclose(obb.width[0], 1) or np.isclose(obb.width[0], 2)
        assert np.isclose(obb.width[1], 1) or np.isclose(obb.width[1], 2)
        assert np.isclose(obb.region_measure, 2)
        assert np.isclose(obb.subregion_measure, 6)
        assert isinstance(obb.region, Rectangle)
        assert np.isclose(obb.region_measure, obb.region.region_measure)

    def test_OrientedBoundingBoxShapely_2d_points_rotated(self):
        points = np.array(
            [[0.2, 0.2], [0.3, 0.6], [0, 2], [1, 2], [1, 0], [0.2, 0.2]]
        ) + (1, 2)
        for angle in np.linspace(0, 180, 5):
            rotated_points = affinity.rotate(
                MultiPoint(points), angle, origin=[0, 0], use_radians=False
            )
            rotated_points = np.array([rpts.coords[0] for rpts in rotated_points.geoms])
            obb = _OrientedBoundingBoxShapely(rotated_points)
            assert round(obb.angle) in [0, 45, 90, 135, 180, -135, -90, -45]
            assert len(obb.vertices) == 5
            assert np.isclose(obb.width[0], 1) or np.isclose(obb.width[0], 2)
            assert np.isclose(obb.width[1], 1) or np.isclose(obb.width[1], 2)
            assert np.isclose(obb.region_measure, 2)
            assert np.isclose(obb.subregion_measure, 6)
            assert isinstance(obb.region, Rectangle)
            assert np.isclose(obb.region_measure, obb.region.region_measure)

    def test__OrientedBoundingBoxShapely_2d(self, locdata_2d):
        hull = _OrientedBoundingBoxShapely(locdata_2d.coordinates)
        assert hull.width.shape == (2,)
        assert locdata_2d.properties["region_measure_bb"] == hull.region_measure == 20
        assert hull.subregion_measure == 18
        assert hull.region.region_measure == 20

    def test__OrientedBoundingBoxShapely_3d(self, locdata_3d):
        with pytest.raises(TypeError):
            _OrientedBoundingBoxShapely(locdata_3d.coordinates)

    @pytest.mark.visual
    def test__OrientedBoundingBoxShapely_2d_visual_2(self):
        points = np.array(
            [[0.2, 0.2], [0.3, 0.6], [0, 2], [1, 2], [1, 0], [0.2, 0.2]]
        ) + (1, 2)

        _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        for angle in np.linspace(0, 180, 5):
            rotated_points = affinity.rotate(
                MultiPoint(points), angle, origin=[0, 0], use_radians=False
            )
            rotated_points = np.array([rpts.coords[0] for rpts in rotated_points.geoms])
            ax.plot(*rotated_points.T)

            obb = _OrientedBoundingBoxShapely(rotated_points)
            print(f"{obb.region =}")
            print(f"{obb.region.points =}")
            print(f"{obb.vertices =}")
            print(f"{obb.width =}")
            print(f"{angle = }", f" {obb.angle = }")
            ax.plot(*obb.vertices.T, c="blue", marker="+", alpha=0.5)
            ax.add_patch(obb.region.as_artist())
        ax.set(xlim=(-5, 5), ylim=(-5, 5))
        plt.show()


class TestOrientedBoundingBoxOpen3d:

    @pytest.mark.skipif(
        not HAS_DEPENDENCY["open3d"], reason="requires optional package"
    )
    def test__OrientedBoundingBoxOpen3d_2d(self, locdata_2d):
        with pytest.raises(TypeError):
            _OrientedBoundingBoxOpen3D(locdata_2d.coordinates)

    @pytest.mark.skipif(
        not HAS_DEPENDENCY["open3d"], reason="requires optional package"
    )
    def test__OrientedBoundingBoxOpen3d_3d(self, locdata_3d):
        hull = _OrientedBoundingBoxOpen3D(locdata_3d.coordinates)
        assert hull.width.shape == (3,)
        assert hull.region_measure == pytest.approx(82.66963195800781)
        assert hull.subregion_measure == pytest.approx(116.78721)
        assert hull.elongation == pytest.approx(0.40680307)
        assert hull.vertices.shape == (8, 3)
        assert hull.region.region_measure == hull.region_measure


class TestOrientedBoundingBox:

    def test_OrientedBoundingBox_2d_shapely(self, locdata_2d):
        hull = OrientedBoundingBox(locdata_2d.coordinates, method="shapely")
        assert isinstance(hull.hull, Polygon)
        assert hull.width.shape == (2,)
        assert locdata_2d.properties["region_measure_bb"] == hull.region_measure == 20
        assert hull.subregion_measure == 18
        assert hull.region.region_measure == 20

    @pytest.mark.skipif(
        not HAS_DEPENDENCY["open3d"], reason="requires optional package"
    )
    def test_OrientedBoundingBox_2d_open3d(self, locdata_2d):
        with pytest.raises(TypeError):
            OrientedBoundingBox(locdata_2d.coordinates, method="open3d")

    def test_OrientedBoundingBox_3d_shapely(self, locdata_3d):
        with pytest.raises(TypeError):
            OrientedBoundingBox(locdata_3d.coordinates, method="shapely")

    @pytest.mark.skipif(
        not HAS_DEPENDENCY["open3d"], reason="requires optional package"
    )
    def test_OrientedBoundingBox_3d_open3d(self, locdata_3d):
        hull = OrientedBoundingBox(locdata_3d.coordinates, method="open3d")
        assert hull.region_measure == pytest.approx(82.66963195800781)
        assert hull.hull.volume() == hull.region_measure

        assert hull.width.shape == (3,)
        assert hull.region_measure == pytest.approx(82.66963195800781)
        assert hull.subregion_measure == pytest.approx(116.78721)
        assert hull.elongation == pytest.approx(0.40680307)
        assert hull.vertices.shape == (8, 3)
        assert hull.region.region_measure == hull.region_measure

    @pytest.mark.parametrize(
        "fixture_name, expected",
        [("locdata_empty", 0), ("locdata_single_localization", 0)],
    )
    def test_OrientedBoundingBox(
        self, locdata_empty, locdata_single_localization, fixture_name, expected
    ):
        locdata = eval(fixture_name)
        with pytest.raises(TypeError):
            OrientedBoundingBox(locdata.coordinates)
