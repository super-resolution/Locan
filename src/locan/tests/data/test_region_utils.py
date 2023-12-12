import matplotlib.pyplot as plt  # needed for visual inspection
import numpy as np
import pytest

from locan import (  # needed for visual inspection  # noqa: F401
    Interval,
    MultiPolygon,
    Polygon,
    Rectangle,
    Region,
    RoiRegion,
    expand_region,
    regions_union,
    render_2d_mpl,
    scatter_2d_mpl,
    surrounding_region,
)


def test_regions_union_Rectangles():
    region_0 = Rectangle((0, 0), 1, 1, 0)
    region_1 = Rectangle((2, 2), 1, 1, 0)
    regions = [region_0, region_1]
    union = regions_union(regions)
    assert isinstance(union, MultiPolygon)
    assert union.region_measure == 2

    region_0 = Rectangle((0, 0), 1, 1, 0)
    region_1 = Rectangle((1, 1), 1, 1, 0)
    region_2 = Rectangle((0, 0.5), 1, 1, 0)
    regions = [region_0, region_1, region_2]
    union = regions_union(regions)
    assert isinstance(union, Polygon)
    assert union.region_measure == 2.5

    region_0 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2.5), (2, 2)])
    region_1 = Rectangle((10, 10), 1, 1, 0)
    regions = [region_0, region_1]
    union = regions_union(regions)
    assert isinstance(union, MultiPolygon)
    assert union.region_measure == 1.75

    # visualize
    fig, ax = plt.subplots()
    ax.scatter(*region_0.points.T)
    ax.add_patch(region_0.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(region_1.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(region_2.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(union.as_artist(fill=True, alpha=0.2, color="Red"))
    # plt.show()

    plt.close("all")


def test_extend_Interval():
    region = Interval(0, 1)
    expanded_region = expand_region(region, distance=1)
    assert expanded_region.region_measure == pytest.approx(3)
    assert np.array_equal(expanded_region.bounds, (-1, 2))

    with pytest.raises(NotImplementedError):
        support = Interval(-1, 2)
        expanded_region_with_support = expand_region(
            region, distance=2, support=support
        )
        # the following test are not reached.
        # They should pass when intersection has been implemented.
        assert isinstance(expanded_region_with_support, Region)
        assert expanded_region_with_support.region_measure == 3
        assert np.array_equal(expanded_region_with_support.bounds, (-1, 2))


def test_extend_Rectangle():
    region = Rectangle((0, 0), 1, 1, 0)
    extended_region = expand_region(region, distance=1)
    assert extended_region.region_measure == pytest.approx(8.13654849054594)

    region = Rectangle((0, 0), 2, 1, 0)
    support = Rectangle((0, 0), 3, 2, 0)
    extended_region = expand_region(region, distance=2)
    extended_region_with_support = expand_region(region, distance=2, support=support)
    assert isinstance(extended_region, Region)
    assert isinstance(extended_region_with_support, Region)
    assert region.region_measure == 2
    assert extended_region.region_measure == pytest.approx(26.546193962183754)
    assert extended_region_with_support.region_measure == 6

    # visualize
    fig, ax = plt.subplots()
    ax.scatter(*region.points.T)
    ax.add_patch(region.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(extended_region.as_artist(fill=True, alpha=0.2, color="Red"))
    ax.add_patch(
        extended_region_with_support.as_artist(fill=True, alpha=0.2, color="Green")
    )
    # plt.show()

    plt.close("all")


def test_extend_Rectangles():
    region_0 = Rectangle((0, 0), 1, 1, 0)
    region_1 = Rectangle((2, 2), 1, 1, 0)
    region = regions_union([region_0, region_1])
    extended_region = expand_region(region, distance=0.1)
    assert isinstance(extended_region, Region)
    assert extended_region.region_measure == pytest.approx(2.8627309698109196)

    region_0 = Rectangle((0, 0), 1, 1, 0)
    region_1 = Rectangle((1, 1), 1, 1, 0)
    region_2 = Rectangle((0, 0.5), 1, 1, 0)
    region = regions_union([region_0, region_1, region_2])
    extended_region = expand_region(region, distance=2)
    assert isinstance(extended_region, Region)
    assert region_0.region_measure == 1
    assert extended_region.region_measure == pytest.approx(29.766960686847877)

    # visualize
    fig, ax = plt.subplots()
    ax.scatter(*region_0.points.T)
    ax.add_patch(region_0.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(region_1.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(region_2.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(extended_region.as_artist(fill=True, alpha=0.2, color="Red"))
    # plt.show()

    plt.close("all")


def test_extend_Polygon():
    points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
    holes = [
        ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
        ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
    ]
    region = Polygon(points, holes)
    support = Rectangle((0, 0), 1, 1, 0)
    extended_region = expand_region(region, distance=0.1)
    extended_region_with_support = expand_region(region, distance=0.1, support=support)
    assert isinstance(extended_region, Region)
    assert isinstance(extended_region_with_support, Region)
    assert region.region_measure == pytest.approx(0.7350000000000001)
    assert extended_region.region_measure == pytest.approx(1.1431688585174895)
    assert extended_region_with_support.region_measure == pytest.approx(
        0.8493033988749895
    )

    # visualize
    fig, ax = plt.subplots()
    ax.scatter(*region.points.T)
    ax.add_patch(region.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(extended_region.as_artist(fill=True, alpha=0.2, color="Red"))
    ax.add_patch(
        extended_region_with_support.as_artist(fill=True, alpha=0.2, color="Green")
    )
    # plt.show()

    plt.close("all")


def test_extend_MultiPolygon():
    points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
    holes = [
        ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
        ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
    ]
    region = Polygon(points, holes)

    support = Rectangle((0, 0), 1, 1, 0)
    extended_region = expand_region(region, distance=0.1)
    extended_region_with_support = expand_region(region, distance=0.1, support=support)
    assert isinstance(extended_region, Region)
    assert isinstance(extended_region_with_support, Region)
    assert region.region_measure == pytest.approx(0.7350000000000001)
    assert extended_region.region_measure == pytest.approx(1.1431688585174895)
    assert extended_region_with_support.region_measure == pytest.approx(
        0.8493033988749895
    )

    # visualize
    fig, ax = plt.subplots()
    ax.scatter(*region.points.T)
    ax.add_patch(region.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(extended_region.as_artist(fill=True, alpha=0.2, color="Red"))
    ax.add_patch(
        extended_region_with_support.as_artist(fill=True, alpha=0.2, color="Green")
    )
    # plt.show()

    plt.close("all")


def test_surrounding_region_Rectangles():
    region_0 = Rectangle((0, 0), 1, 1, 0)
    region_1 = Rectangle((1, 1), 1, 1, 0)
    region_2 = Rectangle((0, 0.5), 1, 1, 0)
    region = regions_union([region_0, region_1, region_2])
    extended_region = surrounding_region(region, distance=2)
    assert isinstance(extended_region, Region)
    assert region_0.region_measure == 1
    assert extended_region.region_measure == pytest.approx(27.266960686847888)

    # visualize
    fig, ax = plt.subplots()
    ax.scatter(*region_0.points.T)
    ax.add_patch(region_0.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(region_1.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(region_2.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(extended_region.as_artist(fill=True, alpha=0.2, color="Red"))
    # plt.show()

    plt.close("all")


def test_surrounding_region_Polygon():
    points = ((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
    holes = [
        ((0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.25)),
        ((0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.45)),
    ]
    region = Polygon(points, holes)
    support = Rectangle((0, 0), 1, 1, 0)
    extended_region = surrounding_region(region, distance=0.1)
    extended_region_with_support = surrounding_region(
        region, distance=0.1, support=support
    )
    assert isinstance(extended_region, Region)
    assert isinstance(extended_region_with_support, Region)
    assert region.region_measure == pytest.approx(0.7350000000000001)
    assert extended_region.region_measure == pytest.approx(0.4081688585174893)
    assert extended_region_with_support.region_measure == pytest.approx(
        0.11430339887498947
    )

    # visualize
    fig, ax = plt.subplots()
    ax.scatter(*region.points.T)
    ax.add_patch(region.as_artist(fill=True, alpha=0.2, color="Blue"))
    ax.add_patch(extended_region.as_artist(fill=True, alpha=0.2, color="Red"))
    ax.add_patch(
        extended_region_with_support.as_artist(fill=True, alpha=0.2, color="Green")
    )
    # plt.show()

    plt.close("all")


def test_extend_region_RoiRegion():
    rr = RoiRegion(region_type="rectangle", region_specs=((0, 0), 1, 1, 0))
    er = expand_region(region=rr, distance=1, support=None)
    assert rr.region_measure == pytest.approx(1)

    # visualize
    fig, ax = plt.subplots()
    ax.scatter(*rr.points.T)
    ax.add_patch(rr.as_artist(fill=True))
    ax.scatter(np.array(er.exterior.coords)[:, 0], np.array(er.exterior.coords)[:, 1])
    # plt.show()

    assert er.area == pytest.approx(8.13654849054594)
    er = expand_region(region=rr.to_shapely(), distance=1, support=None)
    assert er.area == pytest.approx(8.13654849054594)

    rr = RoiRegion(region_type="ellipse", region_specs=((0, 0), 1, 3, 0))
    sr = expand_region(region=rr, distance=1, support=None)
    assert sr.area == pytest.approx(12.169350575721547)

    rr = RoiRegion(
        region_type="polygon", region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
    )
    sr = expand_region(region=rr, distance=1, support=None)
    assert sr.area == pytest.approx(7.50457995299987)

    rr = RoiRegion(
        region_type="polygon", region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
    )
    sup = RoiRegion(region_type="rectangle", region_specs=((0, 0), 2, 2, 0))
    sr = expand_region(region=rr, distance=1, support=sup)
    assert sr.area == pytest.approx(3.762009021216654)

    rr = RoiRegion(region_type="rectangle", region_specs=((0, 0), 1, 1, 0))
    sr = expand_region(region=rr, distance=1, support=None)
    assert sr.area == pytest.approx(8.13654849054594)
    assert isinstance(sr, Polygon)
    rr_1 = RoiRegion(region_type="rectangle", region_specs=((10, 10), 1, 1, 0))
    region = regions_union([rr, rr_1])
    sr = expand_region(region=region, distance=1, support=None)
    assert sr.area == pytest.approx(2 * 8.13654849054594)

    # visualize
    fig, ax = plt.subplots()
    ax.scatter((-1, 3), (-1, 3))
    ax.add_patch(rr.as_artist(fill=True))
    ax.add_patch(sr.as_artist(fill=True))
    # plt.show()

    # visualize
    # fig, ax = plt.subplots()
    # #ax.scatter((-1, 3), (-1, 3))
    # ax.add_patch(rr.as_artist(fill=True))
    # ax.add_patch(rr_1.as_artist(fill=True))
    # sr_region = RoiRegion.from_shapely(region='shapelyMultiPolygon', shapely_obj=sr)
    # for pat in sr_region.as_artist(fill=False):
    #     ax.add_patch(pat)
    # plt.show()

    plt.close("all")


def test_surrounding_region():
    rr = RoiRegion(region_type="rectangle", region_specs=((0, 0), 1, 1, 0))
    sr = surrounding_region(region=rr, distance=1, support=None)
    assert sr.area == pytest.approx(7.136548490545939)

    rr = RoiRegion(region_type="ellipse", region_specs=((0, 0), 1, 3, 0))
    sr = surrounding_region(region=rr, distance=1, support=None)
    assert sr.area == pytest.approx(9.816939207812093)

    rr = RoiRegion(
        region_type="polygon", region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
    )
    sr = surrounding_region(region=rr, distance=1, support=None)
    assert sr.area == pytest.approx(6.754579952999869)

    rr = RoiRegion(
        region_type="polygon", region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0))
    )
    sup = RoiRegion(region_type="rectangle", region_specs=((0, 0), 2, 2, 0))
    sr = surrounding_region(region=rr, distance=1, support=sup)
    assert sr.area == pytest.approx(3.012009021216654)

    rr = RoiRegion(region_type="rectangle", region_specs=((0, 0), 1, 1, 0))
    rr_1 = RoiRegion(region_type="rectangle", region_specs=((0, 0), 1, 1, 0))
    sr = surrounding_region(region=regions_union([rr, rr_1]), distance=1, support=None)
    assert sr.area == pytest.approx(7.136548490545939)
    assert isinstance(sr, Polygon)
    rr_1 = RoiRegion(region_type="rectangle", region_specs=((10, 10), 1, 1, 0))
    sr = surrounding_region(region=regions_union([rr, rr_1]), distance=1, support=None)
    assert sr.area == pytest.approx(2 * 7.136548490545939)
    assert isinstance(sr, MultiPolygon)

    # visualize
    # fig, ax = plt.subplots()
    # #ax.scatter((-1, 3), (-1, 3))
    # ax.add_patch(rr.as_artist(fill=True))
    # ax.add_patch(rr_1.as_artist(fill=True))
    # sr_region = RoiRegion.from_shapely(region='shapelyMultiPolygon', shapely_obj=sr)
    # for pat in sr_region.as_artist(fill=False):
    #     ax.add_patch(pat)
    # plt.show()

    plt.close("all")
