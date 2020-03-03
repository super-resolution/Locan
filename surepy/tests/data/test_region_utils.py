import numpy as np
import matplotlib.pyplot as plt  # needed for visual inspection

from surepy import LocData, select_by_condition, HullType, RoiRegion
from surepy import surrounding_region, localizations_in_cluster_regions, distance_to_region, \
    distance_to_region_boundary, localizations_in_region
from surepy import cluster_dbscan
from surepy import render_2d_mpl, scatter_2d_mpl  # needed for visual inspection


def test_surrounding_region():
    rr = RoiRegion(region_type='rectangle', region_specs=((0, 0), 1, 1, 0))
    sr = surrounding_region(region=rr, distance=1, support=None)
    assert sr.area == 7.136548490545939

    rr = RoiRegion(region_type='ellipse', region_specs=((0, 0), 1, 3, 0))
    sr = surrounding_region(region=rr, distance=1, support=None)
    assert sr.area == 9.982235792799617

    rr = RoiRegion(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)))
    sr = surrounding_region(region=rr, distance=1, support=None)
    assert sr.area == 6.754579952999869

    rr = RoiRegion(region_type='polygon', region_specs=((0, 0), (0, 1), (1, 1), (1, 0.5), (0, 0)))
    sup = RoiRegion(region_type='rectangle', region_specs=((0, 0), 2, 2, 0))
    sr = surrounding_region(region=rr, distance=1, support=sup)
    assert sr.area == 3.012009021216654


def test_localizations_in_cluster_regions(locdata_blobs_2d):
    coordinates = [(200, 500), (200, 600), (900, 650), (1000, 600)]
    locdata = LocData.from_coordinates(coordinates)
    collection = cluster_dbscan(locdata_blobs_2d, eps=100, min_samples=3)
    # print(collection.data)

    # visualize
    # ax = render_2d_mpl(locdata_blobs_2d)
    # scatter_2d_mpl(collection.references[2], index=False, marker='o', color='g')
    # scatter_2d_mpl(locdata, index=False, marker='o')
    # scatter_2d_mpl(collection)
    # ax.add_patch(collection.references[2].convex_hull.region.as_artist(fill=False))
    # plt.show()

    # print(locdata_blobs_2d.convex_hull.region.contains(coordinates))
    # print(collection.references[2].convex_hull.region.contains(coordinates))

    # collection with references being a list of other LocData objects, e.g. individual clusters
    result = localizations_in_cluster_regions(locdata, collection)
    assert np.array_equal(result.data.localization_count.values, [0, 0, 1, 0, 0])

    result = localizations_in_cluster_regions(locdata, collection, hull_type=HullType.BOUNDING_BOX)
    assert np.array_equal(result.data.localization_count.values, [0, 0, 1, 0, 1])

    # selection of collection with references being another LocData object
    selected_collection = select_by_condition(collection, condition='subregion_measure_bb > 200')
    result = localizations_in_cluster_regions(locdata, selected_collection)
    assert np.array_equal(result.data.localization_count.values, [0, 1, 0, 0])

    # collection being a list of other LocData objects
    result = localizations_in_cluster_regions(locdata, collection.references)
    assert np.array_equal(result.data.localization_count.values, [0, 0, 1, 0, 0])


def test_distance_to_region(locdata_2d):
    region = RoiRegion(region_type='rectangle', region_specs=((1, 1), 3, 3, 0))
    # visualize
    # ax = scatter_2d_mpl(locdata_2d, index=True, marker='o', color='g')
    # ax.add_patch(region.as_artist(fill=False))
    # plt.show()
    distances = distance_to_region(locdata_2d, region)
    assert np.array_equal(distances[:-1], np.array([0, 1, 0, 2, 0]))


def test_distance_to_region_boundary(locdata_2d):
    region = RoiRegion(region_type='rectangle', region_specs=((1, 1), 3, 3, 0))
    # visualize
    # ax = scatter_2d_mpl(locdata_2d, index=True, marker='o', color='g')
    # ax.add_patch(region.as_artist(fill=False))
    # plt.show()
    distances = distance_to_region_boundary(locdata_2d, region)
    assert np.array_equal(distances[:-1], np.array([0, 1, 1, 2, 0]))


def test_localizations_in_region(locdata_2d):
    region = RoiRegion(region_type='rectangle', region_specs=((1, 1), 3.5, 4.5, 0))
    # visualize
    # ax = scatter_2d_mpl(locdata_2d, index=True, marker='o', color='g')
    # ax.add_patch(region.as_artist(fill=False))
    # plt.show()

    new_locdata = localizations_in_region(locdata_2d, region)
    # print(new_locdata.data)
    assert new_locdata.meta.history[-1].name == "localizations_in_region"
    assert len(new_locdata) == 3

    new_locdata = localizations_in_region(locdata_2d, region.to_shapely())
    assert len(new_locdata) == 2

    new_locdata = localizations_in_region(locdata_2d, region, loc_properties=['position_x', 'frame'])
    assert len(new_locdata) == 4
