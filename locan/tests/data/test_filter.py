import matplotlib.pyplot as plt  # needed for visual inspection  # noqa: F401
import numpy as np
import pandas as pd
import pytest

import locan.data.metadata_pb2
from locan import (
    HullType,
    LocData,
    Rectangle,
    RoiRegion,
    Selector,
    cluster_dbscan,
    exclude_sparse_points,
    filter_condition,
    localizations_in_cluster_regions,
    random_subset,
    render_2d_mpl,  # needed for visual inspection  # noqa: F401
    scatter_2d_mpl,  # needed for visual inspection  # noqa: F401
    select_by_condition,
    select_by_region,
    transform_affine,
)


@pytest.fixture()
def locdata_simple_():
    locdata_dict = {
        "position_x": [0, 1, 2, 3, 0, 1, 4, 5],
        "position_y": [0, 1, 2, 3, 1, 4, 5, 1],
        "position_z": [0, 1, 2, 3, 4, 4, 4, 5],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        "position_x": [0, 1, 2, 3, 0, 1, 4, 5],
        "position_y": [0, 1, 2, 3, 1, 4, 5, 1],
        "position_z": [0, 1, 2, 3, 4, 4, 4, 5],
    }
    df = pd.DataFrame(locdata_dict)
    df.index = [2, 0, 1, 3, 4, 5, 6, 7]
    meta_ = locan.data.metadata_pb2.Metadata()
    meta_.creation_time.FromSeconds(1)
    return LocData.from_dataframe(dataframe=df, meta=meta_)


class TestSelector:
    def test_selector_init(self, locdata_simple):
        selector = Selector(
            loc_property="position_x",
            activate=False,
            lower_bound=1,
            upper_bound=10,
        )
        assert (
            repr(selector) == "Selector(loc_property='position_x', activate=False, "
            "lower_bound=1, upper_bound=10)"
        )
        assert np.array_equal(selector.interval.bounds, (1, 10))
        assert selector.activate is False
        assert selector.condition == ""

        selector.lower_bound = 2
        selector.activate = True
        assert np.array_equal(selector.interval.bounds, (2, 10))
        assert selector.activate is True
        assert selector.condition == "2 < position_x < 10"

        with pytest.raises(TypeError):
            Selector()


def test_filter_condition():
    selectors = [
        Selector(
            loc_property="position_x", activate=False, lower_bound=0, upper_bound=10
        ),
        Selector(
            loc_property="position_y", activate=True, lower_bound=0, upper_bound=10
        ),
        Selector(
            loc_property="intensity", activate=True, lower_bound=0, upper_bound=10
        ),
    ]
    condition = filter_condition(selectors=selectors)
    assert condition == "0 < position_y < 10 and 0 < intensity < 10"

    selectors[0].activate = True
    condition = filter_condition(selectors=selectors)
    assert (
        condition
        == "0 < position_x < 10 and 0 < position_y < 10 and 0 < intensity < 10"
    )


def test_select_by_condition(locdata_simple):
    dat_s = select_by_condition(locdata_simple, "position_x>1")
    assert len(dat_s) == 4
    assert np.all(dat_s.data.index == [1, 3, 6, 7])
    # dat_s.print_meta()
    # print(dat_s.meta)
    # print(dat_s.data)


def test_LocData_selection_from_collection(locdata_simple):
    # print(locdata_simple.meta)
    sel = []
    for i in range(4):
        sel.append(select_by_condition(locdata_simple, f"position_x>{i}"))
    col = LocData.from_collection(sel)
    assert len(col) == 4
    assert len(col.references) == 4
    # print(col.references[0].meta)
    # print(col.data)
    # print(col.meta)

    col_sel = select_by_condition(col, "localization_count>2")
    assert len(col_sel) == 3
    # print(col_sel.data)
    assert col_sel.references is col

    col_sel_sel = select_by_condition(col_sel, "localization_count<4")
    assert len(col_sel_sel) == 1
    # print(col_sel_sel.data)
    assert col_sel_sel.references is col_sel
    # print(col_sel_sel.meta)


def test_random_subset(locdata_simple):
    dat_s = random_subset(locdata_simple, n_points=3)
    assert len(dat_s) == 3
    # dat_s.print_meta()
    # print(dat_s.data)
    # print(dat_s.meta)


def test_localizations_in_cluster_regions(locdata_blobs_2d):
    coordinates = [(200, 500), (200, 600), (900, 650), (1000, 600)]
    locdata = LocData.from_coordinates(coordinates)
    noise, collection = cluster_dbscan(locdata_blobs_2d, eps=100, min_samples=3)
    # print(collection.data)

    # visualize
    # ax = render_2d_mpl(locdata_blobs_2d)
    # scatter_2d_mpl(collection.references[2], index=False, marker='.', color='g')
    # scatter_2d_mpl(locdata, index=False, marker='o')
    # scatter_2d_mpl(collection)
    # ax.add_patch(collection.references[2].convex_hull.region.as_artist(fill=False))
    # plt.show()

    # print(locdata_blobs_2d.convex_hull.region.contains(coordinates))
    # print(collection.references[2].convex_hull.region.contains(coordinates))

    # collection with references being a list of other LocData objects, e.g. individual clusters
    result = localizations_in_cluster_regions(locdata, collection)
    assert np.array_equal(result.data.localization_count.values, [0, 0, 1, 0, 0])

    result = localizations_in_cluster_regions(
        locdata, collection, hull_type=HullType.BOUNDING_BOX
    )
    assert np.array_equal(result.data.localization_count.values, [0, 0, 1, 0, 1])

    # selection of collection with references being another LocData object
    selected_collection = select_by_condition(
        collection, condition="subregion_measure_bb > 200"
    )
    result = localizations_in_cluster_regions(locdata, selected_collection)
    assert np.array_equal(result.data.localization_count.values, [0, 1, 0, 0])

    # collection being a list of other LocData objects
    result = localizations_in_cluster_regions(locdata, collection.references)
    assert np.array_equal(result.data.localization_count.values, [0, 0, 1, 0, 0])

    plt.close("all")


def test_select_by_region(locdata_2d):
    region = Rectangle((1, 1), 3.5, 4.5, 0)

    # visualize
    # ax = scatter_2d_mpl(locdata_2d, index=True, marker='o', color='g')
    # ax.add_patch(region.as_artist(fill=False))
    # plt.show()

    new_locdata = select_by_region(locdata_2d, region)
    # print(new_locdata.data)
    assert new_locdata.meta.history[-1].name == "select_by_region"
    assert len(new_locdata) == 2

    new_locdata = select_by_region(
        locdata_2d, region, loc_properties=["position_x", "frame"]
    )
    assert len(new_locdata) == 3

    region = RoiRegion(region_type="rectangle", region_specs=((1, 1), 3.5, 4.5, 0))
    new_locdata = select_by_region(
        locdata_2d, region, loc_properties=["position_x", "frame"]
    )
    assert len(new_locdata) == 3

    plt.close("all")


def test_exclude_sparse_points(locdata_simple):
    new_locdata = exclude_sparse_points(locdata=locdata_simple, radius=3, min_samples=2)
    assert len(new_locdata) == 4
    # todo: check if the correct points are taken
    locdata_simple_trans = transform_affine(locdata_simple, offset=(0.5, 0, 0))
    new_locdata = exclude_sparse_points(
        locdata=locdata_simple,
        other_locdata=locdata_simple_trans,
        radius=2,
        min_samples=2,
    )
    assert len(new_locdata) == 3
    # print(new_locdata.meta)
