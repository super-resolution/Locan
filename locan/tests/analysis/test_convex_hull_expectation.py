from copy import deepcopy

import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from locan import LocData
from locan.analysis import ConvexHullExpectation, ConvexHullExpectationBatch
from locan.analysis.convex_hull_expectation import (
    ConvexHullExpectationResource,
    ConvexHullExpectationResults,
    ConvexHullExpectationValues,
    ConvexHullProperty,
    _get_convex_hull_property_expectation,
    _get_resource,
    compute_convex_hull_region_measure_2d,
)


def test_get_resource():
    for convex_hull_property_item in ConvexHullProperty:
        resource_ = ConvexHullExpectationResource[convex_hull_property_item.name]
        result = _get_resource(
            resource_directory="locan.analysis.resources.convex_hull_expectation",
            resource=resource_,
        )
        assert isinstance(result, ConvexHullExpectationValues)


def test_compute_convex_hull_region_measure_2d():
    computed = compute_convex_hull_region_measure_2d(n_points=3)
    from_resource = _get_resource(
        resource_directory="locan.analysis.resources.convex_hull_expectation",
        resource=ConvexHullExpectationResource["REGION_MEASURE_2D"],
    ).expectation[0]
    assert pytest.approx(computed, rel=0.01) == from_resource == 0.87

    computed = compute_convex_hull_region_measure_2d(n_points=100)
    from_resource = _get_resource(
        resource_directory="locan.analysis.resources.convex_hull_expectation",
        resource=ConvexHullExpectationResource["REGION_MEASURE_2D"],
    ).expectation[97]
    assert pytest.approx(computed, rel=0.001) == from_resource

    computed = compute_convex_hull_region_measure_2d(n_points=2, sigma=2)
    assert np.isnan(computed)


def test__get_convex_hull_property_expectation_region_measure_2d():
    result = _get_convex_hull_property_expectation(
        convex_hull_property="region_measure_2d", n_points=10, sigma=2
    )
    assert result == ConvexHullExpectationValues(
        n_points=np.array([10]), expectation=21.52, std_pos=9.16, std_neg=6.72
    )

    result = _get_convex_hull_property_expectation(
        convex_hull_property="region_measure_2d", n_points=300, sigma=2
    )
    assert result.n_points.size == 0

    result = _get_convex_hull_property_expectation(
        convex_hull_property="region_measure_2d", n_points=[3, 10], sigma=2
    )
    assert np.array_equal(result.expectation, np.array([3.48, 21.52]))

    result = _get_convex_hull_property_expectation(
        convex_hull_property="region_measure_2d", n_points=[1, 3, 10], sigma=2
    )
    assert np.array_equal(result.n_points, np.array([3, 10]))
    assert np.array_equal(result.expectation, np.array([3.48, 21.52]))

    result = _get_convex_hull_property_expectation(
        convex_hull_property="region_measure_2d", n_points=np.array([1, 3, 10]), sigma=2
    )
    assert np.array_equal(result.n_points, np.array([3, 10]))
    assert np.array_equal(result.expectation, np.array([3.48, 21.52]))


def test__get_convex_hull_property_expectation_subregion_measure_2d():
    result = _get_convex_hull_property_expectation(
        convex_hull_property="subregion_measure_2d", n_points=10, sigma=2
    )
    assert result == ConvexHullExpectationValues(
        n_points=np.array([10]),
        expectation=np.array([19.34]),
        std_pos=np.array([3.62]),
        std_neg=np.array([3.26]),
    )

    result = _get_convex_hull_property_expectation(
        convex_hull_property="subregion_measure_2d", n_points=[3, 10], sigma=2
    )
    assert np.array_equal(result.expectation, np.array([10.64, 19.34]))


def test__get_convex_hull_property_expectation_region_measure_3d():
    result = _get_convex_hull_property_expectation(
        convex_hull_property="region_measure_3d", n_points=10, sigma=2
    )
    assert result == ConvexHullExpectationValues(
        n_points=np.array([10]),
        expectation=np.array([49.2]),
        std_pos=np.array([28.08]),
        std_neg=np.array([18.24]),
    )

    result = _get_convex_hull_property_expectation(
        convex_hull_property="region_measure_3d", n_points=[4, 10], sigma=2
    )
    assert np.array_equal(result.expectation, np.array([4.24, 49.2]))


def test__get_convex_hull_property_expectation_subregion_measure_3d():
    result = _get_convex_hull_property_expectation(
        convex_hull_property="subregion_measure_3d", n_points=10, sigma=2
    )
    assert result == ConvexHullExpectationValues(
        n_points=np.array([10]), expectation=86.04, std_pos=27.72, std_neg=21.8
    )

    result = _get_convex_hull_property_expectation(
        convex_hull_property="subregion_measure_3d", n_points=[4, 10], sigma=2
    )
    assert np.array_equal(result.expectation, np.array([27.72, 86.04]))


def test_enums():
    for member in ConvexHullProperty:
        assert ConvexHullExpectationResource[member.name]


class TestConvexHullPropertyExpectation:
    def test_init(self, locdata_2d):
        cpe = ConvexHullExpectation(meta={"comment": "this is an example"})
        assert (
            str(cpe) == "ConvexHullExpectation("
            "convex_hull_property=region_measure_ch, "
            "expected_variance=None)"
        )
        assert cpe.results is None
        assert cpe.meta.comment == "this is an example"

    def test_empty_locdata(self, caplog):
        cpe = ConvexHullExpectation().compute(LocData())
        assert cpe.results is None
        cpe.plot()
        cpe.hist()
        # plt.show()

        assert caplog.record_tuples == [
            ("locan.analysis.convex_hull_expectation", 30, "Locdata is empty."),
        ]

        plt.close("all")

    def test_compute(self, locdata_2d):
        locdata_2d = deepcopy(locdata_2d)
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        cpe = ConvexHullExpectation().compute(locdata=collection)
        # print(cpe.results.values)
        # print(cpe.results.grouped)
        assert isinstance(cpe.results, ConvexHullExpectationResults)
        assert cpe.results.values.index.tolist() == collection.data.index.tolist()

        assert cpe.results.values["region_measure_ch"].tolist() == pytest.approx(
            [14, 14], rel=0.1
        )

        assert cpe.results.values.columns.tolist() == [
            "localization_count",
            "region_measure_ch",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert cpe.results.grouped.index.tolist() == [6]
        assert cpe.results.grouped.columns.tolist() == [
            "region_measure_ch_mean",
            "region_measure_ch_std",
            "expectation",
            "expectation_std_pos",
            "expectation_std_neg",
        ]
        assert cpe.results.grouped.expectation.isna().all()

        cpe.plot()
        # plt.show()
        cpe.hist()
        # plt.show()

        plt.close("all")

    def test_compute_with_expectation(self, locdata_2d):
        locdata_2d = deepcopy(locdata_2d)
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        cpe = ConvexHullExpectation(
            convex_hull_property="region_measure_ch", expected_variance=100
        ).compute(locdata=collection)
        # print(cpe.results.values.columns)
        # print(cpe.results.grouped)
        assert cpe.results.values.index.tolist() == collection.data.index.tolist()
        assert cpe.results.values.columns.tolist() == [
            "localization_count",
            "region_measure_ch",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert cpe.results.grouped.index.tolist() == [6]
        assert cpe.results.grouped.columns.tolist() == [
            "region_measure_ch_mean",
            "region_measure_ch_std",
            "expectation",
            "expectation_std_pos",
            "expectation_std_neg",
        ]

        assert cpe.results.grouped.expectation.tolist() == pytest.approx(
            [321.0], rel=0.1
        )
        assert cpe.results.grouped.expectation_std_pos.tolist() == pytest.approx(
            [198.0], rel=0.1
        )
        assert cpe.results.grouped.expectation_std_neg.tolist() == pytest.approx(
            [132], rel=0.1
        )
        assert cpe.results.values.value_to_expectation_ratio.tolist() == pytest.approx(
            [0.0436137, 0.0436137], rel=1e-3
        )

        cpe.plot()
        # plt.show()
        cpe.hist()
        # plt.show()

        plt.close("all")

    def test_compute_with_expectation_unsorted_index(self, locdata_2d):
        locdata = deepcopy(locdata_2d)
        other_locdata = deepcopy(locdata_2d)
        other_locdata.dataframe.loc[:, "position_x"] = (
            other_locdata.dataframe.loc[:, "position_x"] * 2
        )
        other_locdata.update(dataframe=other_locdata.dataframe.iloc[1:])
        collection = LocData.from_collection(locdatas=[locdata, other_locdata])
        collection.data.index = [3, 1]
        cpe = ConvexHullExpectation(
            convex_hull_property="region_measure_ch", expected_variance=100
        ).compute(locdata=collection)
        assert "region_measure_ch" in locdata.properties.keys()
        assert "region_measure_ch" in collection.data.columns
        assert cpe.results.values.index.tolist() == collection.data.index.tolist()
        assert cpe.results.values.columns.tolist() == [
            "localization_count",
            "region_measure_ch",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert cpe.results.grouped.index.tolist() == [5, 6]
        assert cpe.results.grouped.columns.tolist() == [
            "region_measure_ch_mean",
            "region_measure_ch_std",
            "expectation",
            "expectation_std_pos",
            "expectation_std_neg",
        ]

        assert cpe.results.grouped.expectation.tolist() == pytest.approx(
            [250.99999999999997, 321.0], rel=0.1
        )
        assert cpe.results.grouped.expectation_std_pos.tolist() == pytest.approx(
            [183.0, 198.0], rel=0.1
        )
        assert cpe.results.grouped.expectation_std_neg.tolist() == pytest.approx(
            [113.99999999999999, 132.0], rel=0.1
        )
        assert cpe.results.values.value_to_expectation_ratio.tolist() == pytest.approx(
            [0.04361370716510903, 0.0756972111553785], rel=1e-3
        )

        cpe.plot()
        # plt.show()
        cpe.hist()
        # plt.show()

        plt.close("all")

    def test_compute_with_expectation_subregion(self, locdata_2d):
        locdata_2d = deepcopy(locdata_2d)
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        cpe = ConvexHullExpectation(
            convex_hull_property="subregion_measure_ch", expected_variance=100
        ).compute(locdata=collection)
        # print(cpe.results.values.)
        # print(cpe.results.grouped)
        assert cpe.results.values.index.tolist() == collection.data.index.tolist()
        assert cpe.results.values.columns.tolist() == [
            "localization_count",
            "subregion_measure_ch",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert cpe.results.grouped.index.tolist() == [6]
        assert cpe.results.grouped.columns.tolist() == [
            "subregion_measure_ch_mean",
            "subregion_measure_ch_std",
            "expectation",
            "expectation_std_pos",
            "expectation_std_neg",
        ]

        cpe.plot()
        # plt.show()
        cpe.hist()
        # plt.show()

        plt.close("all")

    def test_compute_with_expectation_region_measure_3d(self, locdata_3d):
        locdata_3d = deepcopy(locdata_3d)
        collection = LocData.from_collection(locdatas=[locdata_3d, locdata_3d])
        cpe = ConvexHullExpectation(
            convex_hull_property="region_measure_ch", expected_variance=100
        ).compute(locdata=collection)
        # print(cpe.results.values)
        # print(cpe.results.grouped)
        assert cpe.results.values.index.tolist() == collection.data.index.tolist()
        assert cpe.results.values.columns.tolist() == [
            "localization_count",
            "region_measure_ch",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert cpe.results.grouped.index.tolist() == [6]
        assert cpe.results.grouped.columns.tolist() == [
            "region_measure_ch_mean",
            "region_measure_ch_std",
            "expectation",
            "expectation_std_pos",
            "expectation_std_neg",
        ]

        cpe.plot()
        # plt.show()
        cpe.hist()
        # plt.show()

        plt.close("all")

    def test_compute_without_references(self):
        collection = LocData.from_dataframe(
            dataframe=pd.DataFrame.from_dict(
                {
                    "localization_count": np.array([6, 5, 300]),
                    "position_x": np.array([2.666667, 2.666667, 5]),
                    "position_y": np.array([3.666667, 3.666667, 5]),
                    "region_measure_ch": np.array([14, 14, 100]),
                }
            )
        )
        collection.data.index = [3, 1, 4]
        cpe = ConvexHullExpectation(
            convex_hull_property="region_measure_ch", expected_variance=100
        ).compute(locdata=collection)
        # print(cpe.results.values.columns)
        # print(cpe.results.values[["localization_count", "expectation"]])
        # print(cpe.results.grouped)
        assert "region_measure_ch" in collection.data.columns
        assert cpe.results.values.index.tolist() == collection.data.index.tolist()
        assert cpe.results.values.columns.tolist() == [
            "localization_count",
            "region_measure_ch",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert cpe.results.grouped.index.tolist() == [5, 6, 300]
        assert cpe.results.grouped.columns.tolist() == [
            "region_measure_ch_mean",
            "region_measure_ch_std",
            "expectation",
            "expectation_std_pos",
            "expectation_std_neg",
        ]

        assert cpe.results.grouped.expectation.tolist() == pytest.approx(
            [250.99999999999997, 321.0, np.nan], rel=0.1, nan_ok=True
        )
        assert cpe.results.grouped.expectation_std_pos.tolist() == pytest.approx(
            [183.0, 198.0, np.nan], rel=0.1, nan_ok=True
        )
        assert cpe.results.grouped.expectation_std_neg.tolist() == pytest.approx(
            [113.99999999999999, 132.0, np.nan], rel=0.1, nan_ok=True
        )
        assert cpe.results.values.value_to_expectation_ratio.tolist() == pytest.approx(
            [0.04361370716510903, 0.05577689243027889, np.nan], rel=1e-3, nan_ok=True
        )

        cpe.plot()
        # plt.show()
        cpe.hist()
        # plt.show()

        plt.close("all")


@pytest.mark.visual
class TestConvexHullPropertyExpectationVisual:
    def test_compute_for_visualization(self, locdata_blobs_2d):
        locdata_blobs_2d = deepcopy(locdata_blobs_2d)
        grouped = locdata_blobs_2d.data.groupby("cluster_label")
        collection = LocData.from_collection(
            locdatas=[
                LocData.from_dataframe(dat.iloc[: (10 - i)])
                for i, (name, dat) in enumerate(grouped)
            ]
        )
        cpe = ConvexHullExpectation(expected_variance=400).compute(locdata=collection)

        print(cpe.results.values.columns)
        print(cpe.results.grouped)

        cpe.plot()
        plt.show()

        cpe.hist()
        plt.show()
        cpe.hist(bin_size=(1, 1000), bin_range=((3, 100), (1, 10_000)), log=False)
        plt.show()
        cpe.hist(bin_size=(1, 1000), bin_range=((3, 100), (1, 10_000)))
        plt.show()

        axes = bh.axis.AxesTuple(
            (
                bh.axis.Regular(20, 3, 503, transform=bh.axis.transform.log),
                bh.axis.Regular(50, 1, 15_000, transform=bh.axis.transform.log),
            )
        )
        cpe.hist(bins=axes)
        plt.show()

        plt.close("all")


class TestConvexHullPropertyExpectationBatch:
    def test_init(self, locdata_2d):
        cpeb = ConvexHullExpectationBatch(meta={"comment": "this is an example"})
        assert (
            str(cpeb) == "ConvexHullExpectationBatch("
            "convex_hull_property=region_measure_ch, "
            "expected_variance=None)"
        )
        assert cpeb.results is None
        assert cpeb.batch is None
        assert cpeb.meta.comment == "this is an example"

    def test_compute_empty_locdata(self, caplog):
        cpeb = ConvexHullExpectationBatch().compute(locdatas=[LocData(), LocData()])
        assert cpeb.results is None
        assert cpeb.batch is None
        assert caplog.record_tuples == [
            ("locan.analysis.convex_hull_expectation", 30, "Locdata is empty."),
            ("locan.analysis.convex_hull_expectation", 30, "Locdata is empty."),
            ("locan.analysis.convex_hull_expectation", 30, "The batch is empty."),
        ]
        cpeb.plot()
        cpeb.hist()
        # plt.show()
        plt.close("all")

    def test_from_batch_empty_item(self, caplog):
        cpeb = ConvexHullExpectationBatch().from_batch(
            batch=[ConvexHullExpectation(), ConvexHullExpectation()]
        )
        assert cpeb.results is None
        assert cpeb.batch is None
        cpeb.plot()
        cpeb.hist()
        # plt.show()

        assert caplog.record_tuples == [
            ("locan.analysis.convex_hull_expectation", 30, "The batch is empty."),
        ]

        plt.close("all")

    def test_compute_empty_locdatas(self, caplog):
        cpeb = ConvexHullExpectationBatch().compute(locdatas=[])
        assert cpeb.results is None
        assert cpeb.batch is None
        assert caplog.record_tuples == [
            ("locan.analysis.convex_hull_expectation", 30, "The batch is empty."),
        ]
        cpeb.plot()
        cpeb.hist()
        # plt.show()
        plt.close("all")

    def test_from_batch_empty(self, caplog):
        cpeb = ConvexHullExpectationBatch().from_batch(batch=[])
        assert cpeb.results is None
        assert cpeb.batch is None
        assert caplog.record_tuples == [
            ("locan.analysis.convex_hull_expectation", 30, "The batch is empty."),
        ]
        cpeb.plot()
        cpeb.hist()
        # plt.show()
        plt.close("all")

    def test_compute(self, locdata_2d):
        collection = LocData.from_dataframe(
            dataframe=pd.DataFrame.from_dict(
                {
                    "localization_count": np.array([6, 5]),
                    "position_x": np.array([2.666667, 2.666667]),
                    "position_y": np.array([3.666667, 3.666667]),
                    "region_measure_ch": np.array([14, 14]),
                }
            )
        )
        collection.data.index = [3, 1]

        collection_2 = LocData.from_dataframe(
            dataframe=pd.DataFrame.from_dict(
                {
                    "localization_count": np.array([6, 5, 300]),
                    "position_x": np.array([2.666667, 2.666667, 5]),
                    "position_y": np.array([3.666667, 3.666667, 5]),
                    "region_measure_ch": np.array([14, 14, 100]),
                }
            )
        )
        collection_2.data.index = [3, 1, 4]

        cpeb = ConvexHullExpectationBatch().compute(locdatas=[collection, collection_2])

        # print(cpeb.results.values)
        # print(cpeb.results.grouped)
        assert isinstance(cpeb.results, ConvexHullExpectationResults)
        assert cpeb.results.values.index.tolist() == list(range(5))

        assert cpeb.results.values["region_measure_ch"].tolist() == pytest.approx(
            [14, 14, 14, 14, 100], rel=0.1
        )

        assert cpeb.results.values.columns.tolist() == [
            "localization_count",
            "region_measure_ch",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert cpeb.results.grouped.index.tolist() == [5, 6, 300]
        assert cpeb.results.grouped.columns.tolist() == [
            "region_measure_ch_mean",
            "region_measure_ch_std",
            "expectation",
            "expectation_std_pos",
            "expectation_std_neg",
        ]
        assert cpeb.results.grouped.expectation.isna().all()

        cpeb.plot()
        # plt.show()
        cpeb.hist()
        # plt.show()

        plt.close("all")
