from copy import deepcopy

import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import pytest

from locan import LocData
from locan.analysis import ConvexHullExpectation
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
    assert bool(result.n_points) is False

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
        # print(cpe.results.values.)
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

    def test_compute_with_expectation_subregion(self, locdata_2d):
        locdata_2d = deepcopy(locdata_2d)
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        cpe = ConvexHullExpectation(
            convex_hull_property="subregion_measure_ch", expected_variance=100
        ).compute(locdata=collection)
        # print(cpe.results.values)
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
