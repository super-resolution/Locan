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
    _get_convex_hull_region_measure_2d_expectation,
    _get_resource,
    compute_convex_hull_region_measure_2d,
)


def test_get_resource():
    for resource_ in ConvexHullExpectationResource:
        result = _get_resource(
            resource_directory="locan.analysis.resources.convex_hull_expectation",
            resource=resource_,
        )
        assert isinstance(result, ConvexHullExpectationValues)


def test_compute_convex_hull_region_measure_2d():
    computed = compute_convex_hull_region_measure_2d(n_points=3)
    from_resource = _get_resource(
        resource_directory="locan.analysis.resources.convex_hull_expectation",
        resource=ConvexHullExpectationResource.REGION_MEASURE_2D,
    ).expectation[0]
    assert pytest.approx(computed, rel=0.01) == from_resource == 0.87

    computed = compute_convex_hull_region_measure_2d(n_points=2, sigma=2)
    assert np.isnan(computed)


def test__get_convex_hull_region_measure_2d_expectation():
    result = _get_convex_hull_region_measure_2d_expectation(n_points=10, sigma=2)
    assert result == ConvexHullExpectationValues(
        n_points=np.array([10]), expectation=21.52, std_pos=9.16, std_neg=6.72
    )

    result = _get_convex_hull_region_measure_2d_expectation(n_points=[3, 10], sigma=2)
    assert np.array_equal(result.n_points, np.array([3, 10]))
    assert np.array_equal(result.expectation, np.array([3.48, 21.52]))

    result = _get_convex_hull_region_measure_2d_expectation(
        n_points=[1, 3, 10], sigma=2
    )
    assert np.array_equal(result.n_points, np.array([3, 10]))
    assert np.array_equal(result.expectation, np.array([3.48, 21.52]))

    result = _get_convex_hull_region_measure_2d_expectation(
        n_points=np.array([1, 3, 10]), sigma=2
    )
    assert np.array_equal(result.n_points, np.array([3, 10]))
    assert np.array_equal(result.expectation, np.array([3.48, 21.52]))


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
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        cpe = ConvexHullExpectation().compute(locdata=collection)
        assert isinstance(cpe.results, ConvexHullExpectationResults)
        assert cpe.results.convex_hull_property[
            "region_measure_ch"
        ].tolist() == pytest.approx([14, 14], rel=0.1)
        assert cpe.results.convex_hull_property_mean is not None
        assert cpe.results.convex_hull_property_std is not None
        assert cpe.results.convex_hull_property_expectation is not None

        cpe.plot()
        # plt.show()

        cpe = ConvexHullExpectation(
            convex_hull_property="subregion_measure_ch", expected_variance=100
        ).compute(locdata=collection)
        assert isinstance(cpe.results, ConvexHullExpectationResults)
        assert cpe.results.convex_hull_property[
            "subregion_measure_ch"
        ].tolist() == pytest.approx([14, 14], rel=0.1)
        assert cpe.results.convex_hull_property_mean is not None
        assert cpe.results.convex_hull_property_std is not None

        cpe.plot()
        # plt.show()

        cpe.hist()
        # plt.show()

        cpe.hist(n_bins=(20, 50), bin_range=((3, 100), (1, 1_000)))
        # plt.show()

        axes = bh.axis.AxesTuple(
            (
                bh.axis.Regular(20, 3, 100, transform=bh.axis.transform.log),
                bh.axis.Regular(50, 1, 1_000, transform=bh.axis.transform.log),
            )
        )
        cpe.hist(bins=axes)
        # plt.show()

        plt.close("all")


@pytest.mark.visual
class TestConvexHullPropertyExpectationVisual:
    def test_compute_for_visualization(self, locdata_blobs_2d):
        grouped = locdata_blobs_2d.data.groupby("cluster_label")
        collection = LocData.from_collection(
            locdatas=[
                LocData.from_dataframe(dat.iloc[: (10 - i)])
                for i, (name, dat) in enumerate(grouped)
            ]
        )
        cpe = ConvexHullExpectation(expected_variance=400).compute(locdata=collection)

        print(cpe.results.convex_hull_property)
        print(cpe.results.convex_hull_property_mean)
        print(cpe.results.convex_hull_property_std)

        assert isinstance(cpe.results, ConvexHullExpectationResults)
        assert cpe.results.convex_hull_property is not None
        assert cpe.results.convex_hull_property_mean is not None
        assert cpe.results.convex_hull_property_std is not None

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
