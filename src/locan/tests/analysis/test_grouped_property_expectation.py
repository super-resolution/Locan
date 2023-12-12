import boost_histogram as bh
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from locan import LocData
from locan.analysis import GroupedPropertyExpectation
from locan.analysis.grouped_property_expectation import (
    GroupedPropertyExpectationResults,
)


class TestGroupedPropertyExpectation:
    def test_init(self, locdata_2d):
        gpe = GroupedPropertyExpectation(meta={"comment": "this is an example"})
        assert (
            str(gpe) == "GroupedPropertyExpectation("
            "loc_property=None, other_loc_property=None, expectation=None)"
        )
        assert gpe.results is None
        assert gpe.meta.comment == "this is an example"

    def test_empty_locdata(self, caplog):
        gpe = GroupedPropertyExpectation().compute(LocData())
        assert gpe.results is None
        gpe.plot()
        gpe.hist()
        # plt.show()

        assert caplog.record_tuples == [
            ("locan.analysis.grouped_property_expectation", 30, "Locdata is empty."),
        ]

        plt.close("all")

    def test_compute(self, locdata_2d):
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        gpe = GroupedPropertyExpectation(
            loc_property="intensity", other_loc_property="localization_count"
        ).compute(locdata=collection)
        # print(gpe.results.values)
        # print(gpe.results.grouped)
        assert isinstance(gpe.results, GroupedPropertyExpectationResults)
        assert gpe.results.values.index.tolist() == collection.data.index.tolist()
        assert gpe.results.values.columns.tolist() == [
            "intensity",
            "localization_count",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert gpe.results.grouped.index.tolist() == [6]
        assert gpe.results.grouped.columns.tolist() == [
            "intensity_mean",
            "intensity_std",
            "expectation",
        ]
        assert gpe.results.grouped.expectation.isna().all()

        gpe.plot()
        # plt.show()
        gpe.hist()
        # plt.show()

        plt.close("all")

    def test_compute_with_expectation(self, locdata_2d):
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        expected = 2
        gpe = GroupedPropertyExpectation(
            loc_property="intensity",
            other_loc_property="localization_count",
            expectation=expected,
        ).compute(locdata=collection)
        assert gpe.results.values.index.tolist() == collection.data.index.tolist()
        assert gpe.results.values.columns.tolist() == [
            "intensity",
            "localization_count",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert gpe.results.grouped.index.tolist() == [6]
        assert gpe.results.grouped.columns.tolist() == [
            "intensity_mean",
            "intensity_std",
            "expectation",
        ]
        assert all(gpe.results.grouped.expectation == 2)
        assert all(
            gpe.results.values.value_to_expectation_ratio
            == gpe.results.values.intensity / expected
        )

        expected = pd.Series(data=[2], index=[6])
        gpe = GroupedPropertyExpectation(
            loc_property="intensity",
            other_loc_property="localization_count",
            expectation=expected,
        ).compute(locdata=collection)
        assert gpe.results.values.index.tolist() == collection.data.index.tolist()
        assert gpe.results.values.columns.tolist() == [
            "intensity",
            "localization_count",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert gpe.results.grouped.index.tolist() == [6]
        assert gpe.results.grouped.columns.tolist() == [
            "intensity_mean",
            "intensity_std",
            "expectation",
        ]
        assert all(gpe.results.grouped.expectation == 2)
        assert all(
            gpe.results.values.value_to_expectation_ratio
            == gpe.results.values.intensity / 2
        )

        expected = {6: 2}
        gpe = GroupedPropertyExpectation(
            loc_property="intensity",
            other_loc_property="localization_count",
            expectation=expected,
        ).compute(locdata=collection)
        assert gpe.results.values.index.tolist() == collection.data.index.tolist()
        assert gpe.results.grouped.index.tolist() == [6]
        assert all(gpe.results.grouped.expectation == 2)
        assert all(
            gpe.results.values.value_to_expectation_ratio
            == gpe.results.values.intensity / 2
        )

        gpe.plot()
        # plt.show()
        gpe.hist()
        # plt.show()

        plt.close("all")


@pytest.mark.visual
class TestGroupedPropertyExpectationVisual:
    def test_compute(self, locdata_blobs_2d):
        gpe = GroupedPropertyExpectation(
            loc_property="position_x",
            other_loc_property="cluster_label",
            expectation=600,
        ).compute(locdata=locdata_blobs_2d)
        assert isinstance(gpe.results, GroupedPropertyExpectationResults)
        print(gpe.results.values.columns)
        print(gpe.results.grouped)

        gpe.plot()
        plt.show()

        gpe.hist()
        plt.show()

        gpe.hist(bin_size=(1, 1000), bin_range=((3, 100), (1, 10_000)), log=False)
        plt.show()
        gpe.hist(bin_size=(1, 1000), bin_range=((3, 100), (1, 10_000)))
        plt.show()

        axes = bh.axis.AxesTuple(
            (
                bh.axis.Regular(20, 3, 503, transform=bh.axis.transform.log),
                bh.axis.Regular(50, 1, 15_000, transform=bh.axis.transform.log),
            )
        )
        gpe.hist(bins=axes)
        plt.show()

        plt.close("all")
