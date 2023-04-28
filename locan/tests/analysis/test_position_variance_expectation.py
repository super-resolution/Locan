import boost_histogram as bh
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from locan import LocData
from locan.analysis import PositionVarianceExpectation
from locan.analysis.position_variance_expectation import (
    PositionVarianceExpectationResults,
    _property_variances,
)


def test__property_variances(locdata_2d):
    collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
    result = _property_variances(collection=collection, loc_property="position_x")
    assert all(result["localization_count"] == [6, 6])
    assert all(
        result["position_x_var"] == [pytest.approx(2.666667), pytest.approx(2.666667)]
    )


class TestPositionVarianceExpectation:
    def test_init(self, locdata_blobs_2d):
        gpe = PositionVarianceExpectation(meta={"comment": "this is an example"})
        assert (
            str(gpe) == "PositionVarianceExpectation("
            "loc_property=position_x, expectation=None, biased=True)"
        )
        assert gpe.results is None
        assert gpe.meta.comment == "this is an example"

        gpe = PositionVarianceExpectation(
            meta={"comment": "this is an example"},
            loc_property="position_y",
        )
        assert (
            str(gpe) == "PositionVarianceExpectation("
            "loc_property=position_y, expectation=None, biased=True)"
        )
        assert gpe.results is None
        assert gpe.meta.comment == "this is an example"

    def test_empty_locdata(self, caplog):
        gpe = PositionVarianceExpectation().compute(LocData())
        assert gpe.results is None
        gpe.plot()
        gpe.hist()
        # plt.show()

        assert caplog.record_tuples == [
            ("locan.analysis.position_variance_expectation", 30, "Locdata is empty."),
        ]

        plt.close("all")

    def test_compute(self, locdata_2d):
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        pve = PositionVarianceExpectation().compute(locdata=collection)
        assert pve.results.values["position_x_var"].tolist() == pytest.approx(
            [2.2222222222222228, 2.2222222222222228], rel=0.1
        )
        # print(pve.results.values)
        # print(pve.results.grouped)
        assert isinstance(pve.results, PositionVarianceExpectationResults)
        assert pve.results.values.index.tolist() == collection.data.index.tolist()
        assert pve.results.values.columns.tolist() == [
            "localization_count",
            "position_x_var",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert pve.results.grouped.index.tolist() == [6]
        assert pve.results.grouped.columns.tolist() == [
            "position_x_var_mean",
            "position_x_var_std",
            "expectation",
        ]
        assert pve.results.grouped.expectation.isna().all()

        pve.plot()
        # plt.show()
        pve.hist()
        # plt.show()

        pve = PositionVarianceExpectation(biased=False).compute(locdata=collection)
        assert pve.results.values["position_x_var"].tolist() == pytest.approx(
            [2.67, 2.67], rel=0.1
        )

        plt.close("all")

    def test_compute_with_expectation(self, locdata_2d):
        # biased
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        expected = 2
        gpe = PositionVarianceExpectation(
            expectation=expected,
        ).compute(locdata=collection)
        assert gpe.results.values.index.tolist() == collection.data.index.tolist()
        assert gpe.results.values.columns.tolist() == [
            "localization_count",
            "position_x_var",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert gpe.results.grouped.index.tolist() == [6]
        assert gpe.results.grouped.columns.tolist() == [
            "position_x_var_mean",
            "position_x_var_std",
            "expectation",
        ]
        assert gpe.results.grouped.expectation.tolist() == [
            pytest.approx(1.666667, rel=1e-5)
        ]
        assert all(
            gpe.results.values.value_to_expectation_ratio
            == [pytest.approx(1.33333, rel=1e-5), pytest.approx(1.33333, rel=1e-5)]
        )

        # unbiased
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        expected = 2
        gpe = PositionVarianceExpectation(expectation=expected, biased=False).compute(
            locdata=collection
        )
        assert gpe.results.values.index.tolist() == collection.data.index.tolist()
        assert gpe.results.values.columns.tolist() == [
            "localization_count",
            "position_x_var",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert gpe.results.grouped.index.tolist() == [6]
        assert gpe.results.grouped.columns.tolist() == [
            "position_x_var_mean",
            "position_x_var_std",
            "expectation",
        ]
        assert all(gpe.results.grouped.expectation == 2)
        assert all(
            gpe.results.values.value_to_expectation_ratio
            == gpe.results.values.position_x_var / expected
        )

        expected = pd.Series(data=[2], index=[6])
        gpe = PositionVarianceExpectation(expectation=expected, biased=False).compute(
            locdata=collection
        )
        assert gpe.results.values.index.tolist() == collection.data.index.tolist()
        assert gpe.results.values.columns.tolist() == [
            "localization_count",
            "position_x_var",
            "expectation",
            "value_to_expectation_ratio",
        ]
        assert gpe.results.grouped.index.tolist() == [6]
        assert gpe.results.grouped.columns.tolist() == [
            "position_x_var_mean",
            "position_x_var_std",
            "expectation",
        ]
        assert all(gpe.results.grouped.expectation == 2)
        assert all(
            gpe.results.values.value_to_expectation_ratio
            == gpe.results.values.position_x_var / 2
        )

        expected = {6: 2}
        gpe = PositionVarianceExpectation(expectation=expected, biased=False).compute(
            locdata=collection
        )
        assert gpe.results.values.index.tolist() == collection.data.index.tolist()
        assert gpe.results.grouped.index.tolist() == [6]
        assert all(gpe.results.grouped.expectation == 2)
        assert all(
            gpe.results.values.value_to_expectation_ratio
            == gpe.results.values.position_x_var / 2
        )

        gpe.plot()
        # plt.show()
        gpe.hist()
        # plt.show()

        plt.close("all")


@pytest.mark.visual
class TestPositionVarianceExpectationVisual:
    def test_compute(self, locdata_blobs_2d):
        grouped = locdata_blobs_2d.data.groupby("cluster_label")
        collection = LocData.from_collection(
            locdatas=[
                LocData.from_dataframe(dat.iloc[: (10 - i)])
                for i, (name, dat) in enumerate(grouped)
            ]
        )
        pve = PositionVarianceExpectation(
            expectation=400,
        ).compute(locdata=collection)
        assert isinstance(pve.results, PositionVarianceExpectationResults)
        print(pve.results.values.columns)
        print(pve.results.grouped)

        pve.plot()
        plt.show()

        pve.hist()
        plt.show()

        pve.hist(bin_size=(1, 1000), bin_range=((3, 100), (1, 10_000)), log=False)
        plt.show()
        pve.hist(bin_size=(1, 1000), bin_range=((3, 100), (1, 10_000)))
        plt.show()

        axes = bh.axis.AxesTuple(
            (
                bh.axis.Regular(20, 3, 503, transform=bh.axis.transform.log),
                bh.axis.Regular(50, 1, 15_000, transform=bh.axis.transform.log),
            )
        )
        pve.hist(bins=axes)
        plt.show()

        plt.close("all")
