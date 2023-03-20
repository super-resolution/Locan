import boost_histogram as bh
import matplotlib.pyplot as plt
import pytest

from locan import LocData
from locan.analysis import PositionVarianceExpectation
from locan.analysis.position_variance_expectation import (
    PositionVarianceExpectationResults,
    _expected_variance_biased,
    _position_variances,
)


def test__biased_variance_expectation():
    result = _expected_variance_biased(
        expected_variance=[1, 2], localization_counts=[100, 100]
    )
    assert result.tolist() == [0.99, 1.98]


def test__position_variances(locdata_2d):
    collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
    result = _position_variances(collection=collection)
    assert result["position_x_var"] == pytest.approx([2.67, 2.67], rel=0.1)
    assert result["position_y_var"] == pytest.approx([3.87, 3.87], rel=0.1)
    result = _position_variances(collection=collection, loc_properties=["position_x"])
    assert result["position_x_var"] == pytest.approx([2.67, 2.67], rel=0.1)


class TestPositionVarianceExpectation:
    def test_init(self, locdata_2d):
        pve = PositionVarianceExpectation(meta={"comment": "this is an example"})
        assert (
            str(pve) == "PositionVarianceExpectation("
            "loc_properties=None, expected_variance=None, biased=False)"
        )
        assert pve.results is None
        assert pve.meta.comment == "this is an example"

    def test_empty_locdata(self, caplog):
        pve = PositionVarianceExpectation().compute(LocData())
        assert pve.results is None
        pve.plot()
        pve.hist()
        # plt.show()

        assert caplog.record_tuples == [
            ("locan.analysis.position_variance_expectation", 30, "Locdata is empty."),
        ]

        plt.close("all")

    def test_compute(self, locdata_2d):
        collection = LocData.from_collection(locdatas=[locdata_2d, locdata_2d])
        pve = PositionVarianceExpectation().compute(locdata=collection)
        assert isinstance(pve.results, PositionVarianceExpectationResults)
        assert pve.results.variances["position_x_var"].tolist() == pytest.approx(
            [2.67, 2.67], rel=0.1
        )
        assert pve.results.variances_mean is not None
        assert pve.results.variances_std is not None

        pve.plot(loc_property="position_x_var")
        pve.plot(loc_property=["position_x_var", "position_y_var"])
        pve.plot()
        # plt.show()

        pve = PositionVarianceExpectation(expected_variance=100).compute(
            locdata=collection
        )
        assert isinstance(pve.results, PositionVarianceExpectationResults)
        assert pve.results.variances["position_x_var"].tolist() == pytest.approx(
            [2.67, 2.67], rel=0.1
        )
        assert pve.results.variances_mean is not None
        assert pve.results.variances_std is not None

        pve.plot(loc_property="position_x_var")
        pve.plot(loc_property=["position_x_var", "position_y_var"])
        pve.plot()
        # plt.show()

        pve.hist()
        pve.hist(n_bins=(20, 50), bin_range=((3, 100), (1, 1_000)))

        axes = bh.axis.AxesTuple(
            (
                bh.axis.Regular(20, 3, 100, transform=bh.axis.transform.log),
                bh.axis.Regular(50, 1, 1_000, transform=bh.axis.transform.log),
            )
        )
        pve.hist(loc_property="position_y_var", bins=axes)
        # plt.show()

        pve = PositionVarianceExpectation(expected_variance=100, biased=True).compute(
            locdata=collection
        )
        assert isinstance(pve.results, PositionVarianceExpectationResults)
        assert pve.results.variances["position_x_var"].tolist() == pytest.approx(
            [2.2, 2.2], rel=0.1
        )
        assert pve.results.variances_mean is not None
        assert pve.results.variances_std is not None

        pve.plot()
        # plt.show()

        pve.hist()
        # plt.show()

        plt.close("all")


@pytest.mark.visual
class TestPositionVarianceExpectationVisual:
    def test_compute_for_visualization(self, locdata_blobs_2d):
        grouped = locdata_blobs_2d.data.groupby("cluster_label")
        collection = LocData.from_collection(
            locdatas=[
                LocData.from_dataframe(dat.iloc[: (10 - i)])
                for i, (name, dat) in enumerate(grouped)
            ]
        )
        pve = PositionVarianceExpectation(expected_variance=400, biased=True).compute(
            locdata=collection
        )

        print(pve.results.variances)
        print(pve.results.variances_mean)
        print(pve.results.variances_std)

        assert isinstance(pve.results, PositionVarianceExpectationResults)
        assert pve.results.variances is not None
        assert pve.results.variances_mean is not None
        assert pve.results.variances_std is not None

        # pve.plot(loc_property="position_x_var")
        plt.show()
        pve.plot(loc_property=["position_x_var", "position_y_var"])
        plt.show()
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
        pve.hist(loc_property="position_y_var", bins=axes)
        plt.show()

        plt.close("all")
