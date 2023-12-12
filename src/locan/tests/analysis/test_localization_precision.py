import matplotlib.pyplot as plt  # this import is required for visual inspection
import numpy as np
import pandas as pd
import pytest

from locan import LocData
from locan.analysis.localization_precision import (
    LocalizationPrecision,
    PairwiseDistance1d,
    PairwiseDistance1dIdenticalSigmaZeroMu,
    PairwiseDistance2d,
    PairwiseDistance2dIdenticalSigma,
    PairwiseDistance2dIdenticalSigmaZeroMu,
    PairwiseDistance3d,
    PairwiseDistance3dIdenticalSigmaZeroMu,
    _DistributionFits,
)


@pytest.fixture()
def locdata_simple_1d():
    locdata_dict = {"position_y": range(10), "frame": [1, 1, 1, 2, 2, 3, 5, 8, 9, 10]}
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def locdata_simple_2d():
    locdata_dict = {
        "position_x": range(10),
        "position_y": range(10),
        "frame": [1, 1, 1, 2, 2, 3, 5, 8, 9, 10],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def locdata_simple_3d():
    locdata_dict = {
        "position_x": range(10),
        "position_y": range(10),
        "position_z": range(10),
        "frame": [1, 1, 1, 2, 2, 3, 5, 8, 9, 10],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


def test_PairwiseDistance1d():
    x_data = np.array([0, 1, 2])
    distribution = PairwiseDistance1d()
    assert distribution.shapes == "mu, sigma_1, sigma_2"
    pdf_values = distribution.pdf(x=x_data, mu=1, sigma_1=0.1, sigma_2=0.2)
    assert pdf_values == pytest.approx([1.61998219e-04, 1.78412412e00, 8.09991096e-05])


def test_PairwiseDistance1dIdenticalSigmaZeroMu():
    x_data = np.array([0, 1, 2])
    distribution = PairwiseDistance1dIdenticalSigmaZeroMu()
    assert distribution.shapes == "sigma"
    pdf_values = distribution.pdf(x=x_data, sigma=0.1)
    assert pdf_values == pytest.approx([7.97884561e00, 1.53891973e-21, 1.10418967e-86])


def test_PairwiseDistance2d():
    x_data = np.array([0, 1, 2])
    distribution = PairwiseDistance2d()
    assert distribution.shapes == "mu, sigma_1, sigma_2"
    pdf_values = distribution.pdf(x=x_data, mu=1, sigma_1=0.1, sigma_2=0.2)
    assert pdf_values == pytest.approx([0.00000000e00, 1.79560624e00, 1.14913178e-04])


def test_PairwiseDistance2dIdenticalSigma():
    x_data = np.array([0, 1, 2])
    distribution = PairwiseDistance2dIdenticalSigma()
    assert distribution.shapes == "mu, sigma"
    pdf_values = distribution.pdf(x=x_data, mu=1, sigma=0.1)
    assert pdf_values == pytest.approx([0.00000000e00, 3.99443793e00, 1.08886261e-21])


def test_PairwiseDistance2dIdenticalSigmaZeroMu():
    x_data = np.array([0, 1, 2])
    distribution = PairwiseDistance2dIdenticalSigmaZeroMu()
    assert distribution.shapes == "sigma"
    pdf_values = distribution.pdf(x=x_data, sigma=0.1)
    assert pdf_values == pytest.approx([0.00000000e00, 1.92874985e-20, 2.76779305e-85])


def test_PairwiseDistance3d():
    x_data = np.array([0, 1, 2])
    distribution = PairwiseDistance3d()
    assert distribution.shapes == "mu, sigma_1, sigma_2"
    pdf_values = distribution.pdf(x=x_data, mu=1, sigma_1=0.1, sigma_2=0.2)
    assert pdf_values == pytest.approx([0.00000000e00, 1.78412412e00, 1.61998219e-04])


def test_PairwiseDistance3dIdenticalSigmaZeroMu():
    x_data = np.array([0, 1, 2])
    distribution = PairwiseDistance3dIdenticalSigmaZeroMu()
    assert distribution.shapes == "sigma"
    pdf_values = distribution.pdf(x=x_data, sigma=0.1)
    assert pdf_values == pytest.approx([0.00000000e00, 1.53891973e-19, 4.41675869e-84])


def test_LocalizationPrecision_empty(caplog):
    lp = LocalizationPrecision().compute(LocData())
    lp.fit_distributions()
    lp.plot()
    lp.hist()
    assert caplog.record_tuples == [
        ("locan.analysis.localization_precision", 30, "Locdata is empty."),
        (
            "locan.analysis.localization_precision",
            30,
            "No results available to be fitted.",
        ),
    ]


def test_Localization_precision(locdata_simple_2d):
    # print(locdata_simple.data)
    lp = LocalizationPrecision().compute(locdata=locdata_simple_2d)
    # print(lp.results)
    for prop in ["position_delta_x", "position_delta_y", "position_distance"]:
        assert prop in lp.results.columns


@pytest.mark.visual
def test_Localization_precision_plot(locdata_rapidSTORM_2d):
    lp = LocalizationPrecision().compute(locdata=locdata_rapidSTORM_2d)
    lp.plot(window=10)
    plt.show()
    print(lp.results)
    print(lp.meta)


class Test_Distribution_fits:
    def test_init(self):
        lp = LocalizationPrecision()
        distribution_statistics = _DistributionFits(lp)
        with pytest.raises(ValueError):
            distribution_statistics.fit()
        distribution_statistics.plot()


def test_Distribution_fits_1d(locdata_simple_1d):
    # print(locdata_simple.data)
    lp = LocalizationPrecision().compute(locdata=locdata_simple_1d)
    distribution_statistics = _DistributionFits(lp)
    distribution_statistics.fit()
    assert "position_distance_sigma" in distribution_statistics.parameters
    distribution_statistics.plot()
    # plt.show()

    lp.fit_distributions(loc_property=None)
    assert lp.distribution_statistics.position_delta_y_loc
    # print(lp.distribution_statistics)

    lp.fit_distributions(loc_property="position_delta_y", floc=0)
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_delta_y_loc == 0

    lp.fit_distributions(loc_property="position_distance")
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_distance_loc == 0

    lp.fit_distributions(loc_property="position_distance", floc=1)
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_distance_loc == 1

    plt.close("all")


def test_Distribution_fits(locdata_simple_2d):
    # print(locdata_simple.data)
    lp = LocalizationPrecision().compute(locdata=locdata_simple_2d)
    distribution_statistics = _DistributionFits(lp)
    distribution_statistics.fit()
    assert "position_distance_sigma" in distribution_statistics.parameters
    distribution_statistics.plot()
    # plt.show()

    lp.fit_distributions(loc_property=None)
    assert lp.distribution_statistics.position_delta_x_loc
    # print(lp.distribution_statistics)

    lp.fit_distributions(loc_property="position_delta_x", floc=0)
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_delta_x_loc == 0

    lp.fit_distributions(loc_property="position_distance")
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_distance_loc == 0

    lp.fit_distributions(loc_property="position_distance", floc=1)
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_distance_loc == 1

    plt.close("all")


def test_Distribution_fits_3d(locdata_simple_3d):
    # print(locdata_simple.data)
    lp = LocalizationPrecision().compute(locdata=locdata_simple_3d)
    distribution_statistics = _DistributionFits(lp)
    distribution_statistics.fit()
    assert "position_distance_sigma" in distribution_statistics.parameters
    distribution_statistics.plot()
    # plt.show()

    lp.fit_distributions(loc_property=None)
    assert lp.distribution_statistics.position_delta_x_loc
    # print(lp.distribution_statistics)

    lp.fit_distributions(loc_property="position_delta_x", floc=0)
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_delta_x_loc == 0

    lp.fit_distributions(loc_property="position_distance")
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_distance_loc == 0

    lp.fit_distributions(loc_property="position_distance", floc=1)
    assert lp.distribution_statistics.parameters
    assert lp.distribution_statistics.parameter_dict()
    assert lp.distribution_statistics.position_distance_loc == 1

    plt.close("all")


def test_Localization_precision_histogram(locdata_simple_2d):
    lp = LocalizationPrecision().compute(locdata=locdata_simple_2d)
    # lp.hist(fit=True)

    with pytest.raises(AttributeError):
        lp.hist(loc_property="position_delta_x", fit=False)
        assert lp.distribution_statistics.position_delta_x_loc

    lp.hist(loc_property="position_delta_x", fit=True)
    # print(lp.distribution_statistics.parameter_dict())
    assert lp.distribution_statistics.position_delta_x_loc
    assert lp.distribution_statistics.position_delta_x_scale

    plt.close("all")


# standard LocData fixtures


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("locdata_empty", 0),
        ("locdata_single_localization", 1),
        ("locdata_2d", 6),
        ("locdata_non_standard_index", 6),
    ],
)
def test_standard_locdata_objects(
    locdata_empty,
    locdata_single_localization,
    locdata_2d,
    locdata_non_standard_index,
    fixture_name,
    expected,
):
    dat = eval(fixture_name)
    LocalizationPrecision(radius=1).compute(locdata=dat)
