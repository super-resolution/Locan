import matplotlib.pyplot as plt  # this import is needed for visual inspection
import numpy as np
import pandas as pd
import pytest

from locan import LocData, NearestNeighborDistances, simulate_uniform
from locan.analysis.nearest_neighbor import (
    NNDistances_csr_2d,
    NNDistances_csr_3d,
    _DistributionFits,
    pdf_nnDistances_csr_2D,
    pdf_nnDistances_csr_3D,
)

# fixtures


@pytest.fixture()
def locdata_simple():
    locdata_dict = {
        "position_x": [0, 0, 1, 4, 5],
        "position_y": [0, 1, 3, 4, 1],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


@pytest.fixture()
def other_locdata_simple():
    locdata_dict = {
        "position_x": [10, 11],
        "position_y": [10, 11],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


# tests


def test_pdf_nnDistances_csr_2D():
    pdf_values = pdf_nnDistances_csr_2D(np.array([0, 1, 2]), density=1)
    assert pdf_values == pytest.approx([0, 2.71521056e-001, 4.38232365e-05])


def test_pdf_nnDistances_csr_3D():
    pdf_values = pdf_nnDistances_csr_3D(np.array([0, 1, 2]), density=1)
    assert pdf_values == pytest.approx([0, 1.90564233e-01, 1.40579528e-13])


def test_NNDistances_csr_2d():
    dist = NNDistances_csr_2d()
    assert dist.shapes == "density"
    assert dist.pdf(x=np.array([0, 1, 2]), density=1) == pytest.approx(
        [0, 2.71521056e-01, 4.38232365e-05]
    )


def test_NNDistances_csr_3d():
    dist = NNDistances_csr_3d()
    assert dist.shapes == "density"
    assert dist.pdf(x=np.array([0, 1, 2]), density=1) == pytest.approx(
        [0, 1.90564233e-01, 1.40579528e-13]
    )


def test_Nearest_neighbor_distances_empty(caplog):
    nn_1 = NearestNeighborDistances().compute(LocData())
    nn_1.hist()
    assert caplog.record_tuples == [
        ("locan.analysis.nearest_neighbor", 30, "Locdata is empty.")
    ]


def test_DistributionFits(locdata_simple):
    nn_1 = NearestNeighborDistances()
    assert repr(nn_1) == "NearestNeighborDistances(k=1)"
    nn_1.compute(locdata_simple)
    assert nn_1.dimension == 2
    assert nn_1.localization_density == 0.25
    # print(nn_1.results)
    assert len(nn_1.results) == 5
    assert all(nn_1.results.columns == ["nn_distance", "nn_index"])
    assert nn_1.results["nn_index"].iloc[0] == 1
    assert nn_1.distribution_statistics is None
    # nn_1.hist()

    ds = _DistributionFits(nn_1)
    ds.fit()
    assert ds.parameter_dict() == {
        "density": pytest.approx(0.058984374999999166),
        "loc": 0,
        "scale": 1,
    }
    ds.fit(with_constraints=False)
    assert ds.parameter_dict() == pytest.approx(
        {
            "density": 0.8900132393760514,
            "loc": 0.35972770464810505,
            "scale": 3.348388757318209,
        }
    )
    # ds.plot()

    nn_1.hist(fit=True)
    assert nn_1.distribution_statistics.parameter_dict() == {
        "density": pytest.approx(0.058984374999999166),
        "loc": 0,
        "scale": 1,
    }
    # plt.show()

    plt.close("all")


def test_DistributionFits_k2(locdata_simple):
    nn_1 = NearestNeighborDistances(k=2)
    assert repr(nn_1) == "NearestNeighborDistances(k=2)"
    nn_1.compute(locdata_simple)
    assert nn_1.dimension == 2
    assert nn_1.localization_density == 0.25
    # print(nn_1.results)
    assert len(nn_1.results) == 5
    assert all(nn_1.results.columns == ["nn_distance", "nn_index"])
    assert nn_1.results["nn_index"].iloc[0] == 2
    assert nn_1.distribution_statistics is None
    nn_1.fit_distributions()
    assert nn_1.distribution_statistics.parameter_dict() == {
        "density": pytest.approx(0.028906249999999137),
        "loc": 0,
        "scale": 1,
    }

    nn_1.hist(fit=True)
    # plt.show()

    plt.close("all")


def test_Nearest_neighbor_distances(locdata_simple, other_locdata_simple):
    nn_2 = NearestNeighborDistances().compute(
        locdata_simple, other_locdata=other_locdata_simple
    )
    # print(nn_2.results)
    assert nn_2.results["nn_distance"].iloc[0] == pytest.approx(14.142135623730951)


def test_NearestNeighborDistances_3d(locdata_3d):
    # print(locdata_3d.data)
    nn_1 = NearestNeighborDistances().compute(locdata_3d)
    assert repr(nn_1) == "NearestNeighborDistances(k=1)"
    assert nn_1.dimension == 3
    assert nn_1.localization_density == 0.075
    # print(nn_1.results)
    assert len(nn_1.results) == 6
    assert all(nn_1.results.columns == ["nn_distance", "nn_index"])
    assert nn_1.results["nn_index"].iloc[0] == 4
    assert nn_1.distribution_statistics is None
    # nn_1.hist()

    ds = _DistributionFits(nn_1)
    ds.fit()
    assert ds.parameter_dict() == {
        "density": pytest.approx(0.007666015624999118),
        "loc": 0,
        "scale": 1,
    }
    ds.fit(with_constraints=False)
    assert ds.parameter_dict() == pytest.approx(
        {
            "density": 1.0047585928889682,
            "loc": 2.4220565244971968,
            "scale": 1.3178818824051617,
        }
    )
    # ds.plot()

    nn_1.hist(fit=True)
    assert nn_1.distribution_statistics.parameter_dict() == {
        "density": pytest.approx(0.007666015624999118),
        "loc": 0,
        "scale": 1,
    }
    # plt.show()

    plt.close("all")


def test_NearestNeighborDistances_1d(caplog, locdata_1d):
    nn_1 = NearestNeighborDistances().compute(locdata_1d)
    assert repr(nn_1) == "NearestNeighborDistances(k=1)"
    assert nn_1.dimension == 1
    assert nn_1.localization_density == 1.5
    # print(nn_1.results)
    assert len(nn_1.results) == 6
    assert all(nn_1.results.columns == ["nn_distance", "nn_index"])
    assert nn_1.results["nn_index"].iloc[0] == 1
    assert nn_1.distribution_statistics is None
    # nn_1.hist()

    ds = _DistributionFits(nn_1)
    ds.fit()
    assert caplog.record_tuples == [
        (
            "locan.analysis.nearest_neighbor",
            30,
            "No fit model for 1 dimensions available.",
        )
    ]
    assert ds.parameter_dict() == {}
    ds.plot()

    nn_1.hist(fit=True)
    # plt.show()

    plt.close("all")


@pytest.mark.visual
def test_NearestNeighborDistances_2d_random():
    locdata = simulate_uniform(n_samples=10_000, region=((0, 1), (0, 1)))
    nn_1 = NearestNeighborDistances().compute(locdata)
    assert nn_1.localization_density == locdata.properties["localization_density_bb"]
    assert len(nn_1.results) == 10_000
    nn_1.hist(fit=True)
    plt.show()


@pytest.mark.visual
def test_NearestNeighborDistances_3d_random():
    locdata = simulate_uniform(n_samples=10_000, region=((0, 1), (0, 1), (0, 1)))
    nn_1 = NearestNeighborDistances().compute(locdata)
    assert nn_1.localization_density == locdata.properties["localization_density_bb"]
    assert len(nn_1.results) == 10_000
    nn_1.hist(fit=True)
    plt.show()
