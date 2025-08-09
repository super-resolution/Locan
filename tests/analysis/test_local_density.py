import matplotlib.pyplot as plt  # this import is needed for visual inspection
import numpy as np
import pandas as pd
import pytest

from locan import Ellipse, LocalDensity, LocData, simulate_uniform
from locan.analysis.local_density import _local_density

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
        "position_x": [1, 11],
        "position_y": [1, 11],
    }
    return LocData(dataframe=pd.DataFrame.from_dict(locdata_dict))


# tests


def test__local_density(locdata_simple, other_locdata_simple):
    points = locdata_simple.coordinates
    local_densities = _local_density(
        points=points,
        radii=[1],
        density=False,
        boundary_correction=None,
        normalization=None,
        other_points=None,
    )
    assert local_densities.shape == (1, 5)
    assert local_densities[0][0] == 1

    local_densities = _local_density(
        points=points,
        radii=[1, 2],
        density=True,
        boundary_correction=None,
        normalization=10,
        other_points=None,
    )
    assert local_densities.shape == (2, 5)

    other_points = other_locdata_simple.coordinates
    local_densities = _local_density(
        points=points,
        radii=[1, 2],
        density=False,
        boundary_correction=None,
        normalization=None,
        other_points=other_points,
    )
    assert local_densities.shape == (2, 2)

    local_densities = _local_density(
        points=points,
        radii=[1],
        density=False,
        boundary_correction=Ellipse((0, 0), 2, 2),
        normalization=None,
        other_points=None,
    )
    assert local_densities.shape == (1, 5)
    assert local_densities[0][0] == pytest.approx(1.00160819)


def test_LocalDensity_distances_empty(caplog):
    ld = LocalDensity(radii=1).compute(LocData())
    assert ld.results is None
    # ld.hist()
    assert caplog.record_tuples == [
        ("locan.analysis.local_density", 30, "Locdata is empty.")
    ]


def test_LocalDensity(locdata_simple):
    ld = LocalDensity(radii=[1])
    assert (
        repr(ld)
        == "LocalDensity(radii=[1], density=True, boundary_correction=False, normalization=None)"
    )
    ld.compute(locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [1])
    assert ld.results[1].iloc[0] == pytest.approx(0.3183098861837907)
    ld.hist()
    # plt.show()

    ld = LocalDensity(radii=[1], density=False)
    ld.compute(locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [1])
    assert ld.results[1].iloc[0] == pytest.approx(1)
    ld.hist()
    # plt.show()

    ld = LocalDensity(radii=[1], density=False, normalization=10)
    ld.compute(locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [1])
    assert ld.results[1].iloc[0] == pytest.approx(0.1)
    ld.hist()
    # plt.show()

    ld = LocalDensity(radii=[1], density=True, normalization=10)
    ld.compute(locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [1])
    assert ld.results[1].iloc[0] == pytest.approx(0.03183098861837907)
    ld.hist()
    # plt.show()

    ld = LocalDensity(radii=[1, 2.5])
    ld.compute(locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [1, 2.5])
    ld.hist()
    # plt.show()

    ld = LocalDensity(radii=[1, 2.5])
    ld.compute(locdata_simple, other_locdata=locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [1, 2.5])
    ld.hist()
    # plt.show()

    ld = LocalDensity(radii=[1], density=False, boundary_correction=True)
    locdata_simple.region = locdata_simple.bounding_box.region
    ld.compute(locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [1])
    assert ld.results[1].iloc[0] == pytest.approx(4.0064327563359)

    plt.close("all")


def test_LocalDensity_normalization(locdata_simple):
    normalization = np.arange(len(locdata_simple)) * 10
    ld = LocalDensity(radii=[10], density=False, normalization=normalization)
    assert (
        repr(ld)
        == "LocalDensity(radii=[10], density=False, boundary_correction=False, normalization=[ 0 10 20 30 40])"
    )
    ld.compute(locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [10])

    ld = LocalDensity(radii=[10, 20], density=False, normalization=normalization)
    ld.compute(locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [10, 20])

    normalization = np.ones((2, len(locdata_simple))) * [[2], [3]]
    ld = LocalDensity(radii=[10, 20], density=False, normalization=normalization)
    ld.compute(locdata_simple)
    assert len(ld.results) == 5
    assert all(ld.results.columns == [10, 20])


def test_LocalDensity_3d(locdata_3d):
    ld = LocalDensity(radii=[1, 2.5])
    ld.compute(locdata_3d)
    assert len(ld.results) == 6
    assert all(ld.results.columns == [1, 2.5])
    ld.hist()
    # plt.show()

    plt.close("all")


def test_LocalDensity_1d_(locdata_1d):
    ld = LocalDensity(radii=[1, 2.5])
    ld.compute(locdata_1d)
    assert len(ld.results) == 6
    assert all(ld.results.columns == [1, 2.5])
    ld.hist()
    # plt.show()

    plt.close("all")


@pytest.mark.visual
def test_LocalDensity_2d_random():
    locdata = simulate_uniform(n_samples=1_000, region=((0, 1), (0, 1)))
    ld = LocalDensity([0.1], density=False).compute(locdata)
    assert len(ld.results) == 1_000
    print(ld.results.describe())
    ld.hist(density=False)
    plt.show()
    ld.hist(density=True)
    plt.show()


@pytest.mark.visual
def test_LocalDensity_3d_random():
    locdata = simulate_uniform(n_samples=1_000, region=((0, 1), (0, 1), (0, 1)))
    ld = LocalDensity(radii=[0.1, 0.2]).compute(locdata)
    assert len(ld.results) == 1_000
    print(ld.results.describe())
    ld.hist(bins=10)
    plt.show()
