import matplotlib.pyplot as plt  # this import is needed for visual inspection
import pandas as pd
import pytest

from locan import LocData, PairDistances, simulate_uniform

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


class TestPairDistances:

    def test_init_empty(self, caplog):
        pd = PairDistances().compute(LocData())
        pd.hist()
        assert caplog.record_tuples == [
            ("locan.analysis.pair_distances", 30, "Locdata is empty.")
        ]
        plt.close("all")

    def test_PairDistances(self, locdata_simple):
        pd = PairDistances()
        assert repr(pd) == "PairDistances()"
        pd.compute(locdata_simple)
        assert pd.dimension == 2
        # print(pd.results)
        assert len(pd.results) == 10
        assert all(pd.results.columns == ["pair_distance"])
        assert pd.results["pair_distance"].iloc[0] == 1
        pd.hist(bins=10)
        # plt.show()

        plt.close("all")

    def test_PairDistances_other(self, locdata_simple, other_locdata_simple):
        pd = PairDistances().compute(locdata_simple, other_locdata=other_locdata_simple)
        assert len(pd.results) == 10

    def test_PairDistances_3D(self, locdata_3d):
        pd = PairDistances()
        assert repr(pd) == "PairDistances()"
        pd.compute(locdata_3d)
        assert pd.dimension == 3
        # print(pd.results)
        assert len(pd.results) == 15
        pd.hist(bins=10)
        # plt.show()

        plt.close("all")


@pytest.mark.visual
def test_PairDistances_2d_random():
    locdata = simulate_uniform(n_samples=1_000, region=((0, 100), (0, 100)))
    pd = PairDistances().compute(locdata)
    pd.hist(bins=10)
    plt.show()


@pytest.mark.visual
def test_PairDistances_3d_random():
    locdata = simulate_uniform(n_samples=1_000, region=((0, 1), (0, 1), (0, 1)))
    pd = PairDistances().compute(locdata)
    pd.hist(bins=10)
    plt.show()
