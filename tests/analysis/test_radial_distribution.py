import matplotlib.pyplot as plt  # this import is needed for visual inspection
import numpy as np
import pandas as pd
import pytest

from locan import (
    LocData,
    PairDistances,
    RadialDistribution,
    RadialDistributionBatch,
    RadialDistributionBatchResults,
    RadialDistributionResults,
    simulate_uniform,
)
from locan.analysis.radial_distribution import _radial_distribution_function

pytestmark = pytest.mark.slow


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


def test__radial_distribution_function():
    with pytest.raises(NotImplementedError):
        _radial_distribution_function(range(5), 4, 1, 1, 10)


class TestRadialDistributionResults:

    def test_init_empty(self):
        results = RadialDistributionResults()
        assert results


class TestRadialDistribution:

    def test_init_empty(self, caplog):
        rdb = RadialDistribution(bins=10)
        assert rdb.results is None

        rdf = RadialDistribution(bins=10).compute(LocData())
        assert rdf.results is None
        rdf.hist()
        assert caplog.record_tuples == [
            ("locan.analysis.radial_distribution", 30, "Locdata is empty.")
        ]

    def test_init(self, locdata_simple):
        rdf = RadialDistribution(bins=10)
        assert repr(rdf) == "RadialDistribution(bins=10, pair_distances=None)"
        rdf.compute(locdata_simple)
        assert rdf.dimension == 2
        assert isinstance(rdf.results, RadialDistributionResults)
        assert len(rdf.results.radii) == 10
        assert len(rdf.results.data) == 10
        assert all(rdf.results.data.columns == ["rdf"])
        rdf.hist()
        # plt.show()

        pd = PairDistances().compute(locdata_simple)
        rdf = RadialDistribution(bins=10, pair_distances=pd)
        assert (
            repr(rdf) == "RadialDistribution(bins=10, pair_distances=PairDistances())"
        )
        rdf.compute(locdata_simple)
        assert rdf.dimension == 2
        assert len(rdf.results.radii) == 10
        assert len(rdf.results.data) == 10
        rdf.hist()
        # plt.show()

        plt.close("all")

    def test_RadialDistribution_other(self, locdata_simple, other_locdata_simple):
        rdf = RadialDistribution(bins=10).compute(
            locdata_simple, other_locdata=other_locdata_simple
        )
        assert len(rdf.results.radii) == 10
        assert len(rdf.results.data) == 10

    def test_RadialDistribution_3D(self, locdata_3d):
        rdf = RadialDistribution(bins=10)
        rdf.compute(locdata_3d)
        assert rdf.dimension == 3
        assert len(rdf.results.radii) == 10
        assert len(rdf.results.data) == 10
        rdf.hist()
        # plt.show()

        plt.close("all")


@pytest.mark.visual
def test_RadialDistribution_2d_random():
    locdata = simulate_uniform(n_samples=1_000, region=((0, 100), (0, 100)))
    rdf = RadialDistribution(bins=10).compute(locdata)
    rdf.hist()
    plt.show()


@pytest.mark.visual
def test_RadialDistribution_3d_random():
    locdata = simulate_uniform(n_samples=1_000, region=((0, 1), (0, 1), (0, 1)))
    rdf = RadialDistribution(bins=10).compute(locdata)
    rdf.hist()
    plt.show()


class TestRadialDistributionBatchResults:

    def test_init_empty(self):
        results = RadialDistributionBatchResults()
        assert results


class TestRadialDistributionBatch:

    def test_init(self, locdata_2d):
        rdb = RadialDistributionBatch()
        assert rdb.results is None
        assert rdb.batch is None

        rdb = RadialDistributionBatch(bins=10, meta={"comment": "this is an example"})
        assert str(rdb) == "RadialDistributionBatch(bins=10)"
        assert rdb.results is None
        assert rdb.batch is None
        assert rdb.meta.comment == "this is an example"

    def test_compute_empty_locdata(self):
        with pytest.raises(ValueError):
            RadialDistributionBatch(bins=10).compute(locdatas=[LocData(), LocData()])

    def test_compute_empty_locdatas(self, caplog):
        with pytest.raises(ValueError):
            RadialDistributionBatch(bins=10).compute(locdatas=[])

    def test_from_batch_empty_item(self, caplog):
        with pytest.raises(ValueError):
            RadialDistributionBatch.from_batch(
                batch=[RadialDistribution(bins=10), RadialDistribution(bins=10)]
            )

    def test_from_batch_empty(self, caplog):
        with pytest.raises(ValueError):
            RadialDistributionBatch.from_batch(batch=[])

    def test_compute(self, locdata_simple):
        rdb = RadialDistributionBatch(bins=10).compute(
            locdatas=[locdata_simple, locdata_simple]
        )
        assert isinstance(rdb.results, RadialDistributionBatchResults)
        assert isinstance(rdb.results.radii, pd.DataFrame)
        assert isinstance(rdb.results.data, pd.DataFrame)
        assert len(rdb.results.data.index) == 10
        assert len(rdb.results.data.columns) == 2

        rdb.hist()
        # plt.show()

        plt.close("all")

    def test_from_batch(self, locdata_simple):
        rdf_0 = RadialDistribution(bins=10).compute(locdata_simple)
        rdf_1 = RadialDistribution(bins=10).compute(locdata_simple)

        rdb = RadialDistributionBatch().from_batch(batch=[rdf_0, rdf_1])
        assert isinstance(rdb.results, RadialDistributionBatchResults)
        assert len(rdb.results.data.index) == 10
        assert len(rdb.results.data.columns) == 2

    @pytest.mark.visual
    def test_from_batch_visual(self, locdata_simple):
        rng = np.random.default_rng(seed=1)
        locdatas = [
            simulate_uniform(n_samples=100, region=((0, 1), (0, 1), (0, 1)), seed=rng)
            for _ in range(5)
        ]
        bins = np.linspace(0, 2, 10)
        rdfs = [
            RadialDistribution(bins=bins).compute(locdata_) for locdata_ in locdatas
        ]
        rdb = RadialDistributionBatch().from_batch(batch=rdfs)

        rdb.hist()
        plt.show()

        plt.close("all")
