"""
Benchmark functions to be used with Airspeed Velocity.
"""
import locan as lc
import numpy as np
from locan.data.aggregate import (
    _histogram_boost_histogram,
    _histogram_fast_histogram,
    _histogram_mean_boost_histogram,
    _histogram_mean_fast_histogram,
)


class BenchmarkBins:
    """
    Benchmarks for initializing Bins objects
    """

    def setup(self):
        self.bins = lc.Bins(n_bins=100, bin_range=(0, 1))

    def time_initialize_bins(self):
        """Time to initialize bins"""
        lc.Bins(n_bins=100, bin_range=(0, 1))

    def mem_bins(self):
        return self.bins


class BenchmarkHistogram:
    """
    Benchmarks for computing histograms
    """

    def setup(self):
        n_points = 1000
        self.points = np.vstack(
            [np.linspace(0, 1, n_points), np.linspace(0, 1, n_points)]
        )
        self.values = np.linspace(0, 1, n_points)
        self.bins = lc.Bins(n_bins=(100, 100), bin_range=(0, 1))

    def time__histogram_fast_histogram(self):
        _histogram_fast_histogram(data=self.points, bins=self.bins)

    def time__histogram_boost_histogram(self):
        _histogram_boost_histogram(data=self.points, bins=self.bins)

    def time__histogram_mean_fast_histogram(self):
        _histogram_mean_fast_histogram(
            data=self.points, bins=self.bins, values=self.values
        )

    def time__histogram_mean_boost_histogram(self):
        _histogram_mean_boost_histogram(
            data=self.points, bins=self.bins, values=self.values
        )


if __name__ == "__main__":
    class_ = BenchmarkHistogram()
    class_.setup()
    class_.time__histogram_fast_histogram()
