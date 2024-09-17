"""
Benchmark functions for :mod:`locan.process.aggregate`
"""

import numpy as np

import locan as lc
from locan.process.aggregate import (
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


def main():
    bm = BenchmarkHistogram()
    bm.setup()
    bm.time__histogram_fast_histogram()
    bm.time__histogram_boost_histogram()
    bm.time__histogram_mean_fast_histogram()
    bm.time__histogram_mean_boost_histogram()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkHistogram()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        bm.time__histogram_fast_histogram()
        bm.time__histogram_boost_histogram()
        bm.time__histogram_mean_fast_histogram()
        bm.time__histogram_mean_boost_histogram()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")


if __name__ == "__main__":
    # main()
    main_profile()
