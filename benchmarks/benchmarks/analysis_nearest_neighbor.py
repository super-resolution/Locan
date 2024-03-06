"""
Benchmark functions for :mod:`locan.analysis.nearest_neighbor`
"""

import locan as lc
from locan.analysis.nearest_neighbor import (
    _nearest_neighbor_distances,
)


class BenchmarkNearestNeighborDistances:
    """
    Benchmarks for nearest neighbor computations
    """

    def setup(self):
        path = lc.ROOT_DIR / "tests/test_data/five_blobs.txt"
        self.locdata = lc.load_txt_file(path)
        print(self.locdata.data.info())

    def check__nearest_neighbor_distances(self):
        results_df = _nearest_neighbor_distances(
            points=self.locdata.coordinates,
        )
        print(results_df)

    def time__nearest_neighbor_distances(self):
        _nearest_neighbor_distances(
            points=self.locdata.coordinates,
        )


def main():
    bm = BenchmarkNearestNeighborDistances()
    bm.setup()
    bm.check__nearest_neighbor_distances()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkNearestNeighborDistances()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(1000):
        bm.time__nearest_neighbor_distances()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")


if __name__ == "__main__":
    main()
    main_profile()
