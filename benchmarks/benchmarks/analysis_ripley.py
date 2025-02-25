"""
Benchmark functions for :mod:`locan.analysis.ripley`
"""

import numpy as np

import locan as lc
from locan.analysis.ripley import (
    _ripleys_k_function,
)


class BenchmarkRipley:
    """
    Benchmarks for Ripley's k function computations
    """

    def setup(self):
        path = lc.ROOT_DIR / "tests/test_data/five_blobs.txt"
        self.locdata = lc.load_txt_file(path)
        print(self.locdata.data.info())
        self.radii = np.arange(10, 110, 10)
        print(self.radii)

    def check__ripleys_k_function(self):
        results_df = _ripleys_k_function(
            points=self.locdata.coordinates, radii=self.radii
        )
        print(results_df)

    def time__ripleys_k_function(self):
        _ripleys_k_function(points=self.locdata.coordinates, radii=self.radii)


def main():
    bm = BenchmarkRipley()
    bm.setup()
    bm.check__ripleys_k_function()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkRipley()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        bm.time__ripleys_k_function()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")


if __name__ == "__main__":
    main()
    main_profile()
