"""
Benchmark functions for :mod:`locan.analysis.fourier_ring_correlation`

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    0.002    0.000    0.768    0.008
"""

import locan as lc
from locan.analysis.fourier_ring_correlation import (
    FourierRingCorrelation,
)
from tests import TEST_DIR


class BenchmarkFourierRingCorrelation:
    """
    Benchmarks for Fourier ring correlation computations
    """

    def setup(self):
        path = TEST_DIR / "test_data/five_blobs.txt"
        self.locdata = lc.load_txt_file(path)

    def check_FourierRingCorrelation(self):
        frc = FourierRingCorrelation().compute(locdata=self.locdata)
        print(frc.results.correlation)

    def time_FourierRingCorrelation(self):
        FourierRingCorrelation().compute(locdata=self.locdata)


def main():
    bm = BenchmarkFourierRingCorrelation()
    bm.setup()
    bm.check_FourierRingCorrelation()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkFourierRingCorrelation()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        bm.time_FourierRingCorrelation()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")


if __name__ == "__main__":
    main()
    main_profile()
