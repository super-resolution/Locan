"""
Benchmark functions for :mod:`locan.analysis.fourier_ring_correlation`
"""

import locan as lc
from locan.analysis.fourier_ring_correlation import (
    FourierRingCorrelation,
)


class BenchmarkFourierRingCorrelation:
    """
    Benchmarks for Fourier ring correlation computations
    """

    def setup(self):
        locdata = lc.load_npc()
        self.locdata = lc.select_by_condition(
            locdata,
            condition="4_000 < position_x < 6_000 and 4_000 < position_y < 6_000",
        )

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

    for _ in range(10):
        bm.time_FourierRingCorrelation()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")


if __name__ == "__main__":
    main()
    main_profile()
