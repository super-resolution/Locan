"""
Benchmark functions for :mod:`locan.analysis.localization_precision`
"""

import locan as lc
from locan.analysis.localization_precision import (
    _localization_precision,
)


class BenchmarkLocalizationPrecision:
    """
    Benchmarks for LocalizationPrecision computations
    """

    def setup(self):
        locdata = lc.load_npc()
        self.locdata = lc.select_by_condition(
            locdata,
            condition="4_000 < position_x < 6_000 and 4_000 < position_y < 6_000",
        )

    def check__localization_precision(self):
        results_df = _localization_precision(locdata=self.locdata)
        print(results_df)

    def time__localization_precision(self):
        _localization_precision(locdata=self.locdata, n_cpus=1)


def main():
    bm = BenchmarkLocalizationPrecision()
    bm.setup()
    bm.check__localization_precision()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkLocalizationPrecision()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):
        bm.time__localization_precision()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")


if __name__ == "__main__":
    main()
    main_profile()
