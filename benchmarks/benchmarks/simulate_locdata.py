"""
Benchmark functions for :mod:`locan.simulation`
"""

import numpy as np

import locan as lc

rng = np.random.default_rng(seed=1)


class BenchmarkResample:
    """
    Benchmarks for data.metadata_pb2.Metadata objects
    """

    def __init__(self):
        self.locdata = None
        self.new_locdata = None

    def setup(self):
        self.locdata = lc.simulate_uniform(n_samples=1000, seed=rng)
        self.locdata.data["uncertainty"] = 0.01
        self.locdata.data["other_property"] = 111

    def time_resample(self):
        self.new_locdata = lc.resample(self.locdata, n_samples=1000, seed=rng)

    def mem_locdata(self):
        return self.new_locdata


def main():
    bm = BenchmarkResample()
    bm.setup()
    print(bm.locdata.data.info())
    bm.time_resample()
    print(bm.new_locdata.data.info())


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkResample()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):
        bm.time_resample()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")


if __name__ == "__main__":
    main()
    main_profile()
