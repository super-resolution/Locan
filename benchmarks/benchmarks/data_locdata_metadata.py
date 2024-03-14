"""
Benchmark functions for :mod:`locan.data.metadata_pb2`
"""

import locan as lc


class BenchmarkMetadata:
    """
    Benchmarks for data.metadata_pb2.Metadata objects
    """

    def setup(self):
        self.metadata = lc.data.metadata_pb2.Metadata()

    def time_initialize_metadata(self):
        lc.data.metadata_pb2.Metadata()

    def mem_metadata(self):
        return self.metadata


def main():
    bm = BenchmarkMetadata()
    bm.setup()
    bm.time_initialize_metadata()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkMetadata()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        bm.time_initialize_metadata()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")
    stats.print_stats()


if __name__ == "__main__":
    # main()
    main_profile()
