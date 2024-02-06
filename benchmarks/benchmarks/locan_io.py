"""
Benchmark functions for :mod:`locan.locan_io`
"""

import locan as lc


class LoadFiles:
    """
    Benchmarks for loading data from files
    """

    def setup(self):
        pass

    def time_load_asdf_file(self):
        file_path = lc.ROOT_DIR / "tests/test_data/npc_gp210.asdf"
        self.locdata = lc.load_asdf_file(path=file_path)

    def time_load_SMLM_file(self):
        file_path = lc.ROOT_DIR / "tests/test_data/SMLM_dstorm_data.smlm"
        self.locdata = lc.load_SMLM_file(path=file_path)

    def time_load_SMAP_file(self):
        file_path = lc.ROOT_DIR / "tests/test_data/smap_dstorm_data.mat"
        self.locdata = lc.load_SMAP_file(path=file_path)

    def time_load_txt_file(self):
        file_path = lc.ROOT_DIR / "tests/test_data/five_blobs_3D.txt"
        self.locdata = lc.load_txt_file(path=file_path)

    def time_load_decode_file(self):
        file_path = lc.ROOT_DIR / "tests/test_data/decode_dstorm_data.h5"
        self.locdata = lc.load_decode_file(path=file_path)


def main():
    bm = LoadFiles()
    bm.setup()
    bm.time_load_asdf_file()
    bm.time_load_SMLM_file()
    bm.time_load_SMAP_file()
    bm.time_load_txt_file()
    bm.time_load_decode_file()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = LoadFiles()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(5):
        bm.time_load_asdf_file()
        bm.time_load_SMLM_file()
        bm.time_load_SMAP_file()
        bm.time_load_txt_file()
        bm.time_load_decode_file()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")


if __name__ == "__main__":
    # main()
    main_profile()
