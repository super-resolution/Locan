"""
Benchmark functions for :func:`locan.data.LocData`
"""

from copy import deepcopy

import numpy as np

import locan as lc
from locan.data.locdata import LocData

rng = np.random.default_rng(seed=1)


class BenchmarkLocDataFromSelection:
    """
    Benchmarks for selecting LocData objects
    """

    def setup(self):
        path = lc.ROOT_DIR / "tests/test_data/five_blobs.txt"
        self.locdata = lc.load_txt_file(path)
        self.df = deepcopy(self.locdata.data)
        # print(self.locdata.data.info())

        n_selected_points = 40
        self.selection_indices = rng.integers(
            low=0, high=50, size=(10, n_selected_points)
        )

        self.selection = LocData.from_selection(
            locdata=self.locdata, indices=self.selection_indices[0].tolist()
        )

        self.locdatas = [
            LocData.from_selection(locdata=self.locdata, indices=idxs.tolist())
            for idxs in self.selection_indices
        ]
        self.collection = lc.LocData.from_collection(locdatas=self.locdatas)

    def time_initialize_locdata_from_dataframe(self):
        LocData.from_dataframe(dataframe=self.df)

    def time_locdata_selection(self):
        LocData.from_selection(
            locdata=self.locdata, indices=self.selection_indices[0].tolist()
        )

    def time_locdata_collection(self):
        lc.LocData.from_collection(locdatas=self.locdatas)

    def mem_empty_locdata(self):
        return lc.LocData()

    def mem_locdata_from_dataframe(self):
        return self.locdata

    def mem_locdata_selection(self):
        return self.selection

    def mem_locdata_selection_reduce(self):
        return self.selection.reduce()

    def mem_locdata_collection(self):
        return self.collection

    def mem_locdata_collection_reduce(self):
        return self.collection.reduce()


def main():
    bm = BenchmarkLocDataFromSelection()
    bm.setup()
    bm.time_initialize_locdata_from_dataframe()
    bm.time_locdata_selection()
    bm.time_locdata_collection()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkLocDataFromSelection()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        bm.time_initialize_locdata_from_dataframe()
        bm.time_locdata_selection()
        bm.time_locdata_collection()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")
    stats.print_stats()


if __name__ == "__main__":
    # main()
    main_profile()
