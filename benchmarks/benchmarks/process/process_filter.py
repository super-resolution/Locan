"""
Benchmark functions for :mod:`locan.process.filter`
"""

import matplotlib.pyplot as plt
import numpy as np

import locan as lc
from locan.data import Image
from locan.process.filter import (
    select_by_image_mask,
)


class BenchmarkFilter:
    """
    Benchmarks for filtering locadata
    """

    def setup(self):
        path = lc.ROOT_DIR / "tests/test_data/five_blobs.txt"
        self.locdata = lc.load_txt_file(path)

        array = np.zeros(shape=(100, 100))
        array[:, 0:70] = 1

        self.image = Image.from_array(array=array, is_rgb=False)
        self.image.bins = lc.Bins(
            n_bins=self.image.data.shape, bin_range=((0, 1100), (0, 1100))
        )

    def check_select_by_image_mask(self):
        print(self.locdata.data.info())
        print(self.locdata.data.describe())
        lc.render_2d_mpl(locdata=self.locdata)
        plt.show()

        plt.imshow(self.image.data)
        plt.show()

        selection = select_by_image_mask(locdata=self.locdata, image=self.image)
        print(selection.data.describe())
        lc.render_2d_mpl(locdata=selection)
        plt.show()

    def time_select_by_image_mask(self):
        select_by_image_mask(locdata=self.locdata, image=self.image)


def main():
    bm = BenchmarkFilter()
    bm.setup()
    bm.check_select_by_image_mask()
    bm.time_select_by_image_mask()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkFilter()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(3):
        bm.time_select_by_image_mask()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats("time_")
    stats.print_stats()


if __name__ == "__main__":
    # main()
    main_profile()
