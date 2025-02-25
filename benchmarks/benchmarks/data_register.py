"""
Benchmark functions for :mod:`locan.process.register`
"""

import numpy as np

import locan as lc
from locan.process.register import (
    _register_cc_skimage,
    _register_icp_open3d,
    register_icp,
)
from locan.process.transform import transform_affine


class BenchmarkRegister:
    """
    Benchmarks for registering locdata
    """

    def setup(self):
        path = lc.ROOT_DIR / "tests/test_data/five_blobs.txt"
        self.locdata = lc.load_txt_file(path)

        self.offset_true = np.array([100.0, 50.0])
        self.matrix_true = np.array([[1, 0], [0, 1]])
        self.locdata_transformed = transform_affine(
            self.locdata, matrix=self.matrix_true, offset=self.offset_true
        )

    def time__register_cc_skimage(self):
        matrix, offset = _register_cc_skimage(
            locdata=self.locdata, other_locdata=self.locdata_transformed, bin_size=50
        )
        assert np.allclose(np.array(offset), self.offset_true, atol=5)
        assert np.allclose(matrix, self.matrix_true)

    def time__register_icp_open3d(self):
        matrix, offset = _register_icp_open3d(
            points=self.locdata.coordinates,
            other_points=self.locdata_transformed.coordinates,
            verbose=False,
        )
        assert np.allclose(np.array(offset), self.offset_true, atol=5)
        assert np.allclose(matrix, self.matrix_true)

    def time_register_icp(self):
        matrix, offset = register_icp(
            locdata=self.locdata, other_locdata=self.locdata_transformed, verbose=False
        )
        assert np.allclose(np.array(offset), self.offset_true, atol=5)
        assert np.allclose(matrix, self.matrix_true)


def main():
    bm = BenchmarkRegister()
    bm.setup()
    bm.time__register_cc_skimage()
    bm.time__register_icp_open3d()
    bm.time_register_icp()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkRegister()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        bm.time__register_cc_skimage()
        bm.time__register_icp_open3d()
        bm.time_register_icp()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")


if __name__ == "__main__":
    # main()
    main_profile()
