"""
Benchmark functions for :func:`locan.data.hull`
"""

import locan as lc
from locan.data.hulls import AlphaShape, BoundingBox, ConvexHull, OrientedBoundingBox


class BenchmarkLocDataHulls:
    """
    Benchmarks for LocData hull computation
    """

    def setup(self):
        path = lc.ROOT_DIR / "tests/test_data/five_blobs.txt"
        self.locdata = lc.load_txt_file(path)
        # print(self.locdata.data.info())

    def check_bounding_box(self):
        bb = BoundingBox(points=self.locdata.coordinates)
        print(bb.region_measure)

    def time_bounding_box(self):
        BoundingBox(points=self.locdata.coordinates)

    def check_oriented_bounding_box(self):
        bb = OrientedBoundingBox(points=self.locdata.coordinates)
        print(bb.region_measure)

    def time_oriented_bounding_box(self):
        OrientedBoundingBox(points=self.locdata.coordinates)

    def check_convex_hull(self):
        bb = ConvexHull(points=self.locdata.coordinates)
        print(bb.region_measure)

    def time_convex_hull_scipy(self):
        ConvexHull(points=self.locdata.coordinates, method="scipy")

    def time_convex_hull_shapely(self):
        ConvexHull(points=self.locdata.coordinates, method="shapely")

    def check_alpha_shape(self):
        bb = AlphaShape(points=self.locdata.coordinates, alpha=100)
        print(bb.region_measure)

    def time_alpha_shape(self):
        AlphaShape(points=self.locdata.coordinates, alpha=100)


def main():
    bm = BenchmarkLocDataHulls()
    bm.setup()

    bm.check_bounding_box()
    bm.check_oriented_bounding_box()
    bm.check_convex_hull()
    bm.check_alpha_shape()

    bm.time_bounding_box()
    bm.time_oriented_bounding_box()
    bm.time_convex_hull_scipy()
    bm.time_convex_hull_shapely()
    bm.time_alpha_shape()


def main_profile():
    """
    Run benchmarks through profiler.
    """
    import cProfile
    import pstats

    bm = BenchmarkLocDataHulls()
    bm.setup()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        bm.time_bounding_box()
        bm.time_oriented_bounding_box()
        bm.time_convex_hull_scipy()
        bm.time_convex_hull_shapely()
        bm.time_alpha_shape()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats("time_")
    stats.print_stats()


if __name__ == "__main__":
    # main()
    main_profile()
