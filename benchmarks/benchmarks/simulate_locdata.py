"""
Benchmark functions to be used with Airspeed Velocity.
"""
import locan as lc
import numpy as np

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


if __name__ == "__main__":
    class_ = BenchmarkResample()
    class_.setup()
    print(class_.collection.data)
