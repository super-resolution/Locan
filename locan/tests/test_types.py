import numpy as np

from locan.locan_types import RandomGeneratorSeed


def test_types():
    seed: RandomGeneratorSeed = np.array([1, 2])
    print(type(seed))
