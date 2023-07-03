import numpy as np  # noqa: F401

from locan.locan_types import RandomGeneratorSeed


def test_types():
    seed: RandomGeneratorSeed = np.array([1, 2])  # type: ignore
    # raises mypy assignment error
    assert isinstance(seed, np.ndarray)

    seed: RandomGeneratorSeed = [1, 2]
    assert isinstance(seed, list)
