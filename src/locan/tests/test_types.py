import numpy as np

from locan import LocData
from locan.locan_types import LocData as LocDataProtocol
from locan.locan_types import RandomGeneratorSeed


def test_types() -> None:
    seed: RandomGeneratorSeed = np.array([1, 2])  # type: ignore
    # raises mypy assignment error
    assert isinstance(seed, np.ndarray)

    seed = [1, 2]
    assert isinstance(seed, list)


def test_locdata() -> None:
    locdata: LocDataProtocol = LocData()
    # LocDataProtocol is not runtime checkable:
    # assert isinstance(locdata, LocDataProtocol)
    assert isinstance(locdata, LocData)
