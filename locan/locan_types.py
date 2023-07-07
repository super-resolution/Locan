"""

Type definitions

"""
from __future__ import annotations

import sys
from typing import Any, Protocol, Union

if sys.version_info >= (3, 9):
    from collections.abc import Sequence  # noqa: F401
else:
    from typing import Sequence  # noqa: F401

import numpy as np

RandomGeneratorSeed = Union[
    None,
    int,
    Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]


class LocData(Protocol):
    data: Any
    meta: Any
    references: Any

    def __len__(self) -> int:
        ...
