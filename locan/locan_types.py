"""

Type definitions

"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, Union

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
