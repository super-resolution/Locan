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


class DataFrame(Protocol):
    """Dataframe that supports the dataframe interchange protocol."""

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> DataFrame:
        ...


class LocData(Protocol):
    meta: Any
    references: Any

    @property
    def data(self) -> Any:
        ...

    def __len__(self) -> int:
        ...
