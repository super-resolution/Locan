"""

Type definitions

"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

RandomGeneratorSeed = Union[
    None,
    int,
    Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]
