"""
Miscellaneous functions without a home.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

__all__: list[str] = ["iterate_2d_array"]


def iterate_2d_array(n_elements: int = 5, n_cols: int = 2) -> Iterator[tuple[int, int]]:
    """
    Iterator for 2-dimensional array iterating first over columns then over rows.

    Parameters
    ----------
    n_elements
        Number of elements
    n_cols
        Number of columns

    Returns
    -------
    Iterator[tuple[int, int]]
        Indices for (row, column) in each iteration.
    """
    n_rows = -(-n_elements // n_cols)
    iterator = ((i, j) for i in range(n_rows) for j in range(n_cols))
    return (item for item, _ in zip(iterator, range(n_elements)))


def _get_subclasses(cls: Any) -> set[Any]:
    """
    Recursively identify all classes that inherit from `cls`.

    Parameters
    ----------
    cls
        class object

    Returns
    -------
    set[Any]
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _get_subclasses(c)]
    )
