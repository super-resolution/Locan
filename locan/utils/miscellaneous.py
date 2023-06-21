"""
Miscellaneous functions without a home.
"""
from __future__ import annotations

from collections.abc import Iterator

__all__: list[str] = ["iterate_2d_array"]


def iterate_2d_array(n_elements=5, n_cols=2) -> Iterator[tuple[int, int]]:
    """
    Iterator for 2-dimensional array iterating first over columns then over rows.

    Parameters
    ----------
    n_elements : int
        Number of elements
    n_cols : int
        Number of columns

    Returns
    -------
    Iterator[tuple[int, int]]
        Indices for (row, column) in each iteration.
    """
    n_rows = -(-n_elements // n_cols)
    iterator = ((i, j) for i in range(n_rows) for j in range(n_cols))
    return (item for item, _ in zip(iterator, range(n_elements)))


def _get_subclasses(cls) -> set:
    """
    Recursively identify all classes that inherit from `cls`.

    Returns
    -------
    set
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _get_subclasses(c)]
    )
