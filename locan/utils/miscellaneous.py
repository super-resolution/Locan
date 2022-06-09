"""
Miscellaneous functions without a home.
"""
from __future__ import annotations

from collections.abc import Generator

__all__ = ["iterate_2d_array"]


def iterate_2d_array(n_elements=5, n_cols=2) -> Generator[tuple[int, int]]:
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
    Generator[tuple[int, int]]
        Indices for (row, column) in each iteration.
    """
    n_rows = -(-n_elements // n_cols)
    iterator = ((i, j) for i in range(n_rows) for j in range(n_cols))
    return (item for item, _ in zip(iterator, range(n_elements)))
