"""

Utility functions for file input/output.

"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

__all__: list[str] = ["find_file_upstream"]

logger = logging.getLogger(__name__)


def find_file_upstream(
    sub_directory: str | os.PathLike[str],
    pattern: str | None,
    regex: str | None = None,
    top_directory: str | os.PathLike[str] | None = None,
) -> Path | None:
    """
    Search for first upstream parent of sub_directory that contains pattern.
    Return first pattern found.
    Return None if no pattern has been found when parent equals directory.

    Parameters
    ----------
    sub_directory :
        Directory or file path to start with.
    pattern :
        glob pattern passed to :func:`Path.glob`
    regex :
        regex pattern passed to :func:`re.search` and applied in addition
        to glob pattern
    top_directory :
        Directory in which to stop the search.

    Returns
    -------
    Path | None
    """
    sub_directory = Path(sub_directory).resolve(strict=True)

    if pattern is None:
        pattern = "*.*"

    if top_directory is None:
        top_directory = sub_directory.anchor

    if regex is not None:
        regex_ = re.compile(regex)
    else:
        regex_ = None

    for parent in sub_directory.parents:
        file_list = list(parent.glob(pattern))

        if regex_ is not None:
            file_list = [
                file_ for file_ in file_list if regex_.search(str(file_)) is not None
            ]
        if file_list:
            return file_list[0]
        if parent == top_directory:
            return None
    return None
