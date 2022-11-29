"""

Utility functions for file input/output.

"""
import logging
from pathlib import Path

__all__ = ["find_pattern_upstream"]

logger = logging.getLogger(__name__)


def find_pattern_upstream(sub_directory, pattern, directory=None):
    """
    Search for first upstream parent of sub_directory that contains pattern.
    Return first pattern found.
    Return None if no pattern has been found when parent equals directory.

    Parameters
    ----------
    sub_directory : str | bytes | os.PathLike
        Directory or file path to start with.
    pattern : str
        The same pattern as in :func:`Path.glob`.
    directory : str | bytes | os.PathLike
        Directory in which to stop the search.

    Returns
    -------
    Path | None
    """
    sub_directory = Path(sub_directory).resolve(strict=True)

    if directory is None:
        directory = sub_directory.anchor

    for parent in sub_directory.parents:
        file_list = list(parent.glob(pattern))
        if file_list:
            return file_list[0]
        if parent == directory:
            return None
