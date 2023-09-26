"""

Provide standard formats.

"""
from __future__ import annotations

import time

__all__: list[str] = []


def _time_string(time_value: float) -> str:
    """
    Convert 'time_value' (typically timestamp from Unix epoch) to the local time
    and return as "%Y-%m-%d %H:%M:%S %z" formatted string.

    This time format is used in metadata of LocData and Analysis classes.

    Parameters
    ----------
    time_value
        Return value from :func:`time.time`

    Returns
    -------
    str
    """
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(time_value))
