"""

Provide standard formats.

"""
import time


__all__ = []


def _time_string(time_value):
    """
    Convert 'time_value' (typically timestamp from Unix epoch) to the local time
    and return as "%Y-%m-%d %H:%M:%S %z" formatted string.

    This time format is used in metadata of LocData and Analysis classes.

    Parameters
    ----------
    time_value : float
        return value from time.time()

    Returns
    -------
    str
    """
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(time_value))