"""

Validate localization data.

This module provides functions to validate LocData properties.

"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import pandas as pd

from locan.data.locdata import LocData

__all__: list[str] = []


class Collection(Protocol):
    data: pd.DataFrame
    references: Iterable[LocData]
    coordinate_keys: list[str]

    def __len__(self) -> int:
        ...


def _check_loc_properties(
    locdata: LocData | Collection, loc_properties: str | Iterable[str] | None = None
) -> list[str]:
    """
    Check that loc_properties are valid properties in locdata.

    Parameters
    ----------
    locdata
        Localization data
    loc_properties
        LocData property names. If None coordinate labels are used.

    Returns
    -------
    list[str]
        Valid localization property names
    """
    if loc_properties is None:  # use coordinate_labels
        labels = locdata.coordinate_keys.copy()
    elif isinstance(loc_properties, str):
        if loc_properties not in locdata.data.columns:
            raise ValueError(
                f"{loc_properties} is not a valid property in locdata.data."
            )
        labels = [loc_properties]
    elif isinstance(loc_properties, (tuple, list)):
        labels = list(loc_properties)
        for loc_property in loc_properties:
            if loc_property not in locdata.data.columns:
                raise ValueError(
                    f"{loc_property} is not a valid property in locdata.data."
                )
    else:
        raise ValueError(f"{loc_properties} is not a valid property in locdata.data.")
    return labels
