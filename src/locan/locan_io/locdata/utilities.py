"""

Utility functions for file input/output of localization data.

"""
from __future__ import annotations

import io
import logging
import os
from collections.abc import Iterable, Mapping
from contextlib import closing
from typing import TYPE_CHECKING, Any, Union, cast

if TYPE_CHECKING:
    from _typeshed import SupportsRead

import pandas as pd

from locan.constants import PROPERTY_KEYS

__all__: list[str] = ["convert_property_types", "convert_property_names"]

logger = logging.getLogger(__name__)


def convert_property_types(
    dataframe: pd.DataFrame,
    types: Mapping[str, str | type],
    loc_properties: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Convert data types according to the column-type mapping in types.
    If the target type is one of 'integer', 'signed', 'unsigned', 'float'
    then :func:`pandas.to_numeric` will be applied.
    Otherwise, if the target type is any type object like `int`, `str`,
    `np.float64` or similar then :func:`pandas.astype` will be applied.

    Parameters
    ----------
    dataframe
        Data to be converted
    types
        Mapping of loc_properties to types
    loc_properties
        The columns in dataframe to be converted.
        If None, all columns will be converted according to types.

    Returns
    -------
    pandas.DataFrame
        A copy of dataframe with converted types
    """
    new_df = dataframe.copy()
    if loc_properties is None:
        loc_properties = dataframe.columns
    else:
        loc_properties = [key for key in loc_properties if key in dataframe.columns]
    selected_property_keys = {key: types[key] for key in loc_properties if key in types}

    for key, value in selected_property_keys.items():
        if isinstance(value, str) and value in [
            "integer",
            "signed",
            "unsigned",
            "float",
        ]:
            new_df[key] = pd.to_numeric(dataframe[key], downcast=value, errors="coerce")  # type: ignore
        else:
            new_df[key] = dataframe[key].astype(value)  # type: ignore

    return new_df


def open_path_or_file_like(
    path_or_file_like: str | bytes | os.PathLike[Any] | int | SupportsRead[Any],
    mode: str = "r",
    encoding: str | None = None,
) -> Any:
    """
    Provide open-file context from `path_or_file_like` input.

    Parameters
    ----------
    path_or_file_like
        Identifier for file
    mode
        same as in `open()`
    encoding
        same as in `open()`

    Returns
    -------
    context for file object
    """
    try:
        all(getattr(path_or_file_like, attr) for attr in ("seek", "read", "close"))
        file = path_or_file_like
    except (AttributeError, io.UnsupportedOperation):
        try:
            # if hasattr(path_or_file_like, "__fspath__")
            # or isinstance(path_or_file_like, (str, bytes)):
            path_or_file_like = cast(
                Union[str, bytes, os.PathLike[str], os.PathLike[bytes], int],
                path_or_file_like,
            )

            file = open(
                path_or_file_like,
                mode=mode,
                encoding=encoding,
            )
        except TypeError as exc:
            raise TypeError(
                "path_or_file_like must be str, bytes, os.PathLike or file-like."
            ) from exc
    return closing(file)


def convert_property_names(
    properties: Iterable[str],
    property_mapping: dict[str, str] | Iterable[dict[str, str]] | None = None,
) -> list[str]:
    """
    Convert property names to standard locan property names if a mapping is
    provided.
    Otherwise, leave the property name as is and throw a warning.

    Parameters
    ----------
    properties
        Properties to be converted
    property_mapping
        Mappings between other property names and locan property names

    Returns
    -------
    list[str]
        Converted property names
    """
    property_mapping_: dict[str, str] = {}
    if property_mapping is None:
        pass
    elif isinstance(property_mapping, dict):
        property_mapping_ = property_mapping
    elif isinstance(property_mapping, Iterable):
        for mapping in property_mapping:
            property_mapping_.update(mapping)  # type: ignore[arg-type]
    else:
        raise TypeError(f"property_mapping has wrong type {type(property_mapping)}")

    column_keys = []
    for i in properties:
        if i in PROPERTY_KEYS.keys():
            column_keys.append(i)
        elif i in property_mapping_:
            column_keys.append(property_mapping_[i])

        else:
            logger.warning(f"Column {i} is not a Locan property standard.")
            column_keys.append(i)

    return column_keys
