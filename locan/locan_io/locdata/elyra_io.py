"""

File input/output for localization data in Elyra files.

"""
from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsReadline

import pandas as pd

import locan.constants
from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.locan_io.locdata.utilities import (
    convert_property_names,
    convert_property_types,
    open_path_or_file_like,
)

__all__: list[str] = ["load_Elyra_header", "load_Elyra_file"]

logger = logging.getLogger(__name__)


def _read_Elyra_header(file: SupportsReadline[Any]) -> list[str]:
    """
    Read xml header from a Zeiss Elyra single-molecule localization file and
    identify column names.

    Parameters
    ----------
    file
        A rapidSTORM file to load.

    Returns
    -------
    list[str]
        A list of valid dataset property keys as derived from the rapidSTORM
        identifiers.
    """
    header = file.readline().split("\n")[0]

    # list identifiers
    identifiers = header.split("\t")

    column_keys = convert_property_names(
        properties=identifiers, property_mapping=locan.constants.ELYRA_KEYS
    )

    return column_keys


def load_Elyra_header(
    path: str | os.PathLike[Any] | SupportsRead[Any],
) -> list[str]:
    """
    Load xml header from a Zeiss Elyra single-molecule localization file and
    identify column names.

    Parameters
    ----------
    path
        File path for a rapidSTORM file to load.

    Returns
    -------
    list[str]
        A list of valid dataset property keys as derived from the rapidSTORM
        identifiers.
    """

    with open_path_or_file_like(path, encoding="latin-1") as file:
        return _read_Elyra_header(file)


def load_Elyra_file(
    path: str | os.PathLike[Any] | SupportsRead[Any],
    nrows: int | None = None,
    convert: bool = True,
    **kwargs: Any,
) -> LocData:
    """
    Load data from a rapidSTORM single-molecule localization file.

    Parameters
    ----------
    path
        File path for a rapidSTORM file to load.
    nrows
        The number of localizations to load from file. None means that all
        available rows are loaded.
    convert
        If True convert types by applying type specifications in
        locan.constants.PROPERTY_KEYS.
    kwargs
        Other parameters passed to `pandas.read_csv()`.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.

    Note
    ----
    Data is loaded with encoding = 'latin-1' and only data before the first
    NUL character is returned.
    Additional information appended at the end of the file is thus ignored.
    """
    with open_path_or_file_like(path, encoding="latin-1") as file:
        columns = _read_Elyra_header(file)
        string = file.read()
        # remove metadata following nul byte
        string = string.split("\x00")[0]

        stream = io.StringIO(string)
        dataframe = pd.read_csv(
            stream, sep="\t", skiprows=0, nrows=nrows, names=columns, **kwargs
        )

    if convert:
        dataframe = convert_property_types(
            dataframe, types=locan.constants.PROPERTY_KEYS
        )

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file.type = metadata_pb2.ELYRA
    dat.meta.file.path = str(path)

    del dat.meta.history[:]
    dat.meta.history.add(
        name="load_Elyra_file",
        parameter=f"path={str(path)}, nrows={nrows}",
    )

    return dat
