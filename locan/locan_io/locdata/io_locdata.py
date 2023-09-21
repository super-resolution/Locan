"""

File input/output for localization data.

"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import SupportsRead

import pandas as pd

import locan.constants
from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.locan_io.locdata.asdf_io import load_asdf_file
from locan.locan_io.locdata.decode_io import load_decode_file
from locan.locan_io.locdata.elyra_io import load_Elyra_file
from locan.locan_io.locdata.nanoimager_io import load_Nanoimager_file
from locan.locan_io.locdata.rapidstorm_io import (
    load_rapidSTORM_file,
    load_rapidSTORM_track_file,
)
from locan.locan_io.locdata.smap_io import load_SMAP_file
from locan.locan_io.locdata.smlm_io import load_SMLM_file
from locan.locan_io.locdata.thunderstorm_io import load_thunderstorm_file
from locan.locan_io.locdata.utilities import (
    convert_property_names,
    convert_property_types,
)

__all__: list[str] = ["load_txt_file", "load_locdata"]

logger = logging.getLogger(__name__)


def load_txt_file(
    path: str | os.PathLike[str] | SupportsRead[Any],
    sep: str = ",",
    columns: list[str] | None = None,
    nrows: int | None = None,
    property_mapping: dict[str, str] | list[dict[str, str]] | None = None,
    convert: bool = True,
    **kwargs: Any,
) -> LocData:
    """
    Load localization data from a txt file.

    Locan column names are either supplied or read from the first line header.

    Parameters
    ----------
    path
        File path for a localization file to load.
    sep
        separator between column values (Default: ',')
    columns
        Locan column names. If None the first line is interpreted as header
        (Default: None).
    nrows
        The number of localizations to load from file. None means that all
        available rows are loaded (Default: None).
    property_mapping
        Mappings between column names and locan property names
    convert
        If True convert types by applying type specifications in
        locan.constants.PROPERTY_KEYS.
    kwargs
        Other parameters passed to `pandas.read_csv()`.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    # define columns
    if columns is None:
        dataframe: pd.DataFrame = pd.read_csv(  # type:ignore[assignment]
            path,  # type:ignore[arg-type]
            sep=sep,
            nrows=nrows,
            **dict(dict(skiprows=0), **kwargs),
        )

        column_keys = convert_property_names(
            dataframe.columns, property_mapping=property_mapping
        )
        dataframe.columns = column_keys  # type: ignore[assignment]
    else:
        column_keys = convert_property_names(columns, property_mapping=property_mapping)
        dataframe = pd.read_csv(  # type:ignore[assignment]
            path,  # type:ignore[arg-type]
            sep=sep,
            nrows=nrows,
            **dict(dict(skiprows=1, names=column_keys), **kwargs),
        )

    if convert:
        dataframe = convert_property_types(
            dataframe, types=locan.constants.PROPERTY_KEYS
        )

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file.type = metadata_pb2.CUSTOM
    dat.meta.file.path = str(path)

    del dat.meta.history[:]
    dat.meta.history.add(
        name="load_txt_file",
        parameter=f"path={path}, sep={sep}, columns={columns}, nrows={nrows}",
    )

    return dat


def _map_file_type_to_load_function(
    file_type: int | str | locan.constants.FileType | locan.data.metadata_pb2.Metadata,
) -> Callable[..., Any]:
    """
    Interpret user input for file_type.

    Parameters
    ----------
    file_type
        Identifier for the file type. Integer or string should be according to
        locan.constants.FileType.

    Returns
    -------
    Callable[..., Any]
        Name of function for loading the localization file of `type`.
    """
    look_up_table = dict(
        load_txt_file=load_txt_file,
        load_rapidSTORM_file=load_rapidSTORM_file,
        load_Elyra_file=load_Elyra_file,
        load_thunderstorm_file=load_thunderstorm_file,
        load_asdf_file=load_asdf_file,
        load_Nanoimager_file=load_Nanoimager_file,
        load_rapidSTORM_track_file=load_rapidSTORM_track_file,
        load_SMLM_file=load_SMLM_file,
        load_decode_file=load_decode_file,
        load_SMAP_file=load_SMAP_file,
    )

    class LoadFunction(Enum):
        load_txt_file = 1
        load_rapidSTORM_file = 2
        load_Elyra_file = 3
        load_thunderstorm_file = 4
        load_asdf_file = 5
        load_Nanoimager_file = 6
        load_rapidSTORM_track_file = 7
        load_SMLM_file = 8
        load_decode_file = 9
        load_SMAP_file = 10

    try:
        if isinstance(file_type, int):
            function_name = LoadFunction(file_type).name
        elif isinstance(file_type, str):
            function_name = LoadFunction(
                locan.constants.FileType[file_type.upper()].value
            ).name
        elif isinstance(file_type, locan.constants.FileType):
            function_name = LoadFunction(file_type.value).name
        elif isinstance(file_type, metadata_pb2):  # type: ignore
            function_name = LoadFunction(file_type).name
        else:
            raise TypeError
        return look_up_table[function_name]  # type: ignore
    except ValueError as exc:
        raise ValueError(f"There is no load function for type {file_type}.") from exc


def load_locdata(
    path: str | os.PathLike[Any] | SupportsRead[Any],
    file_type: int
    | str
    | locan.constants.FileType
    | locan.data.metadata_pb2.Metadata = 1,
    nrows: int | None = None,
    **kwargs: Any,
) -> LocData:
    """
    Load data from localization file as specified by type.

    This function is a wrapper for read functions for the various types of SMLM data.

    Parameters
    ----------
    path
        File path for a localization data file to load.
    file_type
        Indicator for the file type.
        Integer or string should be according to locan.constants.FileType.
    nrows
        The number of localizations to load from file. None means that all
        available rows are loaded.
    kwargs
        kwargs passed to the specific load function.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    return_value: LocData = _map_file_type_to_load_function(file_type)(
        path=path, nrows=nrows, **kwargs
    )
    return return_value
