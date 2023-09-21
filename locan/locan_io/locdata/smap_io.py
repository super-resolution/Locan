"""

File input/output for localization data in SMAP files.

"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite

import pandas as pd

import locan.constants
from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.dependencies import HAS_DEPENDENCY, needs_package
from locan.locan_io.locdata.utilities import (
    convert_property_names,
    convert_property_types,
)

if HAS_DEPENDENCY["h5py"]:
    import h5py


__all__: list[str] = ["load_SMAP_header", "load_SMAP_file", "save_SMAP_csv"]

logger = logging.getLogger(__name__)


@needs_package("h5py")
def _read_SMAP_header(file: Mapping[str, Any]) -> list[str]:
    """
    Identify column names from a SMAP single-molecule localization file.

    Parameters
    ----------
    file
        HDF5 file object.

    Returns
    -------
    list[str]
        A list of valid dataset property keys as derived from the rapidSTORM
        identifiers.
    """
    # list identifiers
    identifiers = list(file["saveloc"]["loc"].keys())
    column_keys = convert_property_names(
        properties=identifiers, property_mapping=locan.constants.SMAP_KEYS
    )
    return column_keys


@needs_package("h5py")
def load_SMAP_header(
    path: str | os.PathLike[Any] | SupportsRead[Any],
) -> list[str]:
    """
    Identify column names from a SMAP single-molecule localization file.

    Parameters
    ----------
    path
        File path for a file to load.

    Returns
    -------
    list[str]
        A list of valid dataset property keys as derived from the rapidSTORM
        identifiers.
    """
    with h5py.File(path, "r") as file:
        return _read_SMAP_header(file)


@needs_package("h5py")
def load_SMAP_file(
    path: str | os.PathLike[Any] | SupportsRead[Any],
    nrows: int | None = None,
    convert: bool = True,
) -> LocData:
    """
    Load data from a SMAP single-molecule localization file.

    Parameters
    ----------
    path
        File path for a Thunderstorm file to load.
    nrows
        The number of localizations to load from file. None means that all
        available rows are loaded.
    convert
        If True convert types by applying type specifications in
        locan.constants.PROPERTY_KEYS.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    with h5py.File(path, "r") as file:
        columns = _read_SMAP_header(file)

        if file["saveloc"]["loc"]["frame"].shape == (0,):  # empty file
            logger.warning("File does not contain any data.")
            locdata = LocData()

        else:  # file not empty
            data = {}
            for key, value in file["saveloc"]["loc"].items():
                if value.shape is not None:
                    if key in locan.constants.SMAP_KEYS:
                        data[locan.constants.SMAP_KEYS[key]] = list(value[0][:nrows])
                    else:
                        data[key] = list(value[0][:nrows])

            dataframe = pd.DataFrame(data)

            if convert:
                dataframe = convert_property_types(
                    dataframe, types=locan.constants.PROPERTY_KEYS
                )

            locdata = LocData.from_dataframe(dataframe=dataframe)

    locdata.meta.source = metadata_pb2.EXPERIMENT
    locdata.meta.state = metadata_pb2.RAW
    locdata.meta.file.type = metadata_pb2.SMAP
    locdata.meta.file.path = str(path)

    for property_ in sorted(
        list(set(columns).intersection({"position_x", "position_y", "position_z"}))
    ):
        locdata.meta.localization_properties.add(
            name=property_, unit="nm", type="float"
        )

    del locdata.meta.history[:]
    locdata.meta.history.add(
        name="load_SMAP_file", parameter=f"path={path}, nrows={nrows}"
    )

    return locdata


def save_SMAP_csv(
    locdata: LocData,
    path: str | os.PathLike[Any] | SupportsWrite[Any],
) -> None:
    """
    Save LocData to SMAP-readable csv-file.

    In the csv-file file format we store only localization data with
    SMAP-readable column names.

    Parameters
    ----------
    locdata
        The LocData object to be saved.
    path
        File path including file name to save to.
    """
    # get data from locdata object
    dataframe = locdata.data

    # create reverse mapping to columns
    inv_map = {v: k for k, v in locan.constants.SMAP_KEYS.items()}

    # rename columns
    dataframe = dataframe.rename(index=str, columns=inv_map, inplace=False)
    valid_smap_columns = [
        key for key in dataframe.columns if key in locan.constants.SMAP_KEYS
    ]

    # write to csv
    dataframe[valid_smap_columns].to_csv(path, float_format="%.10g", index=False)  # type: ignore[arg-type]
