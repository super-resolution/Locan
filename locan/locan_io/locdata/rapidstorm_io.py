"""

File input/output for localization data in rapidSTORM files.

"""
from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsReadline

import numpy as np
import pandas as pd

import locan.constants
from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.locan_io.locdata.utilities import (
    convert_property_names,
    convert_property_types,
    open_path_or_file_like,
)

__all__: list[str] = [
    "load_rapidSTORM_header",
    "load_rapidSTORM_file",
    "load_rapidSTORM_track_header",
    "load_rapidSTORM_track_file",
]

logger = logging.getLogger(__name__)


def _read_rapidSTORM_header(file: SupportsReadline[Any]) -> list[str]:
    """
    Read xml header from a rapidSTORM single-molecule localization file and
    identify column names.

    Parameters
    ----------
    file : SupportsReadline
        A rapidSTORM file to load.

    Returns
    -------
    list[str]
        A list of valid dataset property keys as derived from the rapidSTORM
        identifiers.
    """
    # read xml part in header
    header = file.readline()
    header = header[2:]

    # get iteratible
    parsed = etree.XML(header)

    # list identifiers
    identifiers = []
    for elem in parsed:
        for name, value in sorted(elem.attrib.items()):
            if name == "identifier":
                identifiers.append(value)

    # turn identifiers into valuable LocData keys
    column_keys = convert_property_names(
        properties=identifiers, property_mapping=locan.constants.RAPIDSTORM_KEYS
    )
    return column_keys


def load_rapidSTORM_header(
    path: str | os.PathLike[Any] | SupportsRead[Any],
) -> list[str]:
    """
    Load xml header from a rapidSTORM single-molecule localization file and
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

    # read xml part in header
    with open_path_or_file_like(path) as file:
        return _read_rapidSTORM_header(file)


def load_rapidSTORM_file(
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
    """
    with open_path_or_file_like(path) as file:
        columns = _read_rapidSTORM_header(file)
        dataframe = pd.read_csv(
            file, sep=" ", skiprows=0, nrows=nrows, names=columns, **kwargs
        )

    if convert:
        dataframe = convert_property_types(
            dataframe, types=locan.constants.PROPERTY_KEYS
        )

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file.type = metadata_pb2.RAPIDSTORM
    dat.meta.file.path = str(path)

    for property in sorted(
        list(set(columns).intersection({"position_x", "position_y", "position_z"}))
    ):
        dat.meta.localization_properties.add(name=property, unit="nm", type="float")

    del dat.meta.history[:]
    dat.meta.history.add(
        name="load_rapidSTORM_file", parameter=f"path={str(path)}, nrows={nrows}"
    )

    return dat


def _read_rapidSTORM_track_header(
    file: SupportsReadline[Any],
) -> tuple[list[str], list[str]]:
    """
    Read xml header from a rapidSTORM (track) single-molecule localization
    file and identify column names.

    Parameters
    ----------
    file
        A rapidSTORM file to load.

    Returns
    -------
    tuple[list[str], list[str]]
        A list of valid dataset property keys as derived from the rapidSTORM
        identifiers.
    """
    # read xml part in header
    header = file.readline()
    header = header[2:]

    parsed = etree.XML(header)

    # list identifiers
    identifiers: list[str] = []
    for field in parsed.findall("field"):
        next_identifier = field.get("identifier")
        if next_identifier is not None:
            identifiers.append(next_identifier)

    # turn identifiers into valuable LocData keys
    column_keys = convert_property_names(
        properties=identifiers, property_mapping=locan.constants.RAPIDSTORM_KEYS
    )

    # list child identifiers
    child_identifiers: list[str] = []
    for field in parsed.findall("localizations"):
        for field_ in field.findall("field"):
            next_child_identifiers = field_.get("identifier")
            if next_child_identifiers is not None:
                child_identifiers.append(next_child_identifiers)

    # turn child identifiers into valuable LocData keys
    column_keys_tracks = convert_property_names(
        properties=child_identifiers, property_mapping=locan.constants.RAPIDSTORM_KEYS
    )

    return column_keys, column_keys_tracks


def load_rapidSTORM_track_header(
    path: str | os.PathLike[Any] | SupportsRead[Any],
) -> tuple[list[str], list[str]]:
    """
    Load xml header from a rapidSTORM (track) single-molecule localization
    file and identify column names.

    Parameters
    ----------
    path
        File path for a rapidSTORM file to load.

    Returns
    -------
    tuple[list[str], list[str]]
        A list of valid dataset property keys as derived from the rapidSTORM
        identifiers.
    """

    # read xml part in header
    with open_path_or_file_like(path) as file:
        return _read_rapidSTORM_track_header(file)


def load_rapidSTORM_track_file(
    path: str | os.PathLike[Any] | SupportsRead[Any],
    nrows: int | None = None,
    convert: bool = True,
    collection: bool = True,
    min_localization_count: int = 1,
    **kwargs: Any,
) -> LocData:
    """
    Load data from a rapidSTORM single-molecule localization file with
    tracked localizations.

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
    collection
        If True a collection of all tracks is returned.
        If False LocData with center positions is returned.
    min_localization_count
        If collection is True, only clusters with at least
        `min_localization_count` localizations are loaded.
    kwargs
        Other parameters passed to `pandas.read_csv()`.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations/tracks as a
        collection.
    """
    with open_path_or_file_like(path) as file:
        columns, columns_track = _read_rapidSTORM_track_header(file)
        lines = pd.read_csv(
            file, lineterminator="\n", nrows=nrows, skiprows=1, header=None, **kwargs
        )

    lines = lines[0].str.split(" ", expand=False)

    if collection:
        # prepare dataframes with tracked localizations
        tracks = [
            np.array(line[len(columns) + 1 :]).reshape(-1, len(columns_track))
            for line in lines
            if int(line[len(columns)]) >= min_localization_count
        ]
        # +1 to account for the column with number of locs in track
        track_list = []
        for track in tracks:
            dataframe = pd.DataFrame(track, columns=columns_track)
            if convert:
                dataframe = convert_property_types(
                    dataframe, types=locan.constants.PROPERTY_KEYS
                )
            else:
                dataframe = dataframe.convert_dtypes()
            dat = LocData.from_dataframe(dataframe=dataframe)
            track_list.append(dat)

        new_collection = LocData.from_collection(track_list)

        new_collection.meta.source = metadata_pb2.EXPERIMENT
        new_collection.meta.state = metadata_pb2.RAW
        new_collection.meta.file.type = metadata_pb2.RAPIDSTORMTRACK
        new_collection.meta.file.path = str(path)

        for property in sorted(
            list(
                set(columns_track).intersection(
                    {"position_x", "position_y", "position_z"}
                )
            )
        ):
            new_collection.meta.localization_properties.add(
                name=property, unit="nm", type="float"
            )

        del new_collection.meta.history[:]
        new_collection.meta.history.add(
            name="load_rapidSTORM_track_file",
            parameter=f"path={str(path)}, nrows={nrows}",
        )

        return new_collection

    else:
        # prepare dataframe with center track positions
        dataframe = pd.DataFrame(
            [line[: len(columns)] for line in lines], columns=columns
        )
        if convert:
            dataframe = convert_property_types(
                dataframe, types=locan.constants.PROPERTY_KEYS
            )
        else:
            dataframe = dataframe.convert_dtypes()

        locdata = LocData.from_dataframe(dataframe=dataframe)

        locdata.meta.source = metadata_pb2.EXPERIMENT
        locdata.meta.state = metadata_pb2.RAW
        locdata.meta.file.type = metadata_pb2.RAPIDSTORM
        locdata.meta.file.path = str(path)

        for property in sorted(
            list(set(columns).intersection({"position_x", "position_y", "position_z"}))
        ):
            locdata.meta.localization_properties.add(
                name=property, unit="nm", type="float"
            )

        del locdata.meta.history[:]
        locdata.meta.history.add(
            name="load_rapidSTORM_track_file",
            parameter=f"path={str(path)}, nrows={nrows}",
        )

        return locdata
