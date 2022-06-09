"""

File input/output for localization data in Thunderstorm files.

"""
from __future__ import annotations

import logging

import pandas as pd

import locan.constants
from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.locan_io.locdata.utilities import (
    convert_property_names,
    convert_property_types,
    open_path_or_file_like,
)

__all__ = [
    "load_thunderstorm_header",
    "load_thunderstorm_file",
    "save_thunderstorm_csv",
]

logger = logging.getLogger(__name__)


def _read_thunderstorm_header(file):
    """
    Read csv header from a Thunderstorm single-molecule localization file and identify column names.

    Parameters
    ----------
    file : file-like
        Thunderstorm file to read.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the Thunderstorm identifiers.
    """
    # read csv header
    header = file.readline().split("\n")[0]

    # list identifiers
    identifiers = [x.strip('"') for x in header.split(",")]

    column_keys = convert_property_names(
        properties=identifiers, property_mapping=locan.constants.THUNDERSTORM_KEYS
    )
    return column_keys


def load_thunderstorm_header(path):
    """
    Load csv header from a Thunderstorm single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str, bytes, os.PathLike, file-like
        File path for a Thunderstorm file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the Thunderstorm identifiers.
    """
    # read csv header
    with open_path_or_file_like(path) as file:
        return _read_thunderstorm_header(file)


def load_thunderstorm_file(path, nrows=None, convert=True, **kwargs):
    """
    Load data from a Thunderstorm single-molecule localization file.

    Parameters
    ----------
    path : str, bytes, os.PathLike, file-like
        File path for a Thunderstorm file to load.
    nrows : int, None
        The number of localizations to load from file. None means that all available rows are loaded.
    convert : bool
        If True convert types by applying type specifications in locan.constants.PROPERTY_KEYS.
    kwargs : dict
        Other parameters passed to `pandas.read_csv()`.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    with open_path_or_file_like(path) as file:
        columns = _read_thunderstorm_header(file)
        dataframe = pd.read_csv(
            file, sep=",", skiprows=0, nrows=nrows, names=columns, **kwargs
        )

    if convert:
        dataframe = convert_property_types(
            dataframe, types=locan.constants.PROPERTY_KEYS
        )

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file.type = metadata_pb2.THUNDERSTORM
    dat.meta.file.path = str(path)

    for property_ in sorted(
        list(set(columns).intersection({"position_x", "position_y", "position_z"}))
    ):
        dat.meta.localization_properties.add(name=property_, unit="nm", type="float")

    del dat.meta.history[:]
    dat.meta.history.add(
        name="load_thundestorm_file", parameter="path={}, nrows={}".format(path, nrows)
    )

    return dat


def save_thunderstorm_csv(locdata, path):
    """
    Save LocData attributes Thunderstorm-readable csv-file.

    In the Thunderstorm csv-file file format we store only localization data with Thunderstorm-readable column names.

    Parameters
    ----------
    locdata : LocData
        The LocData object to be saved.
    path : str, bytes, os.PathLike, file-like
        File path including file name to save to.
    """
    # get data from locdata object
    dataframe = locdata.data

    # create reverse mapping to Thunderstorm columns
    inv_map = {v: k for k, v in locan.constants.THUNDERSTORM_KEYS.items()}

    # rename columns
    dataframe = dataframe.rename(index=str, columns=inv_map, inplace=False)

    # write to csv
    dataframe.to_csv(path, float_format="%.10g", index=False)
