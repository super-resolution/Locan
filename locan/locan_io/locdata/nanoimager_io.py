"""

File input/output for localization data Nanoimager files.

"""
from __future__ import annotations
import logging

import pandas as pd

from locan.data.locdata import LocData
import locan.constants
from locan.data import metadata_pb2
from locan.locan_io.locdata.utilities import convert_property_types, open_path_or_file_like, convert_property_names


__all__ = ['load_Nanoimager_header', 'load_Nanoimager_file']

logger = logging.getLogger(__name__)


def _read_Nanoimager_header(file):
    """
    Read csv header from a Nanoimager single-molecule localization file and identify column names.

    Parameters
    ----------
    file : file-like
        Nanoimager file to read.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the Nanoimager identifiers.
    """
    # read csv header
    header = file.readline().split('\n')[0]

    # list identifiers
    identifiers = [x.strip('"') for x in header.split(',')]

    column_keys = convert_property_names(properties=identifiers, property_mapping=locan.constants.NANOIMAGER_KEYS)
    return column_keys


def load_Nanoimager_header(path):
    """
    Load csv header from a Nanoimager single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str, bytes, os.PathLike, file-like
        File path for a Nanoimager file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the Nanoimager identifiers.
    """
    # read csv header
    with open_path_or_file_like(path) as file:
        return _read_Nanoimager_header(file)


def load_Nanoimager_file(path, nrows=None, convert=True, **kwargs):
    """
    Load data from a Nanoimager single-molecule localization file.

    Parameters
    ----------
    path : str, bytes, os.PathLike, file-like
        File path for a Nanoimager file to load.
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
        columns = _read_Nanoimager_header(file)
        dataframe = pd.read_csv(file, sep=',', skiprows=0, nrows=nrows, names=columns, **kwargs)

    if convert:
        dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file.type = metadata_pb2.NANOIMAGER
    dat.meta.file.path = str(path)

    for property_ in sorted(list(set(columns).intersection({'position_x', 'position_y', 'position_z'}))):
        dat.meta.localization_properties.add(name=property_, unit="nm", type="float")

    del dat.meta.history[:]
    dat.meta.history.add(name='load_Nanoimager_file', parameter='path={}, nrows={}'.format(path, nrows))

    return dat
