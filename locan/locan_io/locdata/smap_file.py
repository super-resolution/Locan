"""

File input/output for localization data in SMAP files.

"""
import logging

import pandas as pd

from locan.dependencies import HAS_DEPENDENCY, needs_package
from locan.data.locdata import LocData
import locan.constants
from locan.data import metadata_pb2
from locan.locan_io.locdata.io_locdata import convert_property_types

if HAS_DEPENDENCY["h5py"]: import h5py


__all__ = ['read_SMAP_header', 'load_SMAP_header', 'load_SMAP_file']

logger = logging.getLogger(__name__)


@needs_package("h5py")
def read_SMAP_header(file):
    """
    Identify column names from a SMAP single-molecule localization file.

    Parameters
    ----------
    file : file-like
        File to read.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
    """
    # list identifiers
    identifiers = list(file["saveloc"]["loc"].keys())

    column_keys = []
    for i in identifiers:
        if i in locan.constants.SMAP_KEYS:
            column_keys.append(locan.constants.SMAP_KEYS[i])
        else:
            logger.warning(f'Column {i} is not a Locan property standard.')
            column_keys.append(i)

    return column_keys


@needs_package("h5py")
def load_SMAP_header(path):
    """
    Identify column names from a SMAP single-molecule localization file.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
    """
    with h5py.File(path, 'r') as file:
        return read_SMAP_header(file)


@needs_package("h5py")
def load_SMAP_file(path, nrows=None, convert=True):
    """
    Load data from a SMAP single-molecule localization file.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a Thunderstorm file to load.
    nrows : int, None
        The number of localizations to load from file. None means that all available rows are loaded.
    convert : bool
        If True convert types by applying type specifications in locan.constants.PROPERTY_KEYS.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    with h5py.File(path, 'r') as file:
        columns = read_SMAP_header(file)

        if file["saveloc"]["loc"]["frame"].shape == (0,):  # empty file
            logger.warning(f'File does not contain any data.')
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
                dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)

            locdata = LocData.from_dataframe(dataframe=dataframe)

    locdata.meta.source = metadata_pb2.EXPERIMENT
    locdata.meta.state = metadata_pb2.RAW
    locdata.meta.file_type = metadata_pb2.SMAP
    locdata.meta.file_path = str(path)

    for property_ in sorted(list(set(columns).intersection({'position_x', 'position_y', 'position_z'}))):
        locdata.meta.unit.add(property=property_, unit='nm')

    del locdata.meta.history[:]
    locdata.meta.history.add(name='load_SMAP_file', parameter='path={}, nrows={}'.format(path, nrows))

    return locdata
