"""

File input/output for localization data in DECODE files.

"""
import logging

import pandas as pd

from locan.dependencies import HAS_DEPENDENCY, needs_package
from locan.data.locdata import LocData
import locan.constants
from locan.data import metadata_pb2
from locan.locan_io.locdata.utilities import convert_property_types, convert_property_names

if HAS_DEPENDENCY["h5py"]: import h5py


__all__ = ['read_decode_header', 'load_decode_header', 'load_decode_file']

logger = logging.getLogger(__name__)


@needs_package("h5py")
def read_decode_header(file):
    """
    Read header from a DECODE single-molecule localization file and identify column names.

    Parameters
    ----------
    file : file-like
        File to read.

    Returns
    -------
    Tuple[list[str], Dict, Dict]
        Tuple with identifiers, meta and decode sections.
        Identifiers are list of valid dataset property keys as derived from the DECODE identifiers.
    """
    meta_data = dict(file['meta'].attrs)
    meta_decode = dict(file['decode'].attrs)

    # list identifiers
    identifiers = list(file['data'].keys())

    column_keys = []
    for i in identifiers:
        if i == "xyz":
            column_keys.extend(["position_x", "position_y", "position_z"])
        elif i == "xyz_cr":
            column_keys.extend(["x_cr", "y_cr", "z_cr"])
        elif i == "xyz_sig":
            column_keys.extend(["x_sig", "y_sig", "z_sig"])
        else:
            column_keys.append(i)

    column_keys = convert_property_names(properties=column_keys, property_mapping=locan.constants.DECODE_KEYS)

    return column_keys, meta_data, meta_decode


@needs_package("h5py")
def load_decode_header(path):
    """
    Load header from a DECODE single-molecule localization file and identify column names.

    The hdf5 file should contain the following keys: <KeysViewHDF5 ['data', 'decode', 'meta']>

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a DECODE file to load.

    Returns
    -------
    Tuple[list[str], Dict, Dict]
        Tuple with identifiers, meta and decode sections.
        Identifiers are list of valid dataset property keys as derived from the DECODE identifiers.
    """
    with h5py.File(path, 'r') as file:
        return read_decode_header(file)


@needs_package("h5py")
def load_decode_file(path, nrows=None, convert=True):
    """
    Load data from a DECODE single-molecule localization file.

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
        columns, meta, decode = read_decode_header(file)

        if file['data']['xyz'].shape == (0, 3):  # empty file
            logger.warning(f'File does not contain any data.')
            locdata = LocData()

        else:  # file not empty
            data = {}
            for key, value in file['data'].items():
                if value.shape is not None:
                    if key == "xyz":
                        for i, property_ in enumerate(['position_x', 'position_y', 'position_z']):
                            data[property_] = value[:nrows, i]
                    elif key == "xyz_cr":
                        for i, property_ in enumerate(["x_cr", "y_cr", "z_cr"]):
                            data[property_] = value[:nrows, i]
                    elif key == "xyz_sig":
                        for i, property_ in enumerate(["x_sig", "y_sig", "z_sig"]):
                            data[property_] = value[:nrows, i]
                    elif key in locan.constants.DECODE_KEYS:
                        data[locan.constants.DECODE_KEYS[key]] = value[:nrows]
                    else:
                        data[key] = value[:nrows]

            dataframe = pd.DataFrame(data)

            if convert:
                dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)

            locdata = LocData.from_dataframe(dataframe=dataframe)

    locdata.meta.source = metadata_pb2.EXPERIMENT
    locdata.meta.state = metadata_pb2.RAW
    locdata.meta.file.type = metadata_pb2.DECODE
    locdata.meta.file.path = str(path)

    for property_ in sorted(list(set(columns).intersection({'position_x', 'position_y', 'position_z'}))):
        locdata.meta.units[property_] = meta['xy_unit']

    del locdata.meta.history[:]
    locdata.meta.history.add(name='load_decode_file', parameter='path={}, nrows={}'.format(path, nrows))

    return locdata
