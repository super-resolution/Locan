import logging

import numpy as np
import pandas as pd
import xml.etree.ElementTree as etree

from locan.data.locdata import LocData
import locan.constants
from locan.data import metadata_pb2
from locan.io.locdata.utilities import convert_property_types, open_path_or_file_like


__all__ = ['load_rapidSTORM_file', 'load_rapidSTORM_track_file']

logger = logging.getLogger(__name__)


def read_rapidSTORM_header(file):
    """
    Read xml header from a rapidSTORM single-molecule localization file and identify column names.

    Parameters
    ----------
    file : file-like
        A rapidSTORM file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
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
            if name == 'identifier':
                identifiers.append(value)

    # turn identifiers into valuable LocData keys
    column_keys = []
    for i in identifiers:
        if i in locan.constants.RAPIDSTORM_KEYS:
            column_keys.append(locan.constants.RAPIDSTORM_KEYS[i])
        else:
            logger.warning(f'Column {i} is not a Locan property standard.')
            column_keys.append(i)
    return column_keys


def load_rapidSTORM_header(path):
    """
    Load xml header from a rapidSTORM single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a rapidSTORM file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
    """

    # read xml part in header
    with open_path_or_file_like(path) as file:
        return read_rapidSTORM_header(file)


def load_rapidSTORM_file(path, nrows=None, convert=True, **kwargs):
    """
    Load data from a rapidSTORM single-molecule localization file.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a rapidSTORM file to load.
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
        columns = read_rapidSTORM_header(file)
        dataframe = pd.read_csv(file, sep=" ", skiprows=0, nrows=nrows, names=columns, **kwargs)

    if convert:
        dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file_type = metadata_pb2.RAPIDSTORM
    dat.meta.file_path = str(path)

    for property in sorted(list(set(columns).intersection({'position_x', 'position_y', 'position_z'}))):
        dat.meta.unit.add(property=property, unit='nm')

    del dat.meta.history[:]
    dat.meta.history.add(name='load_rapidSTORM_file', parameter='path={}, nrows={}'.format(path, nrows))

    return dat


def read_rapidSTORM_track_header(file):
    """
    Read xml header from a rapidSTORM (track) single-molecule localization file and identify column names.

    Parameters
    ----------
    file : file-like
        A rapidSTORM file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
    """
    # read xml part in header
    header = file.readline()
    header = header[2:]

    parsed = etree.XML(header)

    # list identifiers
    identifiers = []
    for field in parsed.findall('field'):
        identifiers.append(field.get('identifier'))

    # turn identifiers into valuable LocData keys
    column_keys = []
    for i in identifiers:
        if i in locan.constants.RAPIDSTORM_KEYS:
            column_keys.append(locan.constants.RAPIDSTORM_KEYS[i])
        else:
            logger.warning(f'Column {i} is not a Locan property standard.')
            column_keys.append(i)

    # list child identifiers
    child_identifiers = []
    for field in parsed.findall('localizations'):
        for field_ in field.findall('field'):
            child_identifiers.append(field_.get('identifier'))

    # turn child identifiers into valuable LocData keys
    column_keys_tracks = []
    for i in child_identifiers:
        if i in locan.constants.RAPIDSTORM_KEYS:
            column_keys_tracks.append(locan.constants.RAPIDSTORM_KEYS[i])
        else:
            logger.warning(f'Column {i} is not a Locan property standard.')
            column_keys_tracks.append(i)

    return column_keys, column_keys_tracks


def load_rapidSTORM_track_header(path):
    """
    Load xml header from a rapidSTORM (track) single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a rapidSTORM file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
    """

    # read xml part in header
    with open_path_or_file_like(path) as file:
        return read_rapidSTORM_track_header(file)


def load_rapidSTORM_track_file(path, nrows=None, convert=True, collection=True, min_localization_count=1, **kwargs):
    """
    Load data from a rapidSTORM single-molecule localization file with tracked localizations.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a rapidSTORM file to load.
    nrows : int, None
        The number of localizations to load from file. None means that all available rows are loaded.
    convert : bool
        If True convert types by applying type specifications in locan.constants.PROPERTY_KEYS.
    collection : bool
        If True a collection of all tracks is returned. If False LocData with center positions is returned.
    min_localization_count : int
        If collection is True, only clusters with at least `min_localization_count` localizations are loaded.
    kwargs : dict
        Other parameters passed to `pandas.read_csv()`.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations/tracks as a collection.
    """
    with open_path_or_file_like(path) as file:
        columns, columns_track = read_rapidSTORM_track_header(file)
        lines = pd.read_csv(file, sep='\n', nrows=nrows, skiprows=1, header=None, **kwargs)

    lines = lines[0].str.split(" ", expand=False)

    if collection:
        # prepare dataframes with tracked localizations
        tracks = [np.array(line[len(columns) + 1:]).reshape(-1, len(columns_track)) for line in lines
                  if int(line[len(columns)]) >= min_localization_count]
        # +1 to account for the column with number of locs in track
        track_list = []
        for track in tracks:
            dataframe = pd.DataFrame(track, columns=columns_track)
            if convert:
                dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)
            else:
                dataframe = dataframe.convert_dtypes()
            dat = LocData.from_dataframe(dataframe=dataframe)
            track_list.append(dat)

        collection = LocData.from_collection(track_list)

        collection.meta.source = metadata_pb2.RAPIDSTORMTRACK
        collection.meta.state = metadata_pb2.RAW
        collection.meta.file_type = metadata_pb2.RAPIDSTORMTRACK
        collection.meta.file_path = str(path)

        for property in sorted(list(set(columns_track).intersection({'position_x', 'position_y', 'position_z'}))):
            collection.meta.unit.add(property=property, unit='nm')

        del collection.meta.history[:]
        collection.meta.history.add(name='load_rapidSTORM_track_file',
                                    parameter='path={}, nrows={}'.format(path, nrows))

        return collection

    else:
        # prepare dataframe with center track positions
        dataframe = pd.DataFrame([line[:len(columns)] for line in lines], columns=columns)
        if convert:
            dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)
        else:
            dataframe = dataframe.convert_dtypes()

        locdata = LocData.from_dataframe(dataframe=dataframe)

        locdata.meta.source = metadata_pb2.EXPERIMENT
        locdata.meta.state = metadata_pb2.RAW
        locdata.meta.file_type = metadata_pb2.RAPIDSTORM
        locdata.meta.file_path = str(path)

        for property in sorted(list(set(columns).intersection({'position_x', 'position_y', 'position_z'}))):
            locdata.meta.unit.add(property=property, unit='nm')

        del locdata.meta.history[:]
        locdata.meta.history.add(name='load_rapidSTORM_track_file', parameter='path={}, nrows={}'.format(path, nrows))

        return locdata
