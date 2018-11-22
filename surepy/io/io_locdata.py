"""

This module provides functions for file input/output with LocData objects.

"""
import time
import io
import warnings

import numpy as np
import pandas as pd
import xml.etree.ElementTree as etree
from asdf import AsdfFile
from asdf import open as asdf_open
from google.protobuf import json_format

from surepy import LocData
import surepy.constants
from surepy.data import metadata_pb2


def save_asdf(locdata, path):
    """
    Save LocData attributes in an asdf file.

    In the Advanced Scientific Data Format (ASDF) file format we store metadata, properties and column names as human-
    readable yaml header. The data is stored as binary numpy array.

    Parameters
    ----------
    locdata : LocData object
        The LocData object to be saved.
    path : str or Path object
        File path including file name to save to.
    """
    # Prepare tree
    meta_json = json_format.MessageToJson(locdata.meta, including_default_value_fields=False)
    tree = {
        'data': locdata.data.values,
        'columns': list(locdata.data),
        'properties': locdata.properties,
        'meta': meta_json
    }

    # Create the ASDF file object from tree
    af = AsdfFile(tree)

    # Write the data to a new file
    af.write_to(path)

# todo: handle ambiguous mapping of sigma key for 3D data
def save_thunderstorm_csv(locdata, path):
    """
    Save LocData attributes Thunderstorm-readable csv-file.

    In the Thunderstorm csv-file file format we store only localization data with Thunderstorm-readable column names.

    Parameters
    ----------
    locdata : LocData object
        The LocData object to be saved.
    path : str or Path object
        File path including file name to save to.
    """
    # get data from locdata object
    dataframe = locdata.data

    # create reverse mapping to Thunderstorm columns
    inv_map = {v: k for k, v in surepy.constants.THUNDERSTORM_KEYS.items()}

    # rename columns
    dataframe = dataframe.rename(index=str, columns=inv_map, inplace=False)

    # write to csv
    dataframe.to_csv(path, float_format='%.10g', index=False)


# todo: take out **kwargs

def load_txt_file(path, sep=',', columns=None, nrows=None, **kwargs):
    """
    Load localization data from a txt file.

    Surepy column names are either supplied or read from the first line header.

    Parameters
    ----------
    path : str or Path object
        File path for a localization file to load.
    sep : str
        separator between column values (Default: ',')
    columns : list of str or None
        Surepy column names. If None the first line is interpreted as header (Default: None).
    nrows : int, default: None
        The number of localizations to load from file. None means that all available rows are loaded (Default: None).

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    # define columns
    if columns is None:
        dataframe = pd.read_table(path, sep=sep, skiprows=0, nrows=nrows)

        for c in dataframe.columns:
            if c not in surepy.constants.PROPERTY_KEYS:
                warnings.warn('{} is not a Surepy property standard.'.format(c), UserWarning)
    else:
        for c in columns:
            if c not in surepy.constants.PROPERTY_KEYS:
                warnings.warn('{} is not a Surepy property standard.'.format(c), UserWarning)

        dataframe = pd.read_table(path, sep=sep, skiprows=1, nrows=nrows, names=columns)

    dat = LocData.from_dataframe(dataframe=dataframe, **kwargs)

    dat.meta.creation_date = int(time.time())
    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file_type = metadata_pb2.CUSTOM
    dat.meta.file_path = str(path)

    del dat.meta.history[:]
    dat.meta.history.add(name='load_txt_file',
                         parameter='path={}, sep={}, columns={}, nrows={}'.format(path, sep, columns, nrows))

    return dat


def load_rapidSTORM_header(path):
    """
    Load xml header from a rapidSTORM single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str or Path object
        File path for a rapidSTORM file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
    """

    # read xml part in header
    with open(path) as file:
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
        if i in surepy.constants.RAPIDSTORM_KEYS:
            column_keys.append(surepy.constants.RAPIDSTORM_KEYS[i])
        else:
            warnings.warn('{} is not a Surepy property standard.'.format(i), UserWarning)
            column_keys.append(i)
    return column_keys


def load_rapidSTORM_file(path, nrows=None, **kwargs):
    """
    Load data from a rapidSTORM single-molecule localization file.

    Parameters
    ----------
    path : string or Path object
        File path for a rapidSTORM file to load.
    nrows : int, default: None
        The number of localizations to load from file. None means that all available rows are loaded.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    columns = load_rapidSTORM_header(path)
    dataframe = pd.read_table(path, sep=" ", skiprows=1, nrows=nrows, names=columns)

    dat = LocData.from_dataframe(dataframe=dataframe, **kwargs)

    dat.meta.creation_date = int(time.time())
    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file_type = metadata_pb2.RAPIDSTORM
    dat.meta.file_path = str(path)

    for property in sorted(list(set(columns).intersection({'Position_x', 'Position_y', 'Position_z'}))):
        dat.meta.unit.add(property=property, unit='nm')

    del dat.meta.history[:]
    dat.meta.history.add(name='load_rapidSTORM_file', parameter='path={}, nrows={}'.format(path, nrows))

    return dat


def load_Elyra_header(path):
    """
    Load xml header from a Zeiss Elyra single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str or Path object
        File path for a rapidSTORM file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
    """

    with open(path) as file:
        header = file.readline().split('\n')[0]

    # list identifiers
    identifiers = header.split("\t")

    # turn identifiers into valuable LocData keys
    column_keys = []

    for i in identifiers:
        if i in surepy.constants.ELYRA_KEYS:
            column_keys.append(surepy.constants.ELYRA_KEYS[i])
        else:
            warnings.warn('{} is not a Surepy property standard.'.format(i), UserWarning)
            column_keys.append(i)

    return column_keys


def load_Elyra_file(path, nrows=None, **kwargs):
    """
    Load data from a rapidSTORM single-molecule localization file.

    Parameters
    ----------
    path : string or Path object
        File path for a rapidSTORM file to load.
    nrows : int, default: None
        The number of localizations to load from file. None means that all available rows are loaded.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    columns = load_Elyra_header(path)

    with open(path) as f:
        string = f.read()
        # remove metadata following nul byte
        string = string.split('\x00')[0]

        stream = io.StringIO(string)
        dataframe = pd.read_table(stream, sep="\t", skiprows=1, nrows=nrows, names=columns)

    dat = LocData.from_dataframe(dataframe=dataframe, **kwargs)

    dat.meta.creation_date = int(time.time())
    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file_type = metadata_pb2.ELYRA
    dat.meta.file_path = str(path)

    del dat.meta.history[:]
    dat.meta.history.add(name='load_Elyra_file', parameter='path={}, nrows={}'.format(path, nrows))

    return dat


def load_asdf_file(path):
    """
    Load data from ASDF localization file.

    Parameters
    ----------
    path : string or Path object
        File path for a rapidSTORM file to load.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    with asdf_open(path) as af:
        new_df = pd.DataFrame({k: af.tree['data'][:, n] for n, k in enumerate(af.tree['columns'])})
        locdata = LocData(dataframe=new_df)
        locdata.meta = json_format.Parse(af.tree['meta'], locdata.meta)
    return locdata


def load_thunderstorm_header(path):
    """
    Load csv header from a Thunderstorm single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str or Path object
        File path for a Thunderstorm file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the Thunderstorm identifiers.
    """

    # read csv header
    with open(path) as file:
        header = file.readline().split('\n')[0]

    # list identifiers
    identifiers = [x.strip('"') for x in header.split(',')]

    column_keys = []
    for i in identifiers:
        if i in surepy.constants.THUNDERSTORM_KEYS:
            column_keys.append(surepy.constants.THUNDERSTORM_KEYS[i])
        else:
            warnings.warn('{} is not a Surepy property standard.'.format(i), UserWarning)
            column_keys.append(i)
    return column_keys


def load_thunderstorm_file(path, nrows=None, **kwargs):
    """
    Load data from a Thunderstorm single-molecule localization file.

    Parameters
    ----------
    path : string or Path object
        File path for a Thunderstorm file to load.
    nrows : int, default: None
        The number of localizations to load from file. None means that all available rows are loaded.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    columns = load_thunderstorm_header(path)
    dataframe = pd.read_csv(path, sep=',', skiprows=1, nrows=nrows, names=columns)

    dat = LocData.from_dataframe(dataframe=dataframe, **kwargs)

    dat.meta.creation_date = int(time.time())
    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file_type = metadata_pb2.THUNDERSTORM
    dat.meta.file_path = str(path)

    for property in sorted(list(set(columns).intersection({'Position_x', 'Position_y', 'Position_z'}))):
        dat.meta.unit.add(property=property, unit='nm')

    del dat.meta.history[:]
    dat.meta.history.add(name='load_thundestorm_file', parameter='path={}, nrows={}'.format(path, nrows))

    return dat


def load_locdata(path, type=1, **kwargs):
    """
    Load data from localization file as specified by type.

    This function is a wrapper for read functions for the various types of SMLM data.


    Parameters
    ----------
    path : string or Path object
        File path for a localization data file to load.
    type : Int or str
        Integer or string indicating the file type.
        The integer should be according to surepy.data.metadata_pb2.file_type.
        String can be one of custom, rapidstorm, elyra, asdf.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    # todo fix protobuf constants for ASDF == 4

    if isinstance(type, int):

        if type == 1:
            return load_txt_file(path, **kwargs)
        elif type == 2:
            return load_rapidSTORM_file(path, **kwargs)
        elif type == 3:
            return load_Elyra_file(path, **kwargs)
        elif type == 4:
            return load_asdf_file(path, **kwargs)
        else:
            raise TypeError(f'There is no read function for type {type}.')


    if isinstance(type, str):

        if type.upper() == 'CUSTOM':
            return load_txt_file(path, **kwargs)
        elif type.upper() == 'RAPIDSTORM':
            return load_rapidSTORM_file(path, **kwargs)
        elif type.upper() == 'ELYRA':
            return load_Elyra_file(path, **kwargs)
        elif type.upper() == 'ASDF':
            return load_asdf_file(path, **kwargs)
        else:
            raise TypeError(f'There is no read function for type {type}.')

    else:
        raise TypeError(f'There is no read function for type {type}.')