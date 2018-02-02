"""

Methods for file input/output with Dataset objects.

"""
import io
import warnings

import numpy as np
import pandas as pd
import xml.etree.ElementTree as etree

from surepy import LocData
import surepy.constants

# todo: take out **kwargs

def load_txt_file(path, sep=',', columns=None, nrows=None, **kwargs):
    """
    Load localization data from a txt file.

    Surepy column names are either supplied or read from the first line header.

    Parameters
    ----------
    path : str or Path object
        File path for a rapidSTORM file to load.
    sep : str
        separator between column values (Default: ',')
    columns : list of str or None
        Surepy column names. If None the first line is interpreted as header (Default: None).
    nrows : int, default: None
        The number of localizations to load from file. None means that all available rows are loaded (Default: None).

    Returns
    -------
    Dataset
        a new instance of Dataset with all localizations.
    """
    # define columns
    if columns is None:
        dataframe = pd.read_table(path, sep=sep, skiprows=0, nrows=nrows)
    else:
        for c in columns:
            if c not in surepy.constants.PROPERTY_KEYS:
                warnings.warn('A property key is not Surepy standard.', UserWarning)

        dataframe = pd.read_table(path, sep=sep, skiprows=1, nrows=nrows, names=columns)

    dat = LocData.from_dataframe(dataframe=dataframe, **kwargs)
    dat.meta['State'] = 'raw'
    dat.meta['Experimental setup'] =  {}
    dat.meta['Experimental sample'] =  {}
    dat.meta['File type'] = 'custom'
    dat.meta['File path'] = str(path)
    dat.meta['Units'] = {'Position_x': 'nm', 'Position_y': 'nm'}
    dat.meta['History'] = [{'Method:': 'load_txt_file', 'Parameter': [path, sep, columns, nrows]}]

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
        column_keys.append(surepy.constants.RAPIDSTORM_KEYS[i])

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
    Dataset
        a new instance of Dataset with all localizations.
    """
    columns = load_rapidSTORM_header(path)
    dataframe = pd.read_table(path, sep=" ", skiprows=1, nrows=nrows, names=columns)

    dat = LocData.from_dataframe(dataframe=dataframe, **kwargs)
    dat.meta['State'] = 'raw'
    dat.meta['Experimental setup'] =  {}
    dat.meta['Experimental sample'] =  {}
    dat.meta['File type'] = 'rapidStorm'
    dat.meta['File path'] = str(path)
    dat.meta['Units'] = {'Position_x': 'nm', 'Position_y': 'nm'}
    dat.meta['History'] = [{'Method:': 'load_rapidSTORM_file', 'Parameter': [path, nrows]}]

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
        column_keys.append(surepy.constants.ELYRA_KEYS[i])

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
    Dataset
        a new instance of Dataset with all localizations.
    """
    columns = load_Elyra_header(path)

    with open(path) as f:
        string = f.read()
        # remove metadata following nul byte
        string = string.split('\x00')[0]

        stream = io.StringIO(string)
        dataframe = pd.read_table(stream, sep="\t", skiprows=1, nrows=nrows, names=columns)

    dat = LocData.from_dataframe(dataframe=dataframe, **kwargs)
    dat.meta['State'] = 'raw'
    dat.meta['Experimental setup'] =  {}
    dat.meta['Experimental sample'] =  {}
    dat.meta['File type'] = 'Elyra'
    dat.meta['File path'] = str(path)
    dat.meta['Units'] = {'Position_x': 'nm', 'Position_y': 'nm'}
    dat.meta['History'] = [{'Method:': 'load_Elyra_file', 'Parameter': [path, nrows]}]

    return dat



