"""

Methods for file input/output with Dataset objects.

"""

import numpy as np
import pandas as pd
import time
import xml.etree.ElementTree as etree
from surepy import LocData
import surepy.constants


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

    # turn identifiers into valuable Dataset_column_keys
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
    dat.meta['Modification'] = 'raw'
    dat.meta['Experimental setup'] =  {}
    dat.meta['Experimental sample'] =  {}
    dat.meta['File type'] = 'rapidStorm'
    dat.meta['File path'] = str(path)
    dat.meta['Units'] = {'Position_x': 'nm', 'Position_y': 'nm'}
    dat.meta['History'] = [{'Method:': 'load_rapidSTORM_file', 'Parameter': [path, nrows]}]

    return dat



def load_Elyra_file(path, nrows=None):
    """
    Load data from a rapidSTORM single-molecule localization file.

    Parameters
    ----------
    path : string
        The complete file path of the file to load.
    nrows : int, default: None meaning all available rows
        The number of localizations to load from file.


    Returns
    -------
    Dataset, Selection
        a new instance of Dataset and corresponding Selection of all localizations.
    """
    raise NotImplementedError


