"""

File input/output for localization data in SMLM files.

File specifications are provided at https://github.com/imodpasteur/smlm-file-format/blob/master/specification.md.

Code is adapted from https://github.com/imodpasteur/smlm-file-format/blob/master/implementations/Python/smlm_file.py.
(MIT license)
"""
import logging
import zipfile
import json

import numpy as np
import pandas as pd

from locan.data.locdata import LocData
import locan.constants
from locan.data import metadata_pb2
from locan.io.locdata.io_locdata import convert_property_types

__all__ = ['load_SMLM_file']

logger = logging.getLogger(__name__)


dtype2struct = {'uint8': 'B', 'uint32': 'I', 'float64': 'd', 'float32': 'f'}
dtype2length = {'uint8': 1, 'uint32': 4, 'float64': 8, 'float32': 4}


def load_SMLM_manifest(path):
    """
    Read manifest.json (version 0.2) from a SMLM single-molecule localization (zip) file.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a SMLM file to load.

    Returns
    -------
    dictionary
        manifest in json format
    """
    zf = zipfile.ZipFile(path, 'r')
    file_names = zf.namelist()
    if "manifest.json" not in file_names:
        raise Exception('invalid file: no manifest.json found in the smlm file.')
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest['format_version'] == '0.2'
    return manifest


def load_SMLM_header(path):
    """
    Read header (manifest) from a SMLM single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a SMLM file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the SMLM identifiers.
    """
    zf = zipfile.ZipFile(path, 'r')
    file_names = zf.namelist()
    if "manifest.json" not in file_names:
        raise Exception('invalid file: no manifest.json found in the smlm file.')
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest['format_version'] == '0.2'

    locdata_columns_list = []
    for file_info in manifest['files']:
        if file_info['type'] != "table":
            logger.info('ignore file with type: %s', file_info['type'])
        else:
            name = file_info['name']
            logger.info(f'loading table {name}....')
            format_key = file_info['format']
            file_format = manifest['formats'][format_key]
            if name not in file_names:
                logger.error('ERROR: Did not find %s in zip file', file_info['name'])
            else:
                headers = file_format['headers']

                column_keys = []
                for header in headers:
                    if header in locan.constants.SMLM_KEYS:
                        column_keys.append(locan.constants.SMLM_KEYS[header])
                    else:
                        logger.warning(f'Column {header} is not a Locan property standard.')
                        column_keys.append(header)
                locdata_columns_list.append(column_keys)

    if len(locdata_columns_list) == 1:
        return locdata_columns_list[0]
    else:
        return locdata_columns_list


def load_SMLM_file(path, nrows=None, convert=True):
    """
    Load data from a SMLM single-molecule localization file.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a SMLM file to load.
    nrows : int, None
        The number of localizations to load from file. None means that all available rows are loaded.
    convert : bool
        If True convert types by applying type specifications in locan.constants.PROPERTY_KEYS.

    Returns
    -------
    LocData, list(LocData)
        A new instance of LocData with all localizations, possibly for multiple tables.
    """
    zf = zipfile.ZipFile(path, 'r')
    file_names = zf.namelist()
    if "manifest.json" not in file_names:
        raise Exception('invalid file: no manifest.json found in the smlm file.')
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest['format_version'] == '0.2'

    locdatas = []
    for file_info in manifest['files']:
        if file_info['type'] != "table":
            logger.info('ignore file with type: %s', file_info['type'])
        else:
            name = file_info['name']
            logger.debug(f'start loading {name} ...')
            format_key = file_info['format']
            file_format = manifest['formats'][format_key]
            if file_format['mode'] != 'binary':
                raise Exception(f"format mode {file_format['mode']} not supported.")
            else:
                try:
                    table_file = zf.read(file_info['name'])
                except KeyError:
                    logger.error('ERROR: Did not find %s in zip file', file_info['name'])
                    continue
                else:
                    logger.debug(f'loading {len(table_file)} bytes')
                    headers = file_format['headers']
                    dtype = file_format['dtype']
                    shape = file_format['shape']
                    cols = len(headers)
                    rows = nrows if nrows is not None else file_info['rows']

                    logger.debug('columns: %s', headers)
                    logger.debug('rows: %s, columns: %s', rows, cols)
                    assert len(headers) == len(dtype) == len(shape)

                    rowLen = sum(dtype2length[dtype[i]] for i, header in enumerate(headers))
                    tableDict = {}
                    byteOffset = 0
                    for i, header in enumerate(headers):
                        tableDict[header] = np.ndarray((rows,), buffer=table_file, dtype=dtype[i], offset=byteOffset,
                                                       order='C', strides=(rowLen,))
                        byteOffset += dtype2length[dtype[i]]
                    logger.debug(f'finished loading {name}')

                    dataframe = pd.DataFrame.from_dict(tableDict)

                    column_keys = {}
                    for header in headers:
                        if header in locan.constants.SMLM_KEYS:
                            column_keys[header] = locan.constants.SMLM_KEYS[header]
                        elif header in locan.constants.RAPIDSTORM_KEYS:
                            column_keys[header] = locan.constants.RAPIDSTORM_KEYS[header]
                        else:
                            logger.warning(f'Column {header} is not a Locan property standard.')
                            column_keys[header] = header
                    dataframe.rename(columns=column_keys, inplace=True)

                    if convert:
                        dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)

                    locdata = LocData.from_dataframe(dataframe=dataframe)

                    locdata.meta.source = metadata_pb2.EXPERIMENT
                    locdata.meta.state = metadata_pb2.RAW
                    locdata.meta.file_type = metadata_pb2.SMLM
                    locdata.meta.file_path = str(path)

                    for property_ in sorted(list(set(dataframe.columns).intersection(
                            {'position_x', 'position_y', 'position_z'}))):
                        locdata.meta.unit.add(property=property_, unit='nm')

                    del locdata.meta.history[:]
                    locdata.meta.history.add(name='load_SMLM_file', parameter=f'path={path}, nrows={nrows}')

                    locdatas.append(locdata)

        if len(locdatas) == 1:
            return locdatas[0]
        else:
            return locdatas
