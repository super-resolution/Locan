"""

File input/output for localization data.

There are functions for reading the following file structures
(with an indicator string in parenthesis - see also `locan.constants.FileType`):

* custom text file (CUSTOM)
* rapidSTORM file format (RAPIDSTORM) [1]_
* Elyra file format (ELYRA)
* Thunderstorm file format (THUNDERSTORM) [2]_
* asdf file format (ASDF) [3]_
* Nanoimager file format (NANOIMAGER)
* rapidSTORM track file format (RAPIDSTORMTRACK) [1]_

References
----------
.. [1] Wolter S, Löschberger A, Holm T, Aufmkolk S, Dabauvalle MC, van de Linde S, Sauer M.,
   rapidSTORM: accurate, fast open-source software for localization microscopy.
   Nat Methods 9(11):1040-1, 2012, doi: 10.1038/nmeth.2224

.. [2] M. Ovesný, P. Křížek, J. Borkovec, Z. Švindrych, G. M. Hagen.
   ThunderSTORM: a comprehensive ImageJ plugin for PALM and STORM data analysis and super-resolution imaging.
   Bioinformatics 30(16):2389-2390, 2014.

.. [3] Greenfield, P., Droettboom, M., & Bray, E.,
   ASDF: A new data format for astronomy.
   Astronomy and Computing, 12: 240-251, 2015, doi:10.1016/j.ascom.2015.06.004
"""
import time
import io
import warnings
import logging
from enum import Enum
from contextlib import closing

import numpy as np
import pandas as pd
import xml.etree.ElementTree as etree
from asdf import AsdfFile
from asdf import open as asdf_open
from google.protobuf import json_format

from locan.data.locdata import LocData
import locan.constants
from locan.data import metadata_pb2


__all__ = ['convert_property_types', 'save_asdf', 'save_thunderstorm_csv',
           'load_txt_file', 'load_rapidSTORM_file', 'load_Elyra_file', 'load_asdf_file', 'load_thunderstorm_file',
           'load_Nanoimager_file', 'load_locdata']

logger = logging.getLogger(__name__)


def convert_property_types(dataframe, types, loc_properties=None):
    """
    Convert data types according to the column-type mapping in types.
    If the target type is one of 'integer', 'signed', 'unsigned', 'float'
    then :func:`pandas.to_numeric` will be applied.
    Otherwise, if the target type is any type object like `int`, `str`, `np.float64` or similar
    then :func:`pandas.astype` will be applied.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Data to be converted
    types : dict
        Mapping of loc_properties to types
    loc_properties : list(str), None
        The columns in dataframe to be converted. If None all columns will be converted according to types.

    Returns
    -------
    pandas.DataFrame
        A copy of dataframe with converted types
    """
    new_df = dataframe.copy()
    if loc_properties is None:
        loc_properties = dataframe.columns
    else:
        loc_properties = [key for key in loc_properties if key in dataframe.columns]
    selected_property_keys = {key: types[key] for key in loc_properties if key in types}

    for key, value in selected_property_keys.items():
        if isinstance(value, str):
            new_df[key] = pd.to_numeric(dataframe[key], downcast=value, errors='coerce')
        else:
            new_df[key] = dataframe[key].astype(value)

    return new_df


def open_path_or_file_like(path_or_file_like, mode='r', encoding=None):
    """
    Provide open-file context from `path_or_file_like` input.

    Parameters
    ----------
    path_or_file_like : str, os.PathLike, file-like
        Identifier for file
    mode
        same as in `open()`
    encoding
        same as in `open()`

    Returns
    -------
    context for file object
    """
    try:
        all(getattr(path_or_file_like, attr) for attr in ('seek', 'read', 'close'))
        file = path_or_file_like
    except (AttributeError, io.UnsupportedOperation):
        try:
            # if hasattr(path_or_file_like, "__fspath__")
            # or isinstance(path_or_file_like, (str, bytes)):
            file = open(path_or_file_like, mode=mode, encoding=encoding)
        except TypeError:
            raise TypeError("path_or_file_like must be str, bytes, os.PathLike or file-like.")
    return closing(file)


def save_asdf(locdata, path):
    """
    Save LocData attributes in an asdf file.

    In the Advanced Scientific Data Format (ASDF) file format we store metadata, properties and column names as human-
    readable yaml header. The data is stored as binary numpy.ndarray.

    Note
    ----
    Only selected LocData attributes are saved.
    Currently these are: 'data', 'columns', 'properties', 'meta'.

    Parameters
    ----------
    locdata : LocData
        The LocData object to be saved.
    path : str, os.PathLike, file-like
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


def save_thunderstorm_csv(locdata, path):
    """
    Save LocData attributes Thunderstorm-readable csv-file.

    In the Thunderstorm csv-file file format we store only localization data with Thunderstorm-readable column names.

    Parameters
    ----------
    locdata : LocData
        The LocData object to be saved.
    path : str, os.PathLike, file-like
        File path including file name to save to.
    """
    # get data from locdata object
    dataframe = locdata.data

    # create reverse mapping to Thunderstorm columns
    inv_map = {v: k for k, v in locan.constants.THUNDERSTORM_KEYS.items()}

    # rename columns
    dataframe = dataframe.rename(index=str, columns=inv_map, inplace=False)

    # write to csv
    dataframe.to_csv(path, float_format='%.10g', index=False)


def load_txt_file(path, sep=',', columns=None, nrows=None, **kwargs):
    """
    Load localization data from a txt file.

    Locan column names are either supplied or read from the first line header.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a localization file to load.
    sep : str
        separator between column values (Default: ',')
    columns : list of str, None
        Locan column names. If None the first line is interpreted as header (Default: None).
    nrows : int, None
        The number of localizations to load from file. None means that all available rows are loaded (Default: None).
    kwargs : dict
        Other parameters passed to `pandas.read_csv()`.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    # define columns
    if columns is None:
        dataframe = pd.read_csv(path, sep=sep, skiprows=0, nrows=nrows, **kwargs)

        for c in dataframe.columns:
            if c not in locan.constants.PROPERTY_KEYS:
                logger.warning(f'Column {c} is not a Locan property standard.')
    else:
        for c in columns:
            if c not in locan.constants.PROPERTY_KEYS:
                logger.warning(f'Column {c} is not a Locan property standard.')

        dataframe = pd.read_csv(path, sep=sep, skiprows=1, nrows=nrows, names=columns)

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file_type = metadata_pb2.CUSTOM
    dat.meta.file_path = str(path)

    del dat.meta.history[:]
    dat.meta.history.add(name='load_txt_file',
                         parameter='path={}, sep={}, columns={}, nrows={}'.format(path, sep, columns, nrows))

    return dat


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


def read_Elyra_header(file):
    """
    Read xml header from a Zeiss Elyra single-molecule localization file and identify column names.

    Parameters
    ----------
    file : file-like
        A rapidSTORM file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
    """
    header = file.readline().split('\n')[0]

    # list identifiers
    identifiers = header.split("\t")

    # turn identifiers into valuable LocData keys
    column_keys = []

    for i in identifiers:
        if i in locan.constants.ELYRA_KEYS:
            column_keys.append(locan.constants.ELYRA_KEYS[i])
        else:
            logger.warning(f'Column {i} is not a Locan property standard.')
            column_keys.append(i)

    return column_keys


def load_Elyra_header(path):
    """
    Load xml header from a Zeiss Elyra single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a rapidSTORM file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the rapidSTORM identifiers.
    """

    with open_path_or_file_like(path, encoding='latin-1') as file:
        return read_Elyra_header(file)


def load_Elyra_file(path, nrows=None, convert=True, **kwargs):
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

    Note
    ----
    Data is loaded with encoding = 'latin-1' and only data before the first NUL character is returned.
    Additional information appended at the end of the file is thus ignored.
    """
    with open_path_or_file_like(path, encoding='latin-1') as file:
        columns = read_Elyra_header(file)
        string = file.read()
        # remove metadata following nul byte
        string = string.split('\x00')[0]

        stream = io.StringIO(string)
        dataframe = pd.read_csv(stream, sep="\t", skiprows=0, nrows=nrows, names=columns, **kwargs)

    if convert:
        dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file_type = metadata_pb2.ELYRA
    dat.meta.file_path = str(path)

    del dat.meta.history[:]
    dat.meta.history.add(name='load_Elyra_file', parameter='path={}, nrows={}'.format(path, nrows))

    return dat


def load_asdf_file(path, nrows=None):
    """
    Load data from ASDF localization file.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a rapidSTORM file to load.
    nrows : int, None
        The number of localizations to load from file. None means that all available rows are loaded.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    with asdf_open(path) as af:
        new_df = pd.DataFrame({k: af.tree['data'][slice(nrows), n] for n, k in enumerate(af.tree['columns'])})
        locdata = LocData(dataframe=new_df)
        locdata.meta = json_format.Parse(af.tree['meta'], locdata.meta)
    return locdata


def read_thunderstorm_header(file):
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
    header = file.readline().split('\n')[0]

    # list identifiers
    identifiers = [x.strip('"') for x in header.split(',')]

    column_keys = []
    for i in identifiers:
        if i in locan.constants.THUNDERSTORM_KEYS:
            column_keys.append(locan.constants.THUNDERSTORM_KEYS[i])
        else:
            logger.warning(f'Column {i} is not a Locan property standard.')
            column_keys.append(i)
    return column_keys


def load_thunderstorm_header(path):
    """
    Load csv header from a Thunderstorm single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a Thunderstorm file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the Thunderstorm identifiers.
    """
    # read csv header
    with open_path_or_file_like(path) as file:
        return read_thunderstorm_header(file)


def load_thunderstorm_file(path, nrows=None, convert=True, **kwargs):
    """
    Load data from a Thunderstorm single-molecule localization file.

    Parameters
    ----------
    path : str, os.PathLike, file-like
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
        columns = read_thunderstorm_header(file)
        dataframe = pd.read_csv(file, sep=',', skiprows=0, nrows=nrows, names=columns, **kwargs)

    if convert:
        dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file_type = metadata_pb2.THUNDERSTORM
    dat.meta.file_path = str(path)

    for property in sorted(list(set(columns).intersection({'position_x', 'position_y', 'position_z'}))):
        dat.meta.unit.add(property=property, unit='nm')

    del dat.meta.history[:]
    dat.meta.history.add(name='load_thundestorm_file', parameter='path={}, nrows={}'.format(path, nrows))

    return dat


def read_Nanoimager_header(file):
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

    column_keys = []
    for i in identifiers:
        if i in locan.constants.NANOIMAGER_KEYS:
            column_keys.append(locan.constants.NANOIMAGER_KEYS[i])
        else:
            logger.warning(f'Column {i} is not a Locan property standard.')
            column_keys.append(i)
    return column_keys


def load_Nanoimager_header(path):
    """
    Load csv header from a Nanoimager single-molecule localization file and identify column names.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a Nanoimager file to load.

    Returns
    -------
    list of str
        A list of valid dataset property keys as derived from the Nanoimager identifiers.
    """
    # read csv header
    with open_path_or_file_like(path) as file:
        return read_Nanoimager_header(file)


def load_Nanoimager_file(path, nrows=None, convert=True, **kwargs):
    """
    Load data from a Nanoimager single-molecule localization file.

    Parameters
    ----------
    path : str, os.PathLike, file-like
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
        columns = read_Nanoimager_header(file)
        dataframe = pd.read_csv(file, sep=',', skiprows=0, nrows=nrows, names=columns, **kwargs)

    if convert:
        dataframe = convert_property_types(dataframe, types=locan.constants.PROPERTY_KEYS)

    dat = LocData.from_dataframe(dataframe=dataframe)

    dat.meta.source = metadata_pb2.EXPERIMENT
    dat.meta.state = metadata_pb2.RAW
    dat.meta.file_type = metadata_pb2.NANOIMAGER
    dat.meta.file_path = str(path)

    for property in sorted(list(set(columns).intersection({'position_x', 'position_y', 'position_z'}))):
        dat.meta.unit.add(property=property, unit='nm')

    del dat.meta.history[:]
    dat.meta.history.add(name='load_Nanoimager_file', parameter='path={}, nrows={}'.format(path, nrows))

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

        collection.meta.source = metadata_pb2.RAPIDSTORM
        collection.meta.state = metadata_pb2.RAW
        collection.meta.file_type = metadata_pb2.RAPIDSTORM
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


def _map_file_type_to_load_function(file_type):
    """
    Interpret user input for file_type.

    Parameters
    ----------
    file_type : int, str, locan.constants.FileType, locan.data.metadata_pb2.Metadata
        Identifier for the file type. Integer or string should be according to locan.constants.FileType.

    Returns
    -------
    callable
        Name of function for loading the localization file of `type`.
    """
    look_up_table = dict(
        load_txt_file=load_txt_file,
        load_rapidSTORM_file=load_rapidSTORM_file,
        load_Elyra_file=load_Elyra_file,
        load_thunderstorm_file=load_thunderstorm_file,
        load_asdf_file=load_asdf_file,
        load_Nanoimager_file=load_Nanoimager_file,
        load_rapidSTORM_track_file=load_rapidSTORM_track_file,
    )

    class LoadFunction(Enum):
        load_txt_file = 1
        load_rapidSTORM_file = 2
        load_Elyra_file = 3
        load_thunderstorm_file = 4
        load_asdf_file = 5
        load_Nanoimager_file = 6
        load_rapidSTORM_track_file = 7

    try:
        if isinstance(file_type, int):
            function_name = LoadFunction(file_type).name
        elif isinstance(file_type, str):
            function_name = LoadFunction(locan.constants.FileType[file_type.upper()].value).name
        elif isinstance(file_type, locan.constants.FileType):
            function_name = LoadFunction(file_type.value).name
        elif isinstance(file_type, metadata_pb2):
            function_name = LoadFunction(file_type).name
        else:
            raise TypeError
        return look_up_table[function_name]
    except ValueError:
        raise ValueError(f'There is no load function for type {file_type}.')


def load_locdata(path, file_type=1, nrows=None, **kwargs):
    """
    Load data from localization file as specified by type.

    This function is a wrapper for read functions for the various types of SMLM data.

    Parameters
    ----------
    path : str, os.PathLike, file-like
        File path for a localization data file to load.
    file_type : int, str, locan.constants.FileType, locan.data.metadata_pb2.Metadata
        Indicator for the file type.
        Integer or string should be according to locan.constants.FileType.
    nrows : int, None
        The number of localizations to load from file. None means that all available rows are loaded.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    return _map_file_type_to_load_function(file_type)(path=path, nrows=nrows, **kwargs)
