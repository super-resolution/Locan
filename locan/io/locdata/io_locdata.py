"""

File input/output for localization data.

"""
import io
import logging
from enum import Enum

import pandas as pd
from asdf import AsdfFile
from asdf import open as asdf_open
from google.protobuf import json_format

from locan.data.locdata import LocData
import locan.constants
from locan.data import metadata_pb2
from locan.io.locdata.utilities import convert_property_types, open_path_or_file_like
from locan.io.locdata.rapidstorm import load_rapidSTORM_file, load_rapidSTORM_track_file
from locan.io.locdata.smlm_file import load_SMLM_file


__all__ = ['save_asdf', 'save_thunderstorm_csv',
           'load_txt_file', 'load_Elyra_file', 'load_asdf_file', 'load_thunderstorm_file',
           'load_Nanoimager_file', 'load_locdata']

logger = logging.getLogger(__name__)


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

    for property_ in sorted(list(set(columns).intersection({'position_x', 'position_y', 'position_z'}))):
        dat.meta.unit.add(property=property_, unit='nm')

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

    for property_ in sorted(list(set(columns).intersection({'position_x', 'position_y', 'position_z'}))):
        dat.meta.unit.add(property=property_, unit='nm')

    del dat.meta.history[:]
    dat.meta.history.add(name='load_Nanoimager_file', parameter='path={}, nrows={}'.format(path, nrows))

    return dat


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
        load_SMLM_file=load_SMLM_file,
    )

    class LoadFunction(Enum):
        load_txt_file = 1
        load_rapidSTORM_file = 2
        load_Elyra_file = 3
        load_thunderstorm_file = 4
        load_asdf_file = 5
        load_Nanoimager_file = 6
        load_rapidSTORM_track_file = 7
        load_SMLM_file = 8

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
