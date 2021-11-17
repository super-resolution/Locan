import io
from contextlib import closing

import pandas as pd

__all__ = ['convert_property_types']


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
