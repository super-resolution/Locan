"""

File input/output for localization data in ASDF files

"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite

import pandas as pd
from asdf import AsdfFile
from asdf import open as asdf_open
from google.protobuf import json_format

from locan.constants import PROPERTY_KEYS
from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.locan_io.locdata.utilities import (
    convert_property_names,
    convert_property_types,
)

__all__: list[str] = ["save_asdf", "load_asdf_file"]

logger = logging.getLogger(__name__)


def save_asdf(
    locdata: LocData, path: str | os.PathLike[Any] | SupportsWrite[Any]
) -> None:
    """
    Save LocData attributes in an asdf file.

    In the Advanced Scientific Data Format (ASDF) file format we store
    metadata, properties and column names as human-readable yaml header.
    The data is stored as binary numpy.ndarray.

    Note
    ----
    Only selected LocData attributes are saved.
    Currently these are: 'data', 'columns', 'properties', 'meta'.

    Parameters
    ----------
    locdata
        The LocData object to be saved.
    path
        File path including file name to save to.
    """
    # Prepare tree
    meta_json = json_format.MessageToJson(
        locdata.meta, including_default_value_fields=False
    )
    tree = {
        "data": locdata.data.values,
        "columns": list(locdata.data),
        "properties": locdata.properties,
        "meta": meta_json,
    }

    # Create the ASDF file object from tree
    af = AsdfFile(tree)

    # Write the data to a new file
    af.write_to(path)


def load_asdf_file(
    path: str | os.PathLike[Any] | SupportsRead[Any],
    nrows: int | None = None,
    convert: bool = True,
) -> LocData:
    """
    Load data from ASDF localization file.

    Parameters
    ----------
    path
        File path for a rapidSTORM file to load.
    nrows
        The number of localizations to load from file.
        None means that all available rows are loaded.
    convert
        If True convert types by applying type specifications in
        locan.constants.PROPERTY_KEYS.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations.
    """
    with asdf_open(path) as af:
        new_df = pd.DataFrame(
            {
                k: af.tree["data"][slice(nrows), n]
                for n, k in enumerate(af.tree["columns"])
            }
        )

        column_keys = convert_property_names(
            properties=new_df.columns.tolist(), property_mapping=None
        )
        mapper = {key: value for key, value in zip(new_df.columns, column_keys)}
        new_df = new_df.rename(columns=mapper)

        if convert:
            new_df = convert_property_types(new_df, types=PROPERTY_KEYS)

        locdata = LocData(dataframe=new_df)
        locdata.meta = json_format.Parse(
            af.tree["meta"], locdata.meta, ignore_unknown_fields=True
        )
        locdata.meta.file.type = metadata_pb2.ASDF
        locdata.meta.file.path = str(path)
    return locdata
