"""

File input/output for localization data in SMLM files.

File specifications are provided at https://github.com/imodpasteur/smlm-file-format/blob/master/specification.md.

Code is adapted from https://github.com/imodpasteur/smlm-file-format/blob/master/implementations/Python/smlm_file.py.
(MIT license)
"""
from __future__ import annotations

import json
import logging
import os
import time
import zipfile
from typing import IO, Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from google.protobuf import json_format

import locan.constants
from locan.data import metadata_pb2
from locan.data.locdata import LocData
from locan.locan_io.locdata import manifest_pb2
from locan.locan_io.locdata.utilities import (
    convert_property_names,
    convert_property_types,
)
from locan.utils.format import _time_string

__all__: list[str] = [
    "manifest_format_from_locdata",
    "manifest_file_info_from_locdata",
    "manifest_from_locdata",
    "save_SMLM",
    "load_SMLM_manifest",
    "load_SMLM_header",
    "load_SMLM_file",
]

logger = logging.getLogger(__name__)


dtype2length = {
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float32": 4,
    "float64": 8,
}


def manifest_format_from_locdata(
    locdata: LocData,
) -> locan.locan_io.locdata.manifest_pb2.Format:
    """
    Prepare a manifest["format"] protobuf from locdata.
    The manifest holds metadata for smlm files.

    Parameters
    ----------
    locdata : LocData
        The LocData object.

    Returns
    -------
    locan.locan_io.locdata.manifest_pb2.Format
        The manifest format
    """
    format = manifest_pb2.Format()

    # required by
    format.name = "smlm-table(binary)-0"
    format.type = manifest_pb2.Type.TABLE  # type: ignore
    format.mode = manifest_pb2.Mode.BINARY  # type: ignore
    format.columns = len(locdata.data.columns)
    format.headers.extend(locdata.data.columns)
    format.dtype.extend(
        [manifest_pb2.Dtype.Value(str(dt).upper()) for dt in locdata.data.dtypes.values]  # type: ignore
    )
    format.shape.extend([1] * len(locdata.data.columns))
    format.units.extend([""] * len(locdata.data.columns))  # todo: add correct units

    # optional by specifications
    format.description = "localization table generated from locan.LocData"

    return format


def manifest_file_info_from_locdata(
    locdata: LocData,
) -> locan.locan_io.locdata.manifest_pb2.FileInfo:
    """
    Prepare a manifest["file_info"] protobuf from locdata.
    The manifest holds metadata for smlm files.

    Parameters
    ----------
    locdata : LocData
        The LocData object.

    Returns
    -------
    locan.locan_io.locdata.manifest_pb2.FileInfo
        The manifest file information
    """
    file_info = manifest_pb2.FileInfo()

    # required by specifications
    file_info.name = "table-0.bin"
    file_info.type = manifest_pb2.TABLE
    file_info.format = "smlm-table(binary)-0"
    file_info.channel = "default"
    file_info.rows = len(locdata)
    for column in locdata.data.columns:
        file_info.offset[column] = 0

    # optional by specifications
    for column in locdata.data.columns:
        file_info.min[column] = locdata.data[column].min()
    for column in locdata.data.columns:
        file_info.max[column] = locdata.data[column].max()
    # file_info.exposure = 20

    return file_info


def manifest_from_locdata(
    locdata: LocData, return_json_string: bool = False
) -> locan.locan_io.locdata.manifest_pb2.Manifest | str:
    """
    Prepare a manifest protobuf from locdata.
    The manifest holds metadata for smlm files.

    Parameters
    ----------
    locdata : LocData
        The LocData object.
    return_json_string : bool
        Flag for returning json string

    Returns
    -------
    locan.locan_io.locdata.manifest_pb2.Manifest | str
        The manifest
    """
    manifest = manifest_pb2.Manifest()
    format = manifest_format_from_locdata(locdata)
    file_info = manifest_file_info_from_locdata(locdata)

    # required by specifications
    manifest.format_version = "0.2"
    manifest.formats["smlm-table(binary)-0"].CopyFrom(format)
    manifest.files.append(file_info)

    # recommended by specifications
    manifest.name = locdata.meta.identifier
    manifest.description = locdata.meta.comment
    manifest.tags.append("SMLM")
    manifest.thumbnail = ""
    manifest.sample = ""
    manifest.labeling = ""
    manifest.date = _time_string(time.time())

    # optional by specifications
    manifest.author = ""
    manifest.citation = ""
    manifest.email = ""
    manifest.locdata_meta.CopyFrom(locdata.meta)

    if return_json_string:
        json_string = json_format.MessageToJson(
            manifest,
            preserving_proto_field_name=True,
            including_default_value_fields=False,
        )
        json_string = _change_upper_to_lower_keys(json_string)
        return json_string
    else:
        return manifest


def _change_upper_to_lower_keys(json_string: str) -> str:
    """Switch selected key words in json_string from upper to lower letters."""
    json_string = json_string.replace("BINARY", "binary")
    json_string = json_string.replace("TEXT", "text")
    json_string = json_string.replace("TABLE", "table")
    json_string = json_string.replace("IMAGE", "image")
    json_string = json_string.replace("INT", "int")
    json_string = json_string.replace("UINT", "uint")
    json_string = json_string.replace("FLOAT", "float")
    return json_string


def save_SMLM(
    locdata: LocData,
    path: str | os.PathLike[Any] | IO[Any],
    manifest: locan.locan_io.locdata.manifest_pb2.Manifest | None = None,
) -> None:
    """
    Save LocData attributes in a SMLM single-molecule
    localization (zip) file with manifest.json (version 0.2).

    In the smlm file format we store metadata as a human-
    readable manifest. The data is stored as byte string.

    Note
    ----
    Only selected LocData attributes are saved.
    These are: 'data', 'columns', 'meta'.

    Parameters
    ----------
    locdata
        The LocData object to be saved.
    path
        File path including file name to save to.
    manifest
        Protobuf with manifest to use instead of an autogenerated manifest.
    """
    if manifest is None:
        manifest_json = manifest_from_locdata(locdata, return_json_string=True)
    elif isinstance(manifest, str):
        manifest_json = manifest
    elif isinstance(manifest, manifest_pb2.Manifest):
        manifest_json = json_format.MessageToJson(
            manifest,
            preserving_proto_field_name=True,
            including_default_value_fields=False,
        )
        manifest_json = _change_upper_to_lower_keys(manifest_json)
    else:
        raise TypeError("Type of manifest is not correct.")

    byte_string = b"".join(
        [
            locdata.data[column][i].tobytes()
            for i in range(len(locdata))
            for column in locdata.data.columns
        ]
    )

    # save zip file
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as zf:
        # write manifest
        zf.writestr("manifest.json", cast(str, manifest_json))
        # write files
        zf.writestr("table-0.bin", byte_string)


def load_SMLM_manifest(
    path: str | os.PathLike[str] | IO[Any],
) -> dict[str, Any]:
    """
    Read manifest.json (version 0.2) from a SMLM single-molecule localization
    (zip) file.

    Parameters
    ----------
    path
        File path for a SMLM file to load.

    Returns
    -------
    dict[str, Any]
        manifest in json format
    """
    zf = zipfile.ZipFile(path, "r")
    file_names = zf.namelist()
    if "manifest.json" not in file_names:
        raise Exception("invalid file: no manifest.json found in the smlm file.")
    manifest: dict[str, Any] = json.loads(zf.read("manifest.json"))
    if manifest["format_version"] != "0.2":
        raise NotImplementedError("Not implemented for manifest version unlike 0.2.")
    return manifest


def load_SMLM_header(
    path: str | os.PathLike[str] | IO[Any],
) -> list[str]:
    """
    Read header (manifest) from a SMLM single-molecule localization file and
    identify column names.

    Parameters
    ----------
    path
        File path for a SMLM file to load.

    Returns
    -------
    list[str]
        A list of dataset property keys as derived from the SMLM identifiers.
    """
    zf = zipfile.ZipFile(path, "r")
    file_names = zf.namelist()
    if "manifest.json" not in file_names:
        raise Exception("invalid file: no manifest.json found in the smlm file.")
    manifest = json.loads(zf.read("manifest.json"))
    if manifest["format_version"] != "0.2":
        raise NotImplementedError("Not implemented for manifest version unlike 0.2.")

    locdata_columns_list = []
    for file_info in manifest["files"]:
        if file_info["type"] != "table":
            logger.info("ignore file with type: %s", file_info["type"])
        else:
            name = file_info["name"]
            logger.info(f"loading table {name}....")
            format_key = file_info["format"]
            file_format = manifest["formats"][format_key]
            if name not in file_names:
                logger.error("ERROR: Did not find %s in zip file", file_info["name"])
            else:
                headers = file_format["headers"]

                column_keys = convert_property_names(
                    properties=headers, property_mapping=locan.constants.SMLM_KEYS
                )
                locdata_columns_list.append(column_keys)

    if len(locdata_columns_list) == 1:
        return locdata_columns_list[0]
    else:
        return locdata_columns_list  # type: ignore


def load_SMLM_file(
    path: str | os.PathLike[Any] | IO[Any],
    nrows: int | None = None,
    convert: bool = True,
) -> LocData | list[LocData]:
    """
    Load data from a SMLM single-molecule localization file.

    Parameters
    ----------
    path
        File path for a SMLM file to load.
    nrows
        The number of localizations to load from file.
        None means that all available rows are loaded.
    convert
        If True convert types by applying type specifications in
        locan.constants.PROPERTY_KEYS.

    Returns
    -------
    LocData | list[LocData]
        A new instance of LocData with all localizations. Returns a list of
        LocData if multiple tables are found.
    """
    zf = zipfile.ZipFile(path, "r")
    file_names = zf.namelist()
    if "manifest.json" not in file_names:
        raise Exception("invalid file: no manifest.json found in the smlm file.")
    manifest = json.loads(zf.read("manifest.json"))
    if manifest["format_version"] != "0.2":
        raise NotImplementedError("Not implemented for manifest version unlike 0.2.")

    locdatas = []
    for file_info in manifest["files"]:
        if file_info["type"] != "table":
            logger.info("ignore file with type: %s", file_info["type"])
        else:
            name = file_info["name"]
            logger.debug(f"start loading {name} ...")
            format_key = file_info["format"]
            file_format = manifest["formats"][format_key]
            if file_format["mode"] != "binary":
                raise Exception(f"format mode {file_format['mode']} not supported.")
            else:
                try:
                    table_file = zf.read(file_info["name"])
                except KeyError:
                    logger.error(
                        "ERROR: Did not find %s in zip file", file_info["name"]
                    )
                    continue
                else:
                    logger.debug(f"loading {len(table_file)} bytes")
                    headers = file_format["headers"]
                    dtype = file_format["dtype"]
                    shape = file_format["shape"]
                    cols = len(headers)
                    rows = nrows if nrows is not None else int(file_info["rows"])

                    logger.debug("columns: %s", headers)
                    logger.debug("rows: %s, columns: %s", rows, cols)
                    if not len(headers) == len(dtype) == len(shape):
                        raise ValueError(
                            "headers, dtype, and shape have not the same length."
                        )

                    rowLen = sum(
                        dtype2length[dtype[i]] for i, header in enumerate(headers)
                    )
                    tableDict: dict[str, npt.NDArray[Any]] = {}
                    byteOffset = 0
                    for i, header in enumerate(headers):
                        tableDict[header] = np.ndarray(
                            (rows,),
                            buffer=table_file,
                            dtype=dtype[i],
                            offset=byteOffset,
                            order="C",
                            strides=(rowLen,),
                        )
                        byteOffset += dtype2length[dtype[i]]
                    logger.debug(f"finished loading {name}")

                    dataframe = pd.DataFrame.from_dict(tableDict)

                    column_keys = {}
                    for header in headers:
                        if header in locan.constants.SMLM_KEYS:
                            column_keys[header] = locan.constants.SMLM_KEYS[header]
                        elif header in locan.constants.RAPIDSTORM_KEYS:
                            column_keys[header] = locan.constants.RAPIDSTORM_KEYS[
                                header
                            ]
                        else:
                            logger.warning(
                                f"Column {header} is not a Locan property standard."
                            )
                            column_keys[header] = header
                    dataframe.rename(columns=column_keys, inplace=True)

                    if convert:
                        dataframe = convert_property_types(
                            dataframe, types=locan.constants.PROPERTY_KEYS
                        )

                    locdata = LocData.from_dataframe(dataframe=dataframe)

                    # todo: check on taking correct metadata
                    locdata.meta.source = metadata_pb2.EXPERIMENT
                    locdata.meta.state = metadata_pb2.RAW
                    locdata.meta.file.type = metadata_pb2.SMLM
                    locdata.meta.file.path = str(path)

                    for property_ in sorted(
                        list(
                            set(dataframe.columns).intersection(
                                {"position_x", "position_y", "position_z"}
                            )
                        )
                    ):
                        locdata.meta.localization_properties.add(
                            name=property_, unit="nm", type="float"
                        )

                    del locdata.meta.history[:]
                    locdata.meta.history.add(
                        name="load_SMLM_file", parameter=f"path={path}, nrows={nrows}"
                    )

                    locdatas.append(locdata)

    if len(locdatas) == 1:
        return locdatas[0]
    else:
        return locdatas
