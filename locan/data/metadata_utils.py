"""

Deal with metadata in LocData objects.

Functions to modify metadata in LocData objects.

"""
from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING, Any, BinaryIO

try:
    import tomllib  # type: ignore[import]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore  # for sys.version_info < (3, 11):

from google.protobuf import json_format, text_format
from google.protobuf.message import Message

from locan.data import metadata_pb2

if TYPE_CHECKING:
    from locan.data.locdata import LocData

__all__: list[str] = [
    "metadata_to_formatted_string",
    "metadata_from_toml_string",
    "load_metadata_from_toml",
    "message_scheme",
    "merge_metadata",
]

logger = logging.getLogger(__name__)


def _modify_meta(
    locdata: LocData,
    new_locdata: LocData,
    function_name: str | None = None,
    parameter: dict[str, Any] | None = None,
    meta: metadata_pb2.Metadata | None = None,
) -> metadata_pb2.Metadata:
    """
    Update metadata in Locdata after modification of locdata.

    Parameters
    ----------
    locdata
        original locdata before modification
    new_locdata
        new locdata after modification
    function_name
        Name of function that was applied for modification.
    parameter
        Parameter for function that was applied for modification.
    meta
        Metadata about the current dataset and its history.

    Returns
    -------
    locan.data.metadata_pb2.Metadata
        Metadata about the current dataset and its history.
    """
    meta_ = metadata_pb2.Metadata()
    meta_.CopyFrom(locdata.meta)
    # try:
    #     meta_.ClearField("identifier")
    # except ValueError:
    #     pass
    #
    # try:
    #     meta_.ClearField("element_count")
    # except ValueError:
    #     pass
    #
    # try:
    #     meta_.ClearField("frame_count")
    # except ValueError:
    #     pass

    meta_.identifier = new_locdata.meta.identifier
    meta_.element_count = new_locdata.meta.element_count
    meta_.frame_count = new_locdata.meta.frame_count
    meta_.modification_time.CopyFrom(new_locdata.meta.creation_time)

    meta_.state = metadata_pb2.MODIFIED
    meta_.ancestor_identifiers.append(locdata.meta.identifier)
    meta_.history.add(name=function_name, parameter=str(parameter))

    if meta is None:
        pass
    elif isinstance(meta, dict):
        for key, value in meta.items():
            setattr(meta_, key, value)
    else:
        meta_.MergeFrom(meta)

    return meta_


def _dict_to_protobuf(
    dictionary: dict[str, Any], message: Message, inplace: bool = False
) -> Message | None:
    """
    Parse dictionary with message attributes and their values in message.
    """
    if inplace is False:
        message_ = message.__class__()
        message_.CopyFrom(message)
        message = message_

    for key, value in dictionary.items():
        try:
            attr_ = getattr(message, key)
        except AttributeError as e:
            logger.warning(f"AttributeError while parsing: {e}")
            break

        if isinstance(value, dict):
            _dict_to_protobuf(dictionary=value, message=attr_, inplace=True)
        elif isinstance(value, list):
            try:
                attr_.extend(value)
            except TypeError:
                for element in value:
                    submessage = attr_.add()
                    _dict_to_protobuf(
                        dictionary=element, message=submessage, inplace=True
                    )
        else:
            try:
                setattr(message, key, value)
            except AttributeError:
                if attr_.DESCRIPTOR.name == "Timestamp":
                    attr_.FromJsonString(value)
                elif attr_.DESCRIPTOR.name == "Duration":
                    attr_.FromNanoseconds(value)

    if inplace:
        return None
    else:
        return message


def metadata_to_formatted_string(message: Message, **kwargs: Any) -> str:
    """
    Get formatted string from Locdata.metadata.

    Parameters
    ----------
    message
        Protobuf message like locan.data.metadata_pb2.Metadata
    kwargs
        Other kwargs that are passed to
        :func:`google.protobuf.text_format.MessageToString`.

    Returns
    -------
    str
        Formatted metadata string.
    """

    def message_formatter(
        message: Message, indent: int, as_one_line: bool
    ) -> str | None:
        if message.DESCRIPTOR.name in ["Timestamp", "Duration"]:
            return_value: str | None = message.ToJsonString()  # type: ignore[attr-defined]
        else:
            return_value = None
        return return_value

    return text_format.MessageToString(
        message, message_formatter=message_formatter, **kwargs
    )


def _toml_dict_to_protobuf(toml_dict: dict[str, Any]) -> dict[str, Message]:
    """
    Turn toml dict into protobuf messages.

    Parameters
    ----------
    toml_dict
        Dict from TOML string with metadata.

    Returns
    -------
    dict[str, google.protobuf.message.Message]
        Message instances with name as declared in toml file.
    """
    # instantiate messages
    instances: dict[str, Message] = {}
    for message in toml_dict.pop("messages"):
        module = importlib.import_module(message["module"])
        class_ = getattr(module, message["class_name"])
        instances[message["name"]] = class_()

    # parse values
    for message_name, dictionary in toml_dict.items():
        _dict_to_protobuf(
            dictionary=dictionary, message=instances[message_name], inplace=True
        )

    return instances


def metadata_from_toml_string(toml_string: str | None) -> dict[str, Message] | None:
    """
    Turn toml string into protobuf message instances.

    Note
    -----
    Parses Timestamp elements from string '2022-05-14T06:58:00Z'.
    Parses Duration elements from int in nanoseconds.

    Parameters
    ----------
    toml_string
        TOML string with metadata.

    Returns
    -------
    dict[str, google.protobuf.message.Message] | None
        Message instances with name as declared in toml file.
    """
    if toml_string is None:
        return None

    toml_dict = tomllib.loads(toml_string)
    return _toml_dict_to_protobuf(toml_dict)


def load_metadata_from_toml(
    path_or_file_like: str | bytes | os.PathLike[Any] | BinaryIO | None,
) -> dict[str, Message] | None:
    """
    Turn toml file into protobuf message instances.

    Note
    -----
    Parses Timestamp elements from string '2022-05-14T06:58:00Z'.
    Parses Duration elements from int in nanoseconds.

    Parameters
    ----------
    path_or_file_like
        File path or file-like for a TOML file.

    Returns
    -------
    dict[str, google.protobuf.message.Message] | None
        Message instances with name as declared in toml file.
    """
    if path_or_file_like is None:
        return None

    try:
        toml_dict = tomllib.load(path_or_file_like)  # type: ignore[arg-type]
    except AttributeError:
        with open(path_or_file_like, "rb") as file:  # type: ignore[arg-type]
            toml_dict = tomllib.load(file)

    return _toml_dict_to_protobuf(toml_dict)


def message_scheme(message: Message) -> dict[str, Any]:
    """
    Provide message scheme with defaults including nested messages.

    Parameters
    ----------
    message
        Protobuf message

    Returns
    -------
    dict[str, Any]
        A nested dictionary with all message fields including default values.
    """

    message_dict = json_format.MessageToDict(
        message, including_default_value_fields=True, preserving_proto_field_name=True
    )

    for descriptor in message.DESCRIPTOR.fields:
        if descriptor.type == descriptor.TYPE_MESSAGE:
            attr_ = getattr(message, descriptor.name)

            if descriptor.label != descriptor.LABEL_REPEATED:
                message_dict[descriptor.name] = message_scheme(attr_)

            elif (
                descriptor.label == descriptor.LABEL_REPEATED
                and "ScalarMap" not in type(attr_).__name__
                and "MessageMapContainer" not in type(attr_).__name__
            ):
                attr_ = attr_.add()
                message_dict[descriptor.name] = message_scheme(attr_)

    return message_dict


def merge_metadata(
    metadata: metadata_pb2.Metadata | None = None,
    other_metadata: metadata_pb2.Metadata
    | None
    | dict[str, Any]
    | str
    | bytes
    | os.PathLike[Any]
    | BinaryIO = None,
) -> metadata_pb2.Metadata:
    """
    Merge `other_metadata` into Locdata.meta.

    Parameters
    ----------
    metadata
        Original LocData metadata before modification
    other_metadata
        Metadata to be merged.

    Returns
    -------
    locan.data.metadata_pb2.Metadata
        Merged metadata
    """
    if metadata is None:
        new_metadata = metadata_pb2.Metadata()
    else:
        new_metadata = metadata

    if other_metadata is None:
        pass
    elif isinstance(other_metadata, (str, bytes, os.PathLike, BinaryIO)):
        meta_ = load_metadata_from_toml(other_metadata)
        if meta_ is not None:
            new_metadata.MergeFrom(meta_["metadata"])  # type: ignore[arg-type]
    elif isinstance(other_metadata, dict):
        for key, value in other_metadata.items():
            setattr(new_metadata, key, value)
    else:
        new_metadata.MergeFrom(other_metadata)

    return new_metadata
