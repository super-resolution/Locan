"""

Deal with metadata in LocData objects.

Functions to modify metadata in LocData objects.

"""
from __future__ import annotations

import importlib
import logging

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from google.protobuf import json_format, text_format
from google.protobuf.message import Message

from locan.data import metadata_pb2

__all__ = ["metadata_to_formatted_string", "metadata_from_toml", "message_scheme"]

logger = logging.getLogger(__name__)


def _modify_meta(locdata, new_locdata, function_name=None, parameter=None, meta=None):
    """
    Update metadata in Locdata after modification of locdata.

    Parameters
    ----------
    locdata : LocData
        original locdata before modification
    new_locdata : LocData
        new locdata after modification
    function_name : str
        Name of function that was applied for modification.
    parameter : dict
        Parameter for function that was applied for modification.
    meta : locan.data.metadata_pb2.Metadata
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
    dictionary: dict, message: Message, inplace: bool = False
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
            logging.warning(f"AttributeError while parsing: {e}")
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


def metadata_to_formatted_string(message, **kwargs):
    """
    Get formatted string from Locdata.metadata.

    Parameters
    ----------
    message : google.protobuf.message.Message
        Protobuf message like locan.data.metadata_pb2.Metadata

    kwargs : dict
        Other kwargs that are passed to :func:`google.protobuf.text_format.MessageToString`.

    Returns
    -------
    str
        Formatted metadata string.
    """

    def message_formatter(message, indent: int, as_one_line: bool) -> str | None:
        if message.DESCRIPTOR.name in ["Timestamp", "Duration"]:
            return message.ToJsonString()
        else:
            return None

    return text_format.MessageToString(
        message, message_formatter=message_formatter, **kwargs
    )


def metadata_from_toml(file):
    """
    Turn toml file into protobuf message instances.

    Note
    -----
    Parses Timestamp elements from string '2022-05-14T06:58:00Z'.
    Parses Duration elements from int in nanoseconds.

    Parameters
    ----------
    file : str
        TOML file-like to load from.

    Returns
    -------
    dict[str, google.protobuf.message.Message]
        Message instances with name as declared in toml file.
    """
    # load toml file
    toml_dict = tomllib.loads(file)

    # instantiate messages
    instances = {}
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


def message_scheme(message) -> dict:
    """
    Provide message scheme with defaults including nested messages.

    Parameters
    ----------
    message : google.protobuf.message.Message
        Protobuf message

    Returns
    -------
    dict
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
