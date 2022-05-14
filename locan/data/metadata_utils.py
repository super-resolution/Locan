"""

Deal with metadata in LocData objects.

Functions to modify metadata in LocData objects.

"""
import time

from google.protobuf import text_format

from locan.data import metadata_pb2


__all__ = ['metadata_to_formatted_string']


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


def metadata_to_formatted_string(message, **kwargs) -> str:
    """
    Get formatted string to print Locdata.metadata.

    Parameters
    ----------
    message : locan.data.metadata_pb2.Metadata
        protobuf message
    kwargs : dict
        Other kwargs that are passed to :class:`google.protobuf.text_format.MessageToString`.

    Returns
    -------
    Formatted metadata string.
    """
    def message_formatter(message, indent, as_one_line):
        if message.DESCRIPTOR.name in ["Timestamp", "Duration"]:
            return message.ToJsonString()
        else:
            return None

    return text_format.MessageToString(message, message_formatter=message_formatter, **kwargs)
