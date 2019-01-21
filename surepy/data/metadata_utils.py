"""

Deal with metadata in LocData objects.

Functions to modify metadata in LocData objects.

"""
import time

from surepy.data import metadata_pb2


def _modify_meta(locdata, function_name=None, parameter=None, meta=None):
    """
    Update metadata in Locdata after modification of locdata.

    Parameters
    ----------
    locdata : LocData
        original locdata before modification
    function_name : str
        Name of function that was applied for modification.
    parameter : dict
        Parameter for function that was applied for modification.
    meta : Metadata protobuf message
        Metadata about the current dataset and its history.

    Returns
    -------
    Metadata protobuf message
        Metadata about the current dataset and its history.
    """
    meta_ = metadata_pb2.Metadata()
    meta_.CopyFrom(locdata.meta)
    try:
        meta_.ClearField("identifier")
    except ValueError:
        pass

    try:
        meta_.ClearField("element_count")
    except ValueError:
        pass

    try:
        meta_.ClearField("frame_count")
    except ValueError:
        pass

    meta_.modification_date = int(time.time())
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
