from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class Analysis_routine(_message.Message):
    __slots__ = ("name", "parameter")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_FIELD_NUMBER: _ClassVar[int]
    name: str
    parameter: str
    def __init__(
        self, name: _Optional[str] = ..., parameter: _Optional[str] = ...
    ) -> None: ...

class AMetadata(_message.Message):
    __slots__ = (
        "identifier",
        "comment",
        "method",
        "map",
        "creation_time",
        "modification_time",
    )

    class MapEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...

    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_FIELD_NUMBER: _ClassVar[int]
    MODIFICATION_TIME_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    method: Analysis_routine
    map: _containers.ScalarMap[str, str]
    creation_time: _timestamp_pb2.Timestamp
    modification_time: _timestamp_pb2.Timestamp
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        method: _Optional[_Union[Analysis_routine, _Mapping]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
        creation_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        modification_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
    ) -> None: ...
