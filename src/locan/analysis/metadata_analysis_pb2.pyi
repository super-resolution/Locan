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

class AMetadata(_message.Message):
    __slots__ = [
        "comment",
        "creation_time",
        "identifier",
        "map",
        "method",
        "modification_time",
    ]

    class MapEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    MODIFICATION_TIME_FIELD_NUMBER: _ClassVar[int]
    comment: str
    creation_time: _timestamp_pb2.Timestamp
    identifier: str
    map: _containers.ScalarMap[str, str]
    method: Analysis_routine
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

class Analysis_routine(_message.Message):
    __slots__ = ["name", "parameter"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_FIELD_NUMBER: _ClassVar[int]
    name: str
    parameter: str
    def __init__(
        self, name: _Optional[str] = ..., parameter: _Optional[str] = ...
    ) -> None: ...
