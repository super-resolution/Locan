from locan.data import metadata_pb2 as _metadata_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

BINARY: Mode
DESCRIPTOR: _descriptor.FileDescriptor
FLOAT32: Dtype
FLOAT64: Dtype
IMAGE: Type
INT16: Dtype
INT32: Dtype
INT64: Dtype
INT8: Dtype
TABLE: Type
TEXT: Mode
UINT16: Dtype
UINT32: Dtype
UINT64: Dtype
UINT8: Dtype

class FileInfo(_message.Message):
    __slots__ = [
        "channel",
        "exposure",
        "format",
        "max",
        "min",
        "name",
        "offset",
        "rows",
        "type",
    ]

    class MaxEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    class MinEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    class OffsetEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[int] = ...
        ) -> None: ...
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    EXPOSURE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    channel: str
    exposure: float
    format: str
    max: _containers.ScalarMap[str, float]
    min: _containers.ScalarMap[str, float]
    name: str
    offset: _containers.ScalarMap[str, int]
    rows: int
    type: Type
    def __init__(
        self,
        name: _Optional[str] = ...,
        type: _Optional[_Union[Type, str]] = ...,
        format: _Optional[str] = ...,
        channel: _Optional[str] = ...,
        rows: _Optional[int] = ...,
        offset: _Optional[_Mapping[str, int]] = ...,
        min: _Optional[_Mapping[str, float]] = ...,
        max: _Optional[_Mapping[str, float]] = ...,
        exposure: _Optional[float] = ...,
    ) -> None: ...

class Format(_message.Message):
    __slots__ = [
        "columns",
        "description",
        "dtype",
        "headers",
        "mode",
        "name",
        "shape",
        "type",
        "units",
    ]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    UNITS_FIELD_NUMBER: _ClassVar[int]
    columns: int
    description: str
    dtype: _containers.RepeatedScalarFieldContainer[Dtype]
    headers: _containers.RepeatedScalarFieldContainer[str]
    mode: Mode
    name: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    type: Type
    units: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        name: _Optional[str] = ...,
        type: _Optional[_Union[Type, str]] = ...,
        mode: _Optional[_Union[Mode, str]] = ...,
        columns: _Optional[int] = ...,
        headers: _Optional[_Iterable[str]] = ...,
        dtype: _Optional[_Iterable[_Union[Dtype, str]]] = ...,
        shape: _Optional[_Iterable[int]] = ...,
        units: _Optional[_Iterable[str]] = ...,
        description: _Optional[str] = ...,
    ) -> None: ...

class Manifest(_message.Message):
    __slots__ = [
        "author",
        "citation",
        "date",
        "description",
        "email",
        "files",
        "format_version",
        "formats",
        "labeling",
        "locdata_meta",
        "map",
        "name",
        "sample",
        "tags",
        "thumbnail",
    ]

    class FormatsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Format
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[Format, _Mapping]] = ...,
        ) -> None: ...

    class MapEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    CITATION_FIELD_NUMBER: _ClassVar[int]
    DATE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    FORMATS_FIELD_NUMBER: _ClassVar[int]
    FORMAT_VERSION_FIELD_NUMBER: _ClassVar[int]
    LABELING_FIELD_NUMBER: _ClassVar[int]
    LOCDATA_META_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    THUMBNAIL_FIELD_NUMBER: _ClassVar[int]
    author: str
    citation: str
    date: str
    description: str
    email: str
    files: _containers.RepeatedCompositeFieldContainer[FileInfo]
    format_version: str
    formats: _containers.MessageMap[str, Format]
    labeling: str
    locdata_meta: _metadata_pb2.Metadata
    map: _containers.ScalarMap[str, str]
    name: str
    sample: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    thumbnail: str
    def __init__(
        self,
        format_version: _Optional[str] = ...,
        formats: _Optional[_Mapping[str, Format]] = ...,
        files: _Optional[_Iterable[_Union[FileInfo, _Mapping]]] = ...,
        name: _Optional[str] = ...,
        description: _Optional[str] = ...,
        tags: _Optional[_Iterable[str]] = ...,
        thumbnail: _Optional[str] = ...,
        sample: _Optional[str] = ...,
        labeling: _Optional[str] = ...,
        date: _Optional[str] = ...,
        author: _Optional[str] = ...,
        citation: _Optional[str] = ...,
        email: _Optional[str] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
        locdata_meta: _Optional[_Union[_metadata_pb2.Metadata, _Mapping]] = ...,
    ) -> None: ...

class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class Dtype(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
