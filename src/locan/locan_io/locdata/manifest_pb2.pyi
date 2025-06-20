from locan.data import metadata_pb2 as _metadata_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TABLE: _ClassVar[Type]
    IMAGE: _ClassVar[Type]

class Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BINARY: _ClassVar[Mode]
    TEXT: _ClassVar[Mode]

class Dtype(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INT8: _ClassVar[Dtype]
    UINT8: _ClassVar[Dtype]
    INT16: _ClassVar[Dtype]
    UINT16: _ClassVar[Dtype]
    INT32: _ClassVar[Dtype]
    UINT32: _ClassVar[Dtype]
    INT64: _ClassVar[Dtype]
    UINT64: _ClassVar[Dtype]
    FLOAT32: _ClassVar[Dtype]
    FLOAT64: _ClassVar[Dtype]

TABLE: Type
IMAGE: Type
BINARY: Mode
TEXT: Mode
INT8: Dtype
UINT8: Dtype
INT16: Dtype
UINT16: Dtype
INT32: Dtype
UINT32: Dtype
INT64: Dtype
UINT64: Dtype
FLOAT32: Dtype
FLOAT64: Dtype

class Format(_message.Message):
    __slots__ = (
        "name",
        "type",
        "mode",
        "columns",
        "headers",
        "dtype",
        "shape",
        "units",
        "description",
    )
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    UNITS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: Type
    mode: Mode
    columns: int
    headers: _containers.RepeatedScalarFieldContainer[str]
    dtype: _containers.RepeatedScalarFieldContainer[Dtype]
    shape: _containers.RepeatedScalarFieldContainer[int]
    units: _containers.RepeatedScalarFieldContainer[str]
    description: str
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

class FileInfo(_message.Message):
    __slots__ = (
        "name",
        "type",
        "format",
        "channel",
        "rows",
        "offset",
        "min",
        "max",
        "exposure",
    )

    class OffsetEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[int] = ...
        ) -> None: ...

    class MinEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    class MaxEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    EXPOSURE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: Type
    format: str
    channel: str
    rows: int
    offset: _containers.ScalarMap[str, int]
    min: _containers.ScalarMap[str, float]
    max: _containers.ScalarMap[str, float]
    exposure: float
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

class Manifest(_message.Message):
    __slots__ = (
        "format_version",
        "formats",
        "files",
        "name",
        "description",
        "tags",
        "thumbnail",
        "sample",
        "labeling",
        "date",
        "author",
        "citation",
        "email",
        "map",
        "locdata_meta",
    )

    class FormatsEntry(_message.Message):
        __slots__ = ("key", "value")
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
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...

    FORMAT_VERSION_FIELD_NUMBER: _ClassVar[int]
    FORMATS_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    THUMBNAIL_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_FIELD_NUMBER: _ClassVar[int]
    LABELING_FIELD_NUMBER: _ClassVar[int]
    DATE_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    CITATION_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    LOCDATA_META_FIELD_NUMBER: _ClassVar[int]
    format_version: str
    formats: _containers.MessageMap[str, Format]
    files: _containers.RepeatedCompositeFieldContainer[FileInfo]
    name: str
    description: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    thumbnail: str
    sample: str
    labeling: str
    date: str
    author: str
    citation: str
    email: str
    map: _containers.ScalarMap[str, str]
    locdata_meta: _metadata_pb2.Metadata
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
