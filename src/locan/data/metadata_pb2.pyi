from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import duration_pb2 as _duration_pb2
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

DESCRIPTOR: _descriptor.FileDescriptor

class Source(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN_SOURCE: _ClassVar[Source]
    DESIGN: _ClassVar[Source]
    EXPERIMENT: _ClassVar[Source]
    SIMULATION: _ClassVar[Source]
    IMPORT: _ClassVar[Source]

class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN_STATE: _ClassVar[State]
    RAW: _ClassVar[State]
    MODIFIED: _ClassVar[State]

class File_type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN_FILE_TYPE: _ClassVar[File_type]
    CUSTOM: _ClassVar[File_type]
    RAPIDSTORM: _ClassVar[File_type]
    ELYRA: _ClassVar[File_type]
    THUNDERSTORM: _ClassVar[File_type]
    ASDF: _ClassVar[File_type]
    NANOIMAGER: _ClassVar[File_type]
    RAPIDSTORMTRACK: _ClassVar[File_type]
    SMLM: _ClassVar[File_type]
    DECODE: _ClassVar[File_type]
    SMAP: _ClassVar[File_type]

UNKNOWN_SOURCE: Source
DESIGN: Source
EXPERIMENT: Source
SIMULATION: Source
IMPORT: Source
UNKNOWN_STATE: State
RAW: State
MODIFIED: State
UNKNOWN_FILE_TYPE: File_type
CUSTOM: File_type
RAPIDSTORM: File_type
ELYRA: File_type
THUNDERSTORM: File_type
ASDF: File_type
NANOIMAGER: File_type
RAPIDSTORMTRACK: File_type
SMLM: File_type
DECODE: File_type
SMAP: File_type

class Operation(_message.Message):
    __slots__ = ("name", "parameter")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_FIELD_NUMBER: _ClassVar[int]
    name: str
    parameter: str
    def __init__(
        self, name: _Optional[str] = ..., parameter: _Optional[str] = ...
    ) -> None: ...

class File(_message.Message):
    __slots__ = ("identifier", "comment", "type", "path", "groups")
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    type: File_type
    path: str
    groups: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        type: _Optional[_Union[File_type, str]] = ...,
        path: _Optional[str] = ...,
        groups: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class Address(_message.Message):
    __slots__ = ("address_lines", "city", "city_code", "country")
    ADDRESS_LINES_FIELD_NUMBER: _ClassVar[int]
    CITY_FIELD_NUMBER: _ClassVar[int]
    CITY_CODE_FIELD_NUMBER: _ClassVar[int]
    COUNTRY_FIELD_NUMBER: _ClassVar[int]
    address_lines: _containers.RepeatedScalarFieldContainer[str]
    city: str
    city_code: str
    country: str
    def __init__(
        self,
        address_lines: _Optional[_Iterable[str]] = ...,
        city: _Optional[str] = ...,
        city_code: _Optional[str] = ...,
        country: _Optional[str] = ...,
    ) -> None: ...

class Affiliation(_message.Message):
    __slots__ = ("institute", "department", "address")
    INSTITUTE_FIELD_NUMBER: _ClassVar[int]
    DEPARTMENT_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    institute: str
    department: str
    address: Address
    def __init__(
        self,
        institute: _Optional[str] = ...,
        department: _Optional[str] = ...,
        address: _Optional[_Union[Address, _Mapping]] = ...,
    ) -> None: ...

class Person(_message.Message):
    __slots__ = (
        "identifier",
        "comment",
        "first_name",
        "last_name",
        "title",
        "affiliations",
        "address",
        "emails",
        "roles",
    )
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    FIRST_NAME_FIELD_NUMBER: _ClassVar[int]
    LAST_NAME_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    AFFILIATIONS_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    EMAILS_FIELD_NUMBER: _ClassVar[int]
    ROLES_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    first_name: str
    last_name: str
    title: str
    affiliations: _containers.RepeatedCompositeFieldContainer[Affiliation]
    address: Address
    emails: _containers.RepeatedScalarFieldContainer[str]
    roles: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        first_name: _Optional[str] = ...,
        last_name: _Optional[str] = ...,
        title: _Optional[str] = ...,
        affiliations: _Optional[_Iterable[_Union[Affiliation, _Mapping]]] = ...,
        address: _Optional[_Union[Address, _Mapping]] = ...,
        emails: _Optional[_Iterable[str]] = ...,
        roles: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class ExperimentalSample(_message.Message):
    __slots__ = ("identifier", "comment", "targets", "fluorophores", "buffers", "map")

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
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    FLUOROPHORES_FIELD_NUMBER: _ClassVar[int]
    BUFFERS_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    targets: _containers.RepeatedScalarFieldContainer[str]
    fluorophores: _containers.RepeatedScalarFieldContainer[str]
    buffers: _containers.RepeatedScalarFieldContainer[str]
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        targets: _Optional[_Iterable[str]] = ...,
        fluorophores: _Optional[_Iterable[str]] = ...,
        buffers: _Optional[_Iterable[str]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class ExperimentalSetup(_message.Message):
    __slots__ = ("identifier", "comment", "optical_units", "map")

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
    OPTICAL_UNITS_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    optical_units: _containers.RepeatedCompositeFieldContainer[OpticalUnit]
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        optical_units: _Optional[_Iterable[_Union[OpticalUnit, _Mapping]]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class OpticalUnit(_message.Message):
    __slots__ = (
        "identifier",
        "comment",
        "illumination",
        "detection",
        "acquisition",
        "lightsheet",
    )
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    ILLUMINATION_FIELD_NUMBER: _ClassVar[int]
    DETECTION_FIELD_NUMBER: _ClassVar[int]
    ACQUISITION_FIELD_NUMBER: _ClassVar[int]
    LIGHTSHEET_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    illumination: Illumination
    detection: Detection
    acquisition: Acquisition
    lightsheet: Lightsheet
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        illumination: _Optional[_Union[Illumination, _Mapping]] = ...,
        detection: _Optional[_Union[Detection, _Mapping]] = ...,
        acquisition: _Optional[_Union[Acquisition, _Mapping]] = ...,
        lightsheet: _Optional[_Union[Lightsheet, _Mapping]] = ...,
    ) -> None: ...

class Illumination(_message.Message):
    __slots__ = (
        "identifier",
        "comment",
        "lightsource",
        "power",
        "area",
        "power_density",
        "wavelength",
        "map",
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
    LIGHTSOURCE_FIELD_NUMBER: _ClassVar[int]
    POWER_FIELD_NUMBER: _ClassVar[int]
    AREA_FIELD_NUMBER: _ClassVar[int]
    POWER_DENSITY_FIELD_NUMBER: _ClassVar[int]
    WAVELENGTH_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    lightsource: str
    power: float
    area: float
    power_density: float
    wavelength: float
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        lightsource: _Optional[str] = ...,
        power: _Optional[float] = ...,
        area: _Optional[float] = ...,
        power_density: _Optional[float] = ...,
        wavelength: _Optional[float] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Detection(_message.Message):
    __slots__ = ("identifier", "comment", "camera", "map")

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
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    camera: Camera
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        camera: _Optional[_Union[Camera, _Mapping]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Camera(_message.Message):
    __slots__ = (
        "identifier",
        "comment",
        "name",
        "model",
        "gain",
        "electrons_per_count",
        "integration_time",
        "pixel_count_x",
        "pixel_count_y",
        "pixel_size_x",
        "pixel_size_y",
        "flipped",
        "map",
        "offset",
        "serial_number",
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
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    GAIN_FIELD_NUMBER: _ClassVar[int]
    ELECTRONS_PER_COUNT_FIELD_NUMBER: _ClassVar[int]
    INTEGRATION_TIME_FIELD_NUMBER: _ClassVar[int]
    PIXEL_COUNT_X_FIELD_NUMBER: _ClassVar[int]
    PIXEL_COUNT_Y_FIELD_NUMBER: _ClassVar[int]
    PIXEL_SIZE_X_FIELD_NUMBER: _ClassVar[int]
    PIXEL_SIZE_Y_FIELD_NUMBER: _ClassVar[int]
    FLIPPED_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    SERIAL_NUMBER_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    name: str
    model: str
    gain: float
    electrons_per_count: float
    integration_time: _duration_pb2.Duration
    pixel_count_x: int
    pixel_count_y: int
    pixel_size_x: float
    pixel_size_y: float
    flipped: bool
    map: _containers.ScalarMap[str, str]
    offset: float
    serial_number: str
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        name: _Optional[str] = ...,
        model: _Optional[str] = ...,
        gain: _Optional[float] = ...,
        electrons_per_count: _Optional[float] = ...,
        integration_time: _Optional[_Union[_duration_pb2.Duration, _Mapping]] = ...,
        pixel_count_x: _Optional[int] = ...,
        pixel_count_y: _Optional[int] = ...,
        pixel_size_x: _Optional[float] = ...,
        pixel_size_y: _Optional[float] = ...,
        flipped: bool = ...,
        map: _Optional[_Mapping[str, str]] = ...,
        offset: _Optional[float] = ...,
        serial_number: _Optional[str] = ...,
    ) -> None: ...

class Acquisition(_message.Message):
    __slots__ = (
        "identifier",
        "comment",
        "frame_count",
        "frame_of_interest_first",
        "frame_of_interest_last",
        "time_start",
        "time_end",
        "stack_count",
        "stack_step_count",
        "stack_step_size",
        "map",
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
    FRAME_COUNT_FIELD_NUMBER: _ClassVar[int]
    FRAME_OF_INTEREST_FIRST_FIELD_NUMBER: _ClassVar[int]
    FRAME_OF_INTEREST_LAST_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
    STACK_COUNT_FIELD_NUMBER: _ClassVar[int]
    STACK_STEP_COUNT_FIELD_NUMBER: _ClassVar[int]
    STACK_STEP_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    frame_count: int
    frame_of_interest_first: int
    frame_of_interest_last: int
    time_start: _timestamp_pb2.Timestamp
    time_end: _timestamp_pb2.Timestamp
    stack_count: int
    stack_step_count: int
    stack_step_size: float
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        frame_count: _Optional[int] = ...,
        frame_of_interest_first: _Optional[int] = ...,
        frame_of_interest_last: _Optional[int] = ...,
        time_start: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        time_end: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        stack_count: _Optional[int] = ...,
        stack_step_count: _Optional[int] = ...,
        stack_step_size: _Optional[float] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Lightsheet(_message.Message):
    __slots__ = ("identifier", "comment", "angle_x", "angle_y", "angle_z", "map")

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
    ANGLE_X_FIELD_NUMBER: _ClassVar[int]
    ANGLE_Y_FIELD_NUMBER: _ClassVar[int]
    ANGLE_Z_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    angle_x: float
    angle_y: float
    angle_z: float
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        angle_x: _Optional[float] = ...,
        angle_y: _Optional[float] = ...,
        angle_z: _Optional[float] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Experiment(_message.Message):
    __slots__ = ("identifier", "comment", "experimenters", "samples", "setups", "map")

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
    EXPERIMENTERS_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    SETUPS_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    experimenters: _containers.RepeatedCompositeFieldContainer[Person]
    samples: _containers.RepeatedCompositeFieldContainer[ExperimentalSample]
    setups: _containers.RepeatedCompositeFieldContainer[ExperimentalSetup]
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        experimenters: _Optional[_Iterable[_Union[Person, _Mapping]]] = ...,
        samples: _Optional[_Iterable[_Union[ExperimentalSample, _Mapping]]] = ...,
        setups: _Optional[_Iterable[_Union[ExperimentalSetup, _Mapping]]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Localizer(_message.Message):
    __slots__ = (
        "identifier",
        "comment",
        "software",
        "intensity_threshold",
        "psf_fixed",
        "psf_size",
        "map",
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
    SOFTWARE_FIELD_NUMBER: _ClassVar[int]
    INTENSITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    PSF_FIXED_FIELD_NUMBER: _ClassVar[int]
    PSF_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    software: str
    intensity_threshold: float
    psf_fixed: bool
    psf_size: float
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        software: _Optional[str] = ...,
        intensity_threshold: _Optional[float] = ...,
        psf_fixed: bool = ...,
        psf_size: _Optional[float] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Relation(_message.Message):
    __slots__ = ("identifier", "comment", "file", "map")

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
    FILE_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    file: File
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        file: _Optional[_Union[File, _Mapping]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Property(_message.Message):
    __slots__ = ("identifier", "comment", "name", "unit", "type", "map")

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
    NAME_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    name: str
    unit: str
    type: str
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        name: _Optional[str] = ...,
        unit: _Optional[str] = ...,
        type: _Optional[str] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Metadata(_message.Message):
    __slots__ = (
        "identifier",
        "comment",
        "source",
        "state",
        "history",
        "ancestor_identifiers",
        "properties",
        "localization_properties",
        "element_count",
        "frame_count",
        "file",
        "relations",
        "experiment",
        "localizer",
        "map",
        "creation_time",
        "modification_time",
        "production_time",
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
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    ANCESTOR_IDENTIFIERS_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    LOCALIZATION_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_COUNT_FIELD_NUMBER: _ClassVar[int]
    FRAME_COUNT_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    RELATIONS_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_FIELD_NUMBER: _ClassVar[int]
    LOCALIZER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_FIELD_NUMBER: _ClassVar[int]
    MODIFICATION_TIME_FIELD_NUMBER: _ClassVar[int]
    PRODUCTION_TIME_FIELD_NUMBER: _ClassVar[int]
    identifier: str
    comment: str
    source: Source
    state: State
    history: _containers.RepeatedCompositeFieldContainer[Operation]
    ancestor_identifiers: _containers.RepeatedScalarFieldContainer[str]
    properties: _containers.RepeatedCompositeFieldContainer[Property]
    localization_properties: _containers.RepeatedCompositeFieldContainer[Property]
    element_count: int
    frame_count: int
    file: File
    relations: _containers.RepeatedCompositeFieldContainer[Relation]
    experiment: Experiment
    localizer: Localizer
    map: _containers.ScalarMap[str, str]
    creation_time: _timestamp_pb2.Timestamp
    modification_time: _timestamp_pb2.Timestamp
    production_time: _timestamp_pb2.Timestamp
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        source: _Optional[_Union[Source, str]] = ...,
        state: _Optional[_Union[State, str]] = ...,
        history: _Optional[_Iterable[_Union[Operation, _Mapping]]] = ...,
        ancestor_identifiers: _Optional[_Iterable[str]] = ...,
        properties: _Optional[_Iterable[_Union[Property, _Mapping]]] = ...,
        localization_properties: _Optional[_Iterable[_Union[Property, _Mapping]]] = ...,
        element_count: _Optional[int] = ...,
        frame_count: _Optional[int] = ...,
        file: _Optional[_Union[File, _Mapping]] = ...,
        relations: _Optional[_Iterable[_Union[Relation, _Mapping]]] = ...,
        experiment: _Optional[_Union[Experiment, _Mapping]] = ...,
        localizer: _Optional[_Union[Localizer, _Mapping]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
        creation_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        modification_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
        production_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...,
    ) -> None: ...
