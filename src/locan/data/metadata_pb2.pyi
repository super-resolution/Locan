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

ASDF: File_type
CUSTOM: File_type
DECODE: File_type
DESCRIPTOR: _descriptor.FileDescriptor
DESIGN: Source
ELYRA: File_type
EXPERIMENT: Source
IMPORT: Source
MODIFIED: State
NANOIMAGER: File_type
RAPIDSTORM: File_type
RAPIDSTORMTRACK: File_type
RAW: State
SIMULATION: Source
SMAP: File_type
SMLM: File_type
THUNDERSTORM: File_type
UNKNOWN_FILE_TYPE: File_type
UNKNOWN_SOURCE: Source
UNKNOWN_STATE: State

class Acquisition(_message.Message):
    __slots__ = [
        "comment",
        "frame_count",
        "frame_of_interest_first",
        "frame_of_interest_last",
        "identifier",
        "map",
        "stack_count",
        "stack_step_count",
        "stack_step_size",
        "time_end",
        "time_start",
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
    FRAME_COUNT_FIELD_NUMBER: _ClassVar[int]
    FRAME_OF_INTEREST_FIRST_FIELD_NUMBER: _ClassVar[int]
    FRAME_OF_INTEREST_LAST_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    STACK_COUNT_FIELD_NUMBER: _ClassVar[int]
    STACK_STEP_COUNT_FIELD_NUMBER: _ClassVar[int]
    STACK_STEP_SIZE_FIELD_NUMBER: _ClassVar[int]
    TIME_END_FIELD_NUMBER: _ClassVar[int]
    TIME_START_FIELD_NUMBER: _ClassVar[int]
    comment: str
    frame_count: int
    frame_of_interest_first: int
    frame_of_interest_last: int
    identifier: str
    map: _containers.ScalarMap[str, str]
    stack_count: int
    stack_step_count: int
    stack_step_size: float
    time_end: _timestamp_pb2.Timestamp
    time_start: _timestamp_pb2.Timestamp
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

class Address(_message.Message):
    __slots__ = ["address_lines", "city", "city_code", "country"]
    ADDRESS_LINES_FIELD_NUMBER: _ClassVar[int]
    CITY_CODE_FIELD_NUMBER: _ClassVar[int]
    CITY_FIELD_NUMBER: _ClassVar[int]
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
    __slots__ = ["address", "department", "institute"]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DEPARTMENT_FIELD_NUMBER: _ClassVar[int]
    INSTITUTE_FIELD_NUMBER: _ClassVar[int]
    address: Address
    department: str
    institute: str
    def __init__(
        self,
        institute: _Optional[str] = ...,
        department: _Optional[str] = ...,
        address: _Optional[_Union[Address, _Mapping]] = ...,
    ) -> None: ...

class Camera(_message.Message):
    __slots__ = [
        "comment",
        "electrons_per_count",
        "flipped",
        "gain",
        "identifier",
        "integration_time",
        "map",
        "model",
        "name",
        "offset",
        "pixel_count_x",
        "pixel_count_y",
        "pixel_size_x",
        "pixel_size_y",
        "serial_number",
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
    ELECTRONS_PER_COUNT_FIELD_NUMBER: _ClassVar[int]
    FLIPPED_FIELD_NUMBER: _ClassVar[int]
    GAIN_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    INTEGRATION_TIME_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    PIXEL_COUNT_X_FIELD_NUMBER: _ClassVar[int]
    PIXEL_COUNT_Y_FIELD_NUMBER: _ClassVar[int]
    PIXEL_SIZE_X_FIELD_NUMBER: _ClassVar[int]
    PIXEL_SIZE_Y_FIELD_NUMBER: _ClassVar[int]
    SERIAL_NUMBER_FIELD_NUMBER: _ClassVar[int]
    comment: str
    electrons_per_count: float
    flipped: bool
    gain: float
    identifier: str
    integration_time: _duration_pb2.Duration
    map: _containers.ScalarMap[str, str]
    model: str
    name: str
    offset: float
    pixel_count_x: int
    pixel_count_y: int
    pixel_size_x: float
    pixel_size_y: float
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

class Detection(_message.Message):
    __slots__ = ["camera", "comment", "identifier", "map"]

    class MapEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    camera: Camera
    comment: str
    identifier: str
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        camera: _Optional[_Union[Camera, _Mapping]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Experiment(_message.Message):
    __slots__ = ["comment", "experimenters", "identifier", "map", "samples", "setups"]

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
    EXPERIMENTERS_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    SETUPS_FIELD_NUMBER: _ClassVar[int]
    comment: str
    experimenters: _containers.RepeatedCompositeFieldContainer[Person]
    identifier: str
    map: _containers.ScalarMap[str, str]
    samples: _containers.RepeatedCompositeFieldContainer[ExperimentalSample]
    setups: _containers.RepeatedCompositeFieldContainer[ExperimentalSetup]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        experimenters: _Optional[_Iterable[_Union[Person, _Mapping]]] = ...,
        samples: _Optional[_Iterable[_Union[ExperimentalSample, _Mapping]]] = ...,
        setups: _Optional[_Iterable[_Union[ExperimentalSetup, _Mapping]]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class ExperimentalSample(_message.Message):
    __slots__ = ["buffers", "comment", "fluorophores", "identifier", "map", "targets"]

    class MapEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...
    BUFFERS_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    FLUOROPHORES_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    buffers: _containers.RepeatedScalarFieldContainer[str]
    comment: str
    fluorophores: _containers.RepeatedScalarFieldContainer[str]
    identifier: str
    map: _containers.ScalarMap[str, str]
    targets: _containers.RepeatedScalarFieldContainer[str]
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
    __slots__ = ["comment", "identifier", "map", "optical_units"]

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
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    OPTICAL_UNITS_FIELD_NUMBER: _ClassVar[int]
    comment: str
    identifier: str
    map: _containers.ScalarMap[str, str]
    optical_units: _containers.RepeatedCompositeFieldContainer[OpticalUnit]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        optical_units: _Optional[_Iterable[_Union[OpticalUnit, _Mapping]]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class File(_message.Message):
    __slots__ = ["comment", "groups", "identifier", "path", "type"]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    comment: str
    groups: _containers.RepeatedScalarFieldContainer[str]
    identifier: str
    path: str
    type: File_type
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        type: _Optional[_Union[File_type, str]] = ...,
        path: _Optional[str] = ...,
        groups: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class Illumination(_message.Message):
    __slots__ = [
        "area",
        "comment",
        "identifier",
        "lightsource",
        "map",
        "power",
        "power_density",
        "wavelength",
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
    AREA_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    LIGHTSOURCE_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    POWER_DENSITY_FIELD_NUMBER: _ClassVar[int]
    POWER_FIELD_NUMBER: _ClassVar[int]
    WAVELENGTH_FIELD_NUMBER: _ClassVar[int]
    area: float
    comment: str
    identifier: str
    lightsource: str
    map: _containers.ScalarMap[str, str]
    power: float
    power_density: float
    wavelength: float
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

class Lightsheet(_message.Message):
    __slots__ = ["angle_x", "angle_y", "angle_z", "comment", "identifier", "map"]

    class MapEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...
    ANGLE_X_FIELD_NUMBER: _ClassVar[int]
    ANGLE_Y_FIELD_NUMBER: _ClassVar[int]
    ANGLE_Z_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    angle_x: float
    angle_y: float
    angle_z: float
    comment: str
    identifier: str
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

class Localizer(_message.Message):
    __slots__ = [
        "comment",
        "identifier",
        "intensity_threshold",
        "map",
        "psf_fixed",
        "psf_size",
        "software",
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
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    INTENSITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    PSF_FIXED_FIELD_NUMBER: _ClassVar[int]
    PSF_SIZE_FIELD_NUMBER: _ClassVar[int]
    SOFTWARE_FIELD_NUMBER: _ClassVar[int]
    comment: str
    identifier: str
    intensity_threshold: float
    map: _containers.ScalarMap[str, str]
    psf_fixed: bool
    psf_size: float
    software: str
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

class Metadata(_message.Message):
    __slots__ = [
        "ancestor_identifiers",
        "comment",
        "creation_time",
        "element_count",
        "experiment",
        "file",
        "frame_count",
        "history",
        "identifier",
        "localization_properties",
        "localizer",
        "map",
        "modification_time",
        "production_time",
        "properties",
        "relations",
        "source",
        "state",
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
    ANCESTOR_IDENTIFIERS_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_COUNT_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    FRAME_COUNT_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    LOCALIZATION_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    LOCALIZER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    MODIFICATION_TIME_FIELD_NUMBER: _ClassVar[int]
    PRODUCTION_TIME_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    RELATIONS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ancestor_identifiers: _containers.RepeatedScalarFieldContainer[str]
    comment: str
    creation_time: _timestamp_pb2.Timestamp
    element_count: int
    experiment: Experiment
    file: File
    frame_count: int
    history: _containers.RepeatedCompositeFieldContainer[Operation]
    identifier: str
    localization_properties: _containers.RepeatedCompositeFieldContainer[Property]
    localizer: Localizer
    map: _containers.ScalarMap[str, str]
    modification_time: _timestamp_pb2.Timestamp
    production_time: _timestamp_pb2.Timestamp
    properties: _containers.RepeatedCompositeFieldContainer[Property]
    relations: _containers.RepeatedCompositeFieldContainer[Relation]
    source: Source
    state: State
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

class Operation(_message.Message):
    __slots__ = ["name", "parameter"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_FIELD_NUMBER: _ClassVar[int]
    name: str
    parameter: str
    def __init__(
        self, name: _Optional[str] = ..., parameter: _Optional[str] = ...
    ) -> None: ...

class OpticalUnit(_message.Message):
    __slots__ = [
        "acquisition",
        "comment",
        "detection",
        "identifier",
        "illumination",
        "lightsheet",
    ]
    ACQUISITION_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    DETECTION_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    ILLUMINATION_FIELD_NUMBER: _ClassVar[int]
    LIGHTSHEET_FIELD_NUMBER: _ClassVar[int]
    acquisition: Acquisition
    comment: str
    detection: Detection
    identifier: str
    illumination: Illumination
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

class Person(_message.Message):
    __slots__ = [
        "address",
        "affiliations",
        "comment",
        "emails",
        "first_name",
        "identifier",
        "last_name",
        "roles",
        "title",
    ]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    AFFILIATIONS_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    EMAILS_FIELD_NUMBER: _ClassVar[int]
    FIRST_NAME_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    LAST_NAME_FIELD_NUMBER: _ClassVar[int]
    ROLES_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    address: Address
    affiliations: _containers.RepeatedCompositeFieldContainer[Affiliation]
    comment: str
    emails: _containers.RepeatedScalarFieldContainer[str]
    first_name: str
    identifier: str
    last_name: str
    roles: _containers.RepeatedScalarFieldContainer[str]
    title: str
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

class Property(_message.Message):
    __slots__ = ["comment", "identifier", "map", "name", "type", "unit"]

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
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    comment: str
    identifier: str
    map: _containers.ScalarMap[str, str]
    name: str
    type: str
    unit: str
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        name: _Optional[str] = ...,
        unit: _Optional[str] = ...,
        type: _Optional[str] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Relation(_message.Message):
    __slots__ = ["comment", "file", "identifier", "map"]

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
    FILE_FIELD_NUMBER: _ClassVar[int]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    comment: str
    file: File
    identifier: str
    map: _containers.ScalarMap[str, str]
    def __init__(
        self,
        identifier: _Optional[str] = ...,
        comment: _Optional[str] = ...,
        file: _Optional[_Union[File, _Mapping]] = ...,
        map: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class Source(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class File_type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
