# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: locan/analysis/metadata_analysis.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n&locan/analysis/metadata_analysis.proto\x12\x0elocan.analysis\x1a\x1fgoogle/protobuf/timestamp.proto"3\n\x10\x41nalysis_routine\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tparameter\x18\x02 \x01(\t"\xd7\x02\n\tAMetadata\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x30\n\x06method\x18\x05 \x01(\x0b\x32 .locan.analysis.Analysis_routine\x12/\n\x03map\x18\x07 \x03(\x0b\x32".locan.analysis.AMetadata.MapEntry\x12\x31\n\rcreation_time\x18\x08 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x35\n\x11modification_time\x18\t \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01J\x04\x08\x03\x10\x04J\x04\x08\x04\x10\x05R\rcreation_dateR\x11modification_date'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "locan.analysis.metadata_analysis_pb2", globals()
)
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _AMETADATA_MAPENTRY._options = None
    _AMETADATA_MAPENTRY._serialized_options = b"8\001"
    _ANALYSIS_ROUTINE._serialized_start = 91
    _ANALYSIS_ROUTINE._serialized_end = 142
    _AMETADATA._serialized_start = 145
    _AMETADATA._serialized_end = 488
    _AMETADATA_MAPENTRY._serialized_start = 400
    _AMETADATA_MAPENTRY._serialized_end = 442
# @@protoc_insertion_point(module_scope)
