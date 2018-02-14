# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: analysis/metadata_analysis.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='analysis/metadata_analysis.proto',
  package='surepy',
  syntax='proto2',
  serialized_pb=_b('\n analysis/metadata_analysis.proto\x12\x06surepy\"-\n\nOperation_\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tparameter\x18\x02 \x01(\t\"\xdb\x01\n\tMetadata_\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x15\n\rcreation_date\x18\x03 \x01(\x03\x12\x19\n\x11modification_date\x18\x04 \x01(\x03\x12\"\n\x06method\x18\x05 \x01(\x0b\x32\x12.surepy.Operation_\x12\'\n\x03map\x18\x07 \x03(\x0b\x32\x1a.surepy.Metadata_.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01')
)




_OPERATION_ = _descriptor.Descriptor(
  name='Operation_',
  full_name='surepy.Operation_',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='surepy.Operation_.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='parameter', full_name='surepy.Operation_.parameter', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=44,
  serialized_end=89,
)


_METADATA__MAPENTRY = _descriptor.Descriptor(
  name='MapEntry',
  full_name='surepy.Metadata_.MapEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='surepy.Metadata_.MapEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='surepy.Metadata_.MapEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=269,
  serialized_end=311,
)

_METADATA_ = _descriptor.Descriptor(
  name='Metadata_',
  full_name='surepy.Metadata_',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='identifier', full_name='surepy.Metadata_.identifier', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='comment', full_name='surepy.Metadata_.comment', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='creation_date', full_name='surepy.Metadata_.creation_date', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='modification_date', full_name='surepy.Metadata_.modification_date', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='method', full_name='surepy.Metadata_.method', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='map', full_name='surepy.Metadata_.map', index=5,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_METADATA__MAPENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=92,
  serialized_end=311,
)

_METADATA__MAPENTRY.containing_type = _METADATA_
_METADATA_.fields_by_name['method'].message_type = _OPERATION_
_METADATA_.fields_by_name['map'].message_type = _METADATA__MAPENTRY
DESCRIPTOR.message_types_by_name['Operation_'] = _OPERATION_
DESCRIPTOR.message_types_by_name['Metadata_'] = _METADATA_
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Operation_ = _reflection.GeneratedProtocolMessageType('Operation_', (_message.Message,), dict(
  DESCRIPTOR = _OPERATION_,
  __module__ = 'analysis.metadata_analysis_pb2'
  # @@protoc_insertion_point(class_scope:surepy.Operation_)
  ))
_sym_db.RegisterMessage(Operation_)

Metadata_ = _reflection.GeneratedProtocolMessageType('Metadata_', (_message.Message,), dict(

  MapEntry = _reflection.GeneratedProtocolMessageType('MapEntry', (_message.Message,), dict(
    DESCRIPTOR = _METADATA__MAPENTRY,
    __module__ = 'analysis.metadata_analysis_pb2'
    # @@protoc_insertion_point(class_scope:surepy.Metadata_.MapEntry)
    ))
  ,
  DESCRIPTOR = _METADATA_,
  __module__ = 'analysis.metadata_analysis_pb2'
  # @@protoc_insertion_point(class_scope:surepy.Metadata_)
  ))
_sym_db.RegisterMessage(Metadata_)
_sym_db.RegisterMessage(Metadata_.MapEntry)


_METADATA__MAPENTRY.has_options = True
_METADATA__MAPENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
# @@protoc_insertion_point(module_scope)
