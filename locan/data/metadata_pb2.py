# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: locan/data/metadata.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19locan/data/metadata.proto\x12\nlocan.data\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1egoogle/protobuf/duration.proto\",\n\tOperation\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tparameter\x18\x02 \x01(\t\"n\n\x04\x46ile\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12#\n\x04type\x18\x03 \x01(\x0e\x32\x15.locan.data.File_type\x12\x0c\n\x04path\x18\x04 \x01(\t\x12\x0e\n\x06groups\x18\x05 \x03(\t\"R\n\x07\x41\x64\x64ress\x12\x15\n\raddress_lines\x18\x01 \x03(\t\x12\x0c\n\x04\x63ity\x18\x02 \x01(\t\x12\x11\n\tcity_code\x18\x03 \x01(\t\x12\x0f\n\x07\x63ountry\x18\x04 \x01(\t\"Z\n\x0b\x41\x66\x66iliation\x12\x11\n\tinstitute\x18\x01 \x01(\t\x12\x12\n\ndepartment\x18\x02 \x01(\t\x12$\n\x07\x61\x64\x64ress\x18\x03 \x01(\x0b\x32\x13.locan.data.Address\"\xd7\x01\n\x06Person\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x12\n\nfirst_name\x18\x03 \x01(\t\x12\x11\n\tlast_name\x18\x04 \x01(\t\x12\r\n\x05title\x18\x05 \x01(\t\x12-\n\x0c\x61\x66\x66iliations\x18\x06 \x03(\x0b\x32\x17.locan.data.Affiliation\x12$\n\x07\x61\x64\x64ress\x18\x07 \x01(\x0b\x32\x13.locan.data.Address\x12\x0e\n\x06\x65mails\x18\x08 \x03(\t\x12\r\n\x05roles\x18\t \x03(\t\"\xd3\x01\n\x12\x45xperimentalSample\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x0f\n\x07targets\x18\x03 \x03(\t\x12\x14\n\x0c\x66luorophores\x18\x04 \x03(\t\x12\x0f\n\x07\x62uffers\x18\x05 \x03(\t\x12\x34\n\x03map\x18\x06 \x03(\x0b\x32\'.locan.data.ExperimentalSample.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xc9\x01\n\x11\x45xperimentalSetup\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12.\n\roptical_units\x18\x03 \x03(\x0b\x32\x17.locan.data.OpticalUnit\x12\x33\n\x03map\x18\x04 \x03(\x0b\x32&.locan.data.ExperimentalSetup.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xe6\x01\n\x0bOpticalUnit\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12.\n\x0cillumination\x18\x03 \x01(\x0b\x32\x18.locan.data.Illumination\x12(\n\tdetection\x18\x04 \x01(\x0b\x32\x15.locan.data.Detection\x12,\n\x0b\x61\x63quisition\x18\x05 \x01(\x0b\x32\x17.locan.data.Acquisition\x12*\n\nlightsheet\x18\x06 \x01(\x0b\x32\x16.locan.data.Lightsheet\"\xec\x01\n\x0cIllumination\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x13\n\x0blightsource\x18\x03 \x01(\t\x12\r\n\x05power\x18\x04 \x01(\x02\x12\x0c\n\x04\x61rea\x18\x05 \x01(\x02\x12\x15\n\rpower_density\x18\x06 \x01(\x02\x12\x12\n\nwavelength\x18\x07 \x01(\x02\x12.\n\x03map\x18\x08 \x03(\x0b\x32!.locan.data.Illumination.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xad\x01\n\tDetection\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\"\n\x06\x63\x61mera\x18\x03 \x01(\x0b\x32\x12.locan.data.Camera\x12+\n\x03map\x18\x04 \x03(\x0b\x32\x1e.locan.data.Detection.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x92\x03\n\x06\x43\x61mera\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\r\n\x05model\x18\x04 \x01(\t\x12\x0c\n\x04gain\x18\x05 \x01(\x02\x12\x1b\n\x13\x65lectrons_per_count\x18\x06 \x01(\x02\x12\x33\n\x10integration_time\x18\x07 \x01(\x0b\x32\x19.google.protobuf.Duration\x12\x15\n\rpixel_count_x\x18\x08 \x01(\x05\x12\x15\n\rpixel_count_y\x18\t \x01(\x05\x12\x14\n\x0cpixel_size_x\x18\n \x01(\x02\x12\x14\n\x0cpixel_size_y\x18\x0b \x01(\x02\x12\x0f\n\x07\x66lipped\x18\x0c \x01(\x08\x12(\n\x03map\x18\r \x03(\x0b\x32\x1b.locan.data.Camera.MapEntry\x12\x0e\n\x06offset\x18\x0e \x01(\x02\x12\x15\n\rserial_number\x18\x0f \x01(\t\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x89\x03\n\x0b\x41\x63quisition\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x13\n\x0b\x66rame_count\x18\x03 \x01(\x05\x12\x1f\n\x17\x66rame_of_interest_first\x18\x04 \x01(\x05\x12\x1e\n\x16\x66rame_of_interest_last\x18\x05 \x01(\x05\x12.\n\ntime_start\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12,\n\x08time_end\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x13\n\x0bstack_count\x18\x08 \x01(\x05\x12\x18\n\x10stack_step_count\x18\t \x01(\x05\x12\x17\n\x0fstack_step_size\x18\n \x01(\x02\x12-\n\x03map\x18\x0b \x03(\x0b\x32 .locan.data.Acquisition.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xbe\x01\n\nLightsheet\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x0f\n\x07\x61ngle_x\x18\x03 \x01(\x02\x12\x0f\n\x07\x61ngle_y\x18\x04 \x01(\x02\x12\x0f\n\x07\x61ngle_z\x18\x05 \x01(\x02\x12,\n\x03map\x18\x06 \x03(\x0b\x32\x1f.locan.data.Lightsheet.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x96\x02\n\nExperiment\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12)\n\rexperimenters\x18\x03 \x03(\x0b\x32\x12.locan.data.Person\x12/\n\x07samples\x18\x04 \x03(\x0b\x32\x1e.locan.data.ExperimentalSample\x12-\n\x06setups\x18\x05 \x03(\x0b\x32\x1d.locan.data.ExperimentalSetup\x12,\n\x03map\x18\x06 \x03(\x0b\x32\x1f.locan.data.Experiment.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xdd\x01\n\tLocalizer\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x10\n\x08software\x18\x03 \x01(\t\x12\x1b\n\x13intensity_threshold\x18\x04 \x01(\x02\x12\x11\n\tpsf_fixed\x18\x05 \x01(\x08\x12\x10\n\x08psf_size\x18\x06 \x01(\x02\x12+\n\x03map\x18\x07 \x03(\x0b\x32\x1e.locan.data.Localizer.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xa7\x01\n\x08Relation\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x1e\n\x04\x66ile\x18\x03 \x01(\x0b\x32\x10.locan.data.File\x12*\n\x03map\x18\x04 \x03(\x0b\x32\x1d.locan.data.Relation.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xb1\x01\n\x08Property\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x0c\n\x04unit\x18\x04 \x01(\t\x12\x0c\n\x04type\x18\x05 \x01(\t\x12*\n\x03map\x18\x06 \x03(\x0b\x32\x1d.locan.data.Property.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xf5\x06\n\x08Metadata\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12\x0f\n\x07\x63omment\x18\x02 \x01(\t\x12\"\n\x06source\x18\x06 \x01(\x0e\x32\x12.locan.data.Source\x12 \n\x05state\x18\x07 \x01(\x0e\x32\x11.locan.data.State\x12&\n\x07history\x18\x08 \x03(\x0b\x32\x15.locan.data.Operation\x12\x1c\n\x14\x61ncestor_identifiers\x18\t \x03(\t\x12(\n\nproperties\x18\x12 \x03(\x0b\x32\x14.locan.data.Property\x12\x35\n\x17localization_properties\x18\x18 \x03(\x0b\x32\x14.locan.data.Property\x12\x15\n\relement_count\x18\x0b \x01(\x03\x12\x13\n\x0b\x66rame_count\x18\x0c \x01(\x03\x12\x1e\n\x04\x66ile\x18\x14 \x01(\x0b\x32\x10.locan.data.File\x12\'\n\trelations\x18\x15 \x03(\x0b\x32\x14.locan.data.Relation\x12*\n\nexperiment\x18\x16 \x01(\x0b\x32\x16.locan.data.Experiment\x12(\n\tlocalizer\x18\x17 \x01(\x0b\x32\x15.locan.data.Localizer\x12*\n\x03map\x18\x11 \x03(\x0b\x32\x1d.locan.data.Metadata.MapEntry\x12\x31\n\rcreation_time\x18\x19 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x35\n\x11modification_time\x18\x1a \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x33\n\x0fproduction_time\x18\x1b \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01J\x04\x08\x03\x10\x06J\x04\x08\n\x10\x0bJ\x04\x08\r\x10\x11J\x04\x08\x13\x10\x14R\x04unitR\tfile_typeR\tfile_pathR\x12\x65xperimental_setupR\x13\x65xperimental_sampleR\x05unitsR\rcreation_dateR\x11modification_dateR\x0fproduction_date*T\n\x06Source\x12\x12\n\x0eUNKNOWN_SOURCE\x10\x00\x12\n\n\x06\x44\x45SIGN\x10\x01\x12\x0e\n\nEXPERIMENT\x10\x02\x12\x0e\n\nSIMULATION\x10\x03\x12\n\n\x06IMPORT\x10\x04*1\n\x05State\x12\x11\n\rUNKNOWN_STATE\x10\x00\x12\x07\n\x03RAW\x10\x01\x12\x0c\n\x08MODIFIED\x10\x02*\xaa\x01\n\tFile_type\x12\x15\n\x11UNKNOWN_FILE_TYPE\x10\x00\x12\n\n\x06\x43USTOM\x10\x01\x12\x0e\n\nRAPIDSTORM\x10\x02\x12\t\n\x05\x45LYRA\x10\x03\x12\x10\n\x0cTHUNDERSTORM\x10\x04\x12\x08\n\x04\x41SDF\x10\x05\x12\x0e\n\nNANOIMAGER\x10\x06\x12\x13\n\x0fRAPIDSTORMTRACK\x10\x07\x12\x08\n\x04SMLM\x10\x08\x12\n\n\x06\x44\x45\x43ODE\x10\t\x12\x08\n\x04SMAP\x10\n')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'locan.data.metadata_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _EXPERIMENTALSAMPLE_MAPENTRY._options = None
  _EXPERIMENTALSAMPLE_MAPENTRY._serialized_options = b'8\001'
  _EXPERIMENTALSETUP_MAPENTRY._options = None
  _EXPERIMENTALSETUP_MAPENTRY._serialized_options = b'8\001'
  _ILLUMINATION_MAPENTRY._options = None
  _ILLUMINATION_MAPENTRY._serialized_options = b'8\001'
  _DETECTION_MAPENTRY._options = None
  _DETECTION_MAPENTRY._serialized_options = b'8\001'
  _CAMERA_MAPENTRY._options = None
  _CAMERA_MAPENTRY._serialized_options = b'8\001'
  _ACQUISITION_MAPENTRY._options = None
  _ACQUISITION_MAPENTRY._serialized_options = b'8\001'
  _LIGHTSHEET_MAPENTRY._options = None
  _LIGHTSHEET_MAPENTRY._serialized_options = b'8\001'
  _EXPERIMENT_MAPENTRY._options = None
  _EXPERIMENT_MAPENTRY._serialized_options = b'8\001'
  _LOCALIZER_MAPENTRY._options = None
  _LOCALIZER_MAPENTRY._serialized_options = b'8\001'
  _RELATION_MAPENTRY._options = None
  _RELATION_MAPENTRY._serialized_options = b'8\001'
  _PROPERTY_MAPENTRY._options = None
  _PROPERTY_MAPENTRY._serialized_options = b'8\001'
  _METADATA_MAPENTRY._options = None
  _METADATA_MAPENTRY._serialized_options = b'8\001'
  _SOURCE._serialized_start=4461
  _SOURCE._serialized_end=4545
  _STATE._serialized_start=4547
  _STATE._serialized_end=4596
  _FILE_TYPE._serialized_start=4599
  _FILE_TYPE._serialized_end=4769
  _OPERATION._serialized_start=106
  _OPERATION._serialized_end=150
  _FILE._serialized_start=152
  _FILE._serialized_end=262
  _ADDRESS._serialized_start=264
  _ADDRESS._serialized_end=346
  _AFFILIATION._serialized_start=348
  _AFFILIATION._serialized_end=438
  _PERSON._serialized_start=441
  _PERSON._serialized_end=656
  _EXPERIMENTALSAMPLE._serialized_start=659
  _EXPERIMENTALSAMPLE._serialized_end=870
  _EXPERIMENTALSAMPLE_MAPENTRY._serialized_start=828
  _EXPERIMENTALSAMPLE_MAPENTRY._serialized_end=870
  _EXPERIMENTALSETUP._serialized_start=873
  _EXPERIMENTALSETUP._serialized_end=1074
  _EXPERIMENTALSETUP_MAPENTRY._serialized_start=828
  _EXPERIMENTALSETUP_MAPENTRY._serialized_end=870
  _OPTICALUNIT._serialized_start=1077
  _OPTICALUNIT._serialized_end=1307
  _ILLUMINATION._serialized_start=1310
  _ILLUMINATION._serialized_end=1546
  _ILLUMINATION_MAPENTRY._serialized_start=828
  _ILLUMINATION_MAPENTRY._serialized_end=870
  _DETECTION._serialized_start=1549
  _DETECTION._serialized_end=1722
  _DETECTION_MAPENTRY._serialized_start=828
  _DETECTION_MAPENTRY._serialized_end=870
  _CAMERA._serialized_start=1725
  _CAMERA._serialized_end=2127
  _CAMERA_MAPENTRY._serialized_start=828
  _CAMERA_MAPENTRY._serialized_end=870
  _ACQUISITION._serialized_start=2130
  _ACQUISITION._serialized_end=2523
  _ACQUISITION_MAPENTRY._serialized_start=828
  _ACQUISITION_MAPENTRY._serialized_end=870
  _LIGHTSHEET._serialized_start=2526
  _LIGHTSHEET._serialized_end=2716
  _LIGHTSHEET_MAPENTRY._serialized_start=828
  _LIGHTSHEET_MAPENTRY._serialized_end=870
  _EXPERIMENT._serialized_start=2719
  _EXPERIMENT._serialized_end=2997
  _EXPERIMENT_MAPENTRY._serialized_start=828
  _EXPERIMENT_MAPENTRY._serialized_end=870
  _LOCALIZER._serialized_start=3000
  _LOCALIZER._serialized_end=3221
  _LOCALIZER_MAPENTRY._serialized_start=828
  _LOCALIZER_MAPENTRY._serialized_end=870
  _RELATION._serialized_start=3224
  _RELATION._serialized_end=3391
  _RELATION_MAPENTRY._serialized_start=828
  _RELATION_MAPENTRY._serialized_end=870
  _PROPERTY._serialized_start=3394
  _PROPERTY._serialized_end=3571
  _PROPERTY_MAPENTRY._serialized_start=828
  _PROPERTY_MAPENTRY._serialized_end=870
  _METADATA._serialized_start=3574
  _METADATA._serialized_end=4459
  _METADATA_MAPENTRY._serialized_start=828
  _METADATA_MAPENTRY._serialized_end=870
# @@protoc_insertion_point(module_scope)
