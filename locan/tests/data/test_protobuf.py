import pytest

from locan import ROOT_DIR
from locan.data import metadata_pb2


@pytest.fixture
def metadata_pb2_Metadata_v0p11():
    """
    Fixture for protobuf message metadata_pb2.Metadata in locan <= 0.11
    """
    metadata = metadata_pb2.Metadata()

    metadata.identifier = "111"
    metadata.comment = "comment"
    metadata.creation_date = "1111-11-11 11:11:11 +0100"
    metadata.modification_date = "1111-11-11 11:11:11 +0100"
    metadata.production_date = "1111-11-11 11:11:11 +0100"
    metadata.source = 0
    metadata.state = 0
    metadata.history.add(name="function_name", parameter="parameters")
    metadata.ancestor_identifiers.append("110")
    metadata.unit.add(property="name", unit="s")
    metadata.unit.add(property="name_2", unit="m")
    metadata.element_count = 111
    metadata.frame_count = 111
    metadata.file_type = 0
    metadata.file_path = "file_path"
    metadata.experimental_setup["test_key"] = "setup"
    metadata.experimental_sample["test_key"] = "sample"
    metadata.map["test_key"] = "map_value"
    metadata.map["test_key_2"] = "map_value_2"

    return metadata


@pytest.fixture
def metadata_pb2_Metadata_v0p11_latest():
    """
    Fixture for protobuf message metadata_pb2.Metadata in locan <= 0.11
    as usable in latest protobuf definition.
    """
    metadata = metadata_pb2.Metadata()

    metadata.identifier = "111"
    metadata.comment = "comment"
    # metadata.creation_date = "1111-11-11 11:11:11 +0100"
    # metadata.modification_date = "1111-11-11 11:11:11 +0100"
    # metadata.production_date = "1111-11-11 11:11:11 +0100"
    metadata.source = 0
    metadata.state = 0
    metadata.history.add(name="function_name", parameter="parameters")
    metadata.ancestor_identifiers.append("110")
    # metadata.unit.add(property="name", unit="s")
    # metadata.unit.add(property="name_2", unit="m")
    metadata.element_count = 111
    metadata.frame_count = 111
    # metadata.file_type = 0
    # metadata.file_path = "file_path"
    # metadata.experimental_setup["test_key"] = "setup"
    # metadata.experimental_sample["test_key"] = "sample"
    metadata.map["test_key"] = "map_value"
    metadata.map["test_key_2"] = "map_value_2"

    return metadata


@pytest.fixture
def metadata_pb2_Metadata_v0p12():
    """
    Fixture for protobuf message metadata_pb2.Metadata in locan >= 0.12
    """
    operation = metadata_pb2.Operation()

    operation.name = "function_name"
    operation.parameter = "parameters"

    file = metadata_pb2.File()

    file.identifier = "111"
    file.comment = "comment"
    file.type = 0
    file.path = "file_path"
    file.groups.append("group 1")

    address = metadata_pb2.Address()

    address.address_lines.append("address_line")
    address.city = "city"
    address.city_code = "city_code"
    address.country = "country"

    affiliation = metadata_pb2.Affiliation()

    affiliation.institute = "institute"
    affiliation.department = "department"
    affiliation.address.CopyFrom(address)

    person = metadata_pb2.Person()

    person.identifier = "111"
    person.comment = "comment"
    person.first_name = "first_name"
    person.last_name = "last_name"
    person.title = "title"
    person.affiliations.append(affiliation)
    person.address.CopyFrom(address)
    person.emails.append("email_1")
    person.roles.append("roles_1")

    experimental_sample = metadata_pb2.ExperimentalSample()

    experimental_sample.identifier = "111"
    experimental_sample.comment = "comment"
    experimental_sample.targets.append("targets")
    experimental_sample.fluorophores.append("fluorophores")
    experimental_sample.buffers.append("buffers")
    experimental_sample.map["test_key"] = "map_value"

    illumination = metadata_pb2.Illumination()

    illumination.identifier = "111"
    illumination.comment = "comment"
    illumination.lightsource = "lightsource"
    illumination.power = 1.1
    illumination.area = 1.1
    illumination.power_density = 1.1
    illumination.wavelength = 1.1
    illumination.map["test_key"] = "map_value"

    camera = metadata_pb2.Camera()

    camera.identifier = "111"
    camera.comment = "comment"
    camera.name = "name"
    camera.model = "model"
    camera.gain = 1.1
    camera.electrons_per_count = 1.1
    camera.integration_time.FromMilliseconds(10)
    camera.pixel_count_x = 1
    camera.pixel_count_y = 1
    camera.pixel_size_x = 1.1
    camera.pixel_size_y = 1.1
    camera.flipped = True
    camera.map["test_key"] = "map_value"

    detection = metadata_pb2.Detection()

    detection.identifier = "111"
    detection.comment = "comment"
    detection.camera.CopyFrom(camera)
    detection.map["test_key"] = "map_value"

    acquisition = metadata_pb2.Acquisition()

    acquisition.identifier = "111"
    acquisition.comment = "comment"
    acquisition.frame_count = 1
    acquisition.frame_of_interest_first = 1
    acquisition.frame_of_interest_last = 2
    acquisition.time_start.FromJsonString("1970-01-01T00:00:00Z")
    acquisition.time_end.FromJsonString("1970-01-01T00:01:00Z")
    acquisition.stack_count = 1
    acquisition.stack_step_count = 1
    acquisition.stack_step_size = 1.1
    acquisition.map["test_key"] = "map_value"

    lightsheet = metadata_pb2.Lightsheet()

    lightsheet.identifier = "111"
    lightsheet.comment = "comment"
    lightsheet.angle_x = 1.1
    lightsheet.angle_y = 1.1
    lightsheet.angle_z = 1.1
    lightsheet.map["test_key"] = "map_value"

    optical_unit = metadata_pb2.OpticalUnit()

    optical_unit.identifier = "111"
    optical_unit.comment = "comment"
    optical_unit.illumination.CopyFrom(illumination)
    optical_unit.detection.CopyFrom(detection)
    optical_unit.acquisition.CopyFrom(acquisition)
    optical_unit.lightsheet.CopyFrom(lightsheet)

    experimental_setup = metadata_pb2.ExperimentalSetup()

    experimental_setup.identifier = "111"
    experimental_setup.comment = "comment"
    experimental_setup.optical_units.append(optical_unit)
    experimental_setup.map["test_key"] = "map_value"

    experiment = metadata_pb2.Experiment()

    experiment.identifier = "111"
    experiment.comment = "comment"
    experiment.experimenters.append(person)
    experiment.samples.append(experimental_sample)
    experiment.setups.append(experimental_setup)
    experiment.map["test_key"] = "map_value"

    localizer = metadata_pb2.Localizer()

    localizer.identifier = "111"
    localizer.comment = "comment"
    localizer.software = "software"
    localizer.intensity_threshold = 1.1
    localizer.psf_fixed = True
    localizer.psf_size = 1.1
    localizer.map["test_key"] = "map_value"

    relation = metadata_pb2.Relation()

    relation.identifier = "111"
    relation.comment = "comment"
    relation.file.CopyFrom(file)
    relation.map["test_key"] = "map_value"

    metadata = metadata_pb2.Metadata()

    metadata.identifier = "111"
    metadata.comment = "comment"
    metadata.creation_time.FromJsonString("1111-11-11T11:11:11+01:00")
    metadata.modification_time.FromJsonString("1111-11-11T11:11:11+01:00")
    metadata.production_time.FromJsonString("1111-11-11T11:11:11+01:00")
    metadata.source = 0
    metadata.state = 0
    metadata.history.append(operation)
    metadata.ancestor_identifiers.append("110")
    metadata.element_count = 1
    metadata.frame_count = 1
    metadata.properties.add(name="position_x", unit="nm", type="float")
    metadata.localization_properties.add(name="position_x", unit="nm", type="float")
    metadata.file.CopyFrom(file)
    metadata.relations.append(relation)
    metadata.experiment.CopyFrom(experiment)
    metadata.localizer.CopyFrom(localizer)
    metadata.map["test_key"] = "map_value"

    return metadata


@pytest.mark.skip("Use only to write new file with protobuf message!")
def test_save_proto_message_locdata_metadata(metadata_pb2_Metadata_v0p12):
    """
    Write only to produce new file with metadata message. Keep old files to test new proto definitions.
    Do NOT overwrite the old versions:
    1) "tests/test_data/protobuf_message_metadata_pb2.Metadata_v0p11"
    2) "tests/test_data/protobuf_message_metadata_pb2.Metadata_v0p12"
    """
    metadata = metadata_pb2_Metadata_v0p12

    print(metadata)

    path = ROOT_DIR / "tests/test_data/protobuf_message_metadata_pb2.Metadata_v0px"
    with open(path, "wb") as file:
        file.write(metadata.SerializeToString())

    metadata_new = metadata_pb2.Metadata()
    with open(path, "rb") as file:
        metadata_new.ParseFromString(file.read())

    assert metadata_new == metadata


def test_read_proto_message_locdata_metadata_v0p11(metadata_pb2_Metadata_v0p11_latest):
    metadata = metadata_pb2_Metadata_v0p11_latest
    metadata_new = metadata_pb2.Metadata()
    path = ROOT_DIR / "tests/test_data/protobuf_message_metadata_pb2.Metadata_v0p11"
    with open(path, "rb") as file:
        metadata_new.ParseFromString(file.read())

    for key in metadata_new.DESCRIPTOR.fields_by_name.keys():
        assert key in metadata.DESCRIPTOR.fields_by_name.keys()
        if key == "map":
            assert dict(getattr(metadata_new, key)) == dict(getattr(metadata, key))
        else:
            assert getattr(metadata_new, key) == getattr(metadata, key)


def test_read_proto_message_locdata_metadata(metadata_pb2_Metadata_v0p12):
    metadata = metadata_pb2_Metadata_v0p12

    # print(metadata)

    path = ROOT_DIR / "tests/test_data/protobuf_message_metadata_pb2.Metadata_v0p12"
    metadata_new = metadata_pb2.Metadata()
    with open(path, "rb") as file:
        metadata_new.ParseFromString(file.read())

    # print(metadata_new)

    for key in metadata_new.DESCRIPTOR.fields_by_name.keys():
        assert key in metadata.DESCRIPTOR.fields_by_name.keys()

    assert metadata_new == metadata
