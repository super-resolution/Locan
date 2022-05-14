from copy import copy
import pytest

from locan import metadata_to_formatted_string
from locan.data import metadata_pb2
from locan.data.metadata_utils import _modify_meta


@pytest.fixture
def metadata():
    metadata = metadata_pb2.Metadata()
    optical_unit = metadata.experiment.setups.add().optical_units.add()
    optical_unit.detection.camera.integration_time.FromMilliseconds(20)

    metadata.identifier = "111"
    metadata.comment = "comment"
    metadata.creation_time.FromJsonString("1111-11-11T11:11:11+01:00")

    return metadata


def test__modify_meta(metadata, locdata_2d):
    new_locdata = copy(locdata_2d)
    new_metadata = _modify_meta(locdata_2d, new_locdata,
                                function_name="test_function",
                                parameter=dict(first=1, second=2),
                                meta=None)

    assert metadata_to_formatted_string(message=new_metadata.history[-1], as_one_line=True) == \
           "name: \"test_function\" parameter: \"{\\\'first\\\': 1, \\\'second\\\': 2}\""


def test_metadata_to_formatted_string(metadata):
    results_string = metadata_to_formatted_string(message=metadata, as_one_line=True)
    results_string_expected = 'identifier: "111" comment: "comment" experiment { setups { optical_units { detection ' \
                              '{ camera { integration_time { 0.020s } } } } } } creation_time { 1111-11-11T10:11:11Z }'
    assert results_string == results_string_expected
