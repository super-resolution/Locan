import pytest

from locan.data.validation import _check_loc_properties


def test__check_loc_properties(locdata_2d):
    assert _check_loc_properties(locdata_2d) == locdata_2d.coordinate_keys
    assert _check_loc_properties(locdata_2d, loc_properties=["intensity"]) == [
        "intensity"
    ]
    with pytest.raises(ValueError):
        _check_loc_properties(locdata_2d, loc_properties=["undefined"])
