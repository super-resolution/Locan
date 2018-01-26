import surepy.constants

def test_rapidstorm_keys_are_mapped_on_valid_property_keys():
    for item in surepy.constants.RAPIDSTORM_KEYS.values():
        assert (True if item in surepy.constants.PROPERTY_KEYS else False)
