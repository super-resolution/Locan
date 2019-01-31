from surepy.constants import (ROOT_DIR, PROPERTY_KEYS, HULL_KEYS, RAPIDSTORM_KEYS, ELYRA_KEYS, THUNDERSTORM_KEYS,
                              N_JOBS, File_type)

for sc in (ROOT_DIR, PROPERTY_KEYS, HULL_KEYS, RAPIDSTORM_KEYS, ELYRA_KEYS, THUNDERSTORM_KEYS,
                              N_JOBS):
    assert sc

ft = File_type
assert ft

assert N_JOBS == 1

def test_rapidstorm_keys_are_mapped_on_valid_property_keys():
    for item in RAPIDSTORM_KEYS.values():
        assert (True if item in PROPERTY_KEYS else False)

#todo tests for thunderstorm
