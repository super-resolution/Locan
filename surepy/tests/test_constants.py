# from surepy.constants import (ROOT_DIR, PROPERTY_KEYS, HULL_KEYS, RAPIDSTORM_KEYS, ELYRA_KEYS, THUNDERSTORM_KEYS,
#                               N_JOBS, FileType, LOCDATA_ID)

from surepy.constants import *

# for sc in (ROOT_DIR, PROPERTY_KEYS, HULL_KEYS, RAPIDSTORM_KEYS, ELYRA_KEYS, THUNDERSTORM_KEYS,
#                               N_JOBS):
#     assert sc
#
# ft = FileType
# assert ft
#
# assert N_JOBS == 1


def test_rapidstorm_keys_are_mapped_on_valid_property_keys():
    for item in RAPIDSTORM_KEYS.values():
        assert (True if item in PROPERTY_KEYS else False)


# def test_elyra_keys_are_mapped_on_valid_property_keys():
#     for item in ELYRA_KEYS.values():
#         assert (True if item in PROPERTY_KEYS else False)


def test_thunderstorm_keys_are_mapped_on_valid_property_keys():
    for item in THUNDERSTORM_KEYS.values():
        assert (True if item in PROPERTY_KEYS else False)


def test_root_directory():
    assert ROOT_DIR.is_dir()
    print(ROOT_DIR.joinpath('tests/'))
    print(type(ROOT_DIR))
    assert ROOT_DIR.joinpath('tests/').is_dir()
