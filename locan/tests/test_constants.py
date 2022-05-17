from locan import ROOT_DIR
from locan import PropertyDescription, PropertyKey
from locan.constants import *

for sc in (ROOT_DIR, PROPERTY_KEYS, RAPIDSTORM_KEYS, ELYRA_KEYS, THUNDERSTORM_KEYS,
                              N_JOBS):
    assert sc

hulls = HullType
assert hulls
assert hulls.CONVEX_HULL.value == 'convex_hull'

ft = FileType
assert ft

ren_eng = RenderEngine
assert ren_eng

assert N_JOBS == 1


def test_rapidstorm_keys_are_mapped_on_valid_property_keys():
    for item in RAPIDSTORM_KEYS.values():
        assert (True if item in PROPERTY_KEYS else False)


def test_elyra_keys_are_mapped_on_valid_property_keys():
    for item in ELYRA_KEYS.values():
        assert (True if item in PROPERTY_KEYS else False)


def test_thunderstorm_keys_are_mapped_on_valid_property_keys():
    for item in THUNDERSTORM_KEYS.values():
        assert (True if item in PROPERTY_KEYS else False)


def test_root_directory():
    assert ROOT_DIR.is_dir()
    # print(ROOT_DIR.joinpath('tests/'))
    # print(type(ROOT_DIR))
    assert ROOT_DIR.joinpath('tests/').is_dir()


def test_PropertyDescription():
    prop = PropertyDescription(name="prop_name", type='integer', unit_SI="m", unit="nm", description="something")
    assert repr(prop) == "PropertyDescription(name='prop_name', type='integer', unit_SI='m', unit='nm', " \
                         "description='something')"
    assert prop.name == "prop_name"


def test_PropertyKey():
    for element in PropertyKey:
        assert element.name == element.value.name

    assert all(key in PropertyKey.coordinate_labels()
               for key in [PropertyKey.position_x, PropertyKey.position_y, PropertyKey.position_z])

    PropertyKey.position_x.value.unit = "nm"
    assert PropertyKey.position_x.value.unit == "nm"

    string_ = PropertyKey.summary()
    # print(string_)
    assert string_[:5] == "index"

    from locan.constants import PROPERTY_KEYS_deprecated
    assert PROPERTY_KEYS_deprecated == PROPERTY_KEYS
