from locan import FileType, HullType, RenderEngine, PropertyDescription, PropertyKey, PROPERTY_KEYS, RAPIDSTORM_KEYS, ELYRA_KEYS, THUNDERSTORM_KEYS
from locan.constants import N_JOBS

for sc in (PROPERTY_KEYS, RAPIDSTORM_KEYS, ELYRA_KEYS, THUNDERSTORM_KEYS,
                              N_JOBS):
    assert sc

hulls = HullType
assert hulls
assert hulls.CONVEX_HULL.value == 'convex_hull'


def test_FileType():
    from locan.data.metadata_pb2 import File_type
    for ft_enum, ft_pb in zip(FileType, File_type.items()):
        assert ft_enum.name == ft_pb[0]
        assert ft_enum.value == ft_pb[1]


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


def test_PropertyDescription():
    prop = PropertyDescription(name="prop_name", type='integer', unit_SI="m", unit="nm", description="something")
    assert repr(prop) == "PropertyDescription(name='prop_name', type='integer', unit_SI='m', unit='nm', " \
                         "description='something')"
    assert prop.name == "prop_name"


def test_PropertyKey():
    assert "index" in PropertyKey.__members__
    for element in PropertyKey:
        assert element.name == element.value.name

    assert all(key in PropertyKey.coordinate_labels()
               for key in [PropertyKey.position_x, PropertyKey.position_y, PropertyKey.position_z])

    PropertyKey.position_x.value.unit = "nm"
    assert PropertyKey.position_x.value.unit == "nm"

    string_ = PropertyKey.summary()
    # print(string_)
    assert string_[:5] == "index"


