import matplotlib.colors as mcolors

from locan import (
    DECODE_KEYS,
    ELYRA_KEYS,
    NANOIMAGER_KEYS,
    PROPERTY_KEYS,
    RAPIDSTORM_KEYS,
    SMAP_KEYS,
    SMLM_KEYS,
    THUNDERSTORM_KEYS,
    ColorMaps,
    FileType,
    HullType,
    PropertyDescription,
    PropertyKey,
    RenderEngine,
)
from locan.dependencies import HAS_DEPENDENCY


def test_PropertyDescription():
    prop = PropertyDescription(
        name="prop_name",
        type="integer",
        unit_SI="m",
        unit="nm",
        description="something",
    )
    assert (
        repr(prop)
        == "PropertyDescription(name='prop_name', type='integer', unit_SI='m', unit='nm', "
        "description='something')"
    )
    assert prop.name == "prop_name"


def test_PropertyKey():
    assert "index" in PropertyKey.__members__
    for element in PropertyKey:
        assert element.name == element.value.name

    assert all(
        key in PropertyKey.coordinate_labels()
        for key in [
            PropertyKey.position_x,
            PropertyKey.position_y,
            PropertyKey.position_z,
        ]
    )

    PropertyKey.position_x.value.unit = "nm"
    assert PropertyKey.position_x.value.unit == "nm"

    string_ = PropertyKey.summary()
    # print(string_)
    assert string_[:5] == "index"


def test_PROPERTY_KEYS():
    assert all(key in PropertyKey._member_names_ for key in PROPERTY_KEYS.keys())


def test_HullType():
    values = [item.value for item in HullType]
    assert all(
        value in values
        for value in [
            "bounding_box",
            "convex_hull",
            "oriented_bounding_box",
            "alpha_shape",
        ]
    )


def test_FileType():
    from locan.data.metadata_pb2 import File_type

    for ft_enum, ft_pb in zip(FileType, File_type.items()):
        assert ft_enum.name == ft_pb[0]
        assert ft_enum.value == ft_pb[1]


def test_RenderEngine():
    if not HAS_DEPENDENCY["napari"]:
        assert all(key in RenderEngine._member_names_ for key in ["MPL"])
    else:
        assert all(key in RenderEngine._member_names_ for key in ["MPL", "NAPARI"])


def test_ColorMaps():
    assert all([isinstance(item.value, mcolors.Colormap) for item in ColorMaps])


def test_keys_are_mapped_on_valid_property_keys():
    assert all(PropertyKey[value] for value in RAPIDSTORM_KEYS.values())
    assert all(PropertyKey[value] for value in ELYRA_KEYS.values())
    assert all(PropertyKey[value] for value in THUNDERSTORM_KEYS.values())
    assert all(PropertyKey[value] for value in NANOIMAGER_KEYS.values())
    assert all(PropertyKey[value] for value in SMLM_KEYS.values())
    assert all(PropertyKey[value] for value in DECODE_KEYS.values())
    assert all(PropertyKey[value] for value in SMAP_KEYS.values())
