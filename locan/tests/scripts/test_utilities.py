from locan.scripts.utilities import _type_converter_rescale


def test__type_converter_rescale():
    return_value = _type_converter_rescale(input_string="None")
    assert return_value is None
    return_value = _type_converter_rescale(input_string="True")
    assert return_value is True
    return_value = _type_converter_rescale(input_string="False")
    assert return_value is False
    return_value = _type_converter_rescale(input_string="EQUALIZE")
    assert return_value == "EQUALIZE"
    return_value = _type_converter_rescale(input_string="0.2 0.9")
    assert return_value == (0.2, 0.9)
    return_value = _type_converter_rescale(input_string="0.2, 0.9")
    assert return_value == (0.2, 0.9)
    return_value = _type_converter_rescale(input_string="0.2 0.9 1")
    assert return_value == (0.2, 0.9)
