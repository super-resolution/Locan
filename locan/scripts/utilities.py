"""

Utility functions for scripts.

"""
from __future__ import annotations

import re


def _type_converter_rescale(input_string: str) -> str | bool | tuple[float] | None:
    if input_string == "None":
        return None
    elif input_string == "True":
        return True
    elif input_string == "False":
        return False
    else:
        pattern = re.match(
            r"\(?([0-9]*[.]?[0-9]+),?\s?([0-9]*[.]?[0-9]+)\)?", input_string
        )
        if pattern:
            return tuple(float(element) for element in pattern.groups())  # type: ignore[return-value]
        else:
            return input_string
