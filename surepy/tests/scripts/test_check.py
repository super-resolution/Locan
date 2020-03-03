"""

This module contains tests that need gui interactions and should therefore not be run automatically.

"""
import pytest
from surepy.scripts.script_check import sc_check

pytestmark = pytest.mark.skip('GUI tests are skipped because they would need user interaction.')


def test_check_napari():
    sc_check(pixel_size=130)
