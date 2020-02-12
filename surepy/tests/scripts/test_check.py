"""

This module contains tests that need gui interactions and should therefore not be run automatically.

"""
import pytest
from surepy.scripts.check import check_

pytestmark = pytest.mark.skip('GUI tests are skipped because they would need user interaction.')


def test_check_napari():
    check_(pixel_size=130)
