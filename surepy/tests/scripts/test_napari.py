"""

This module contains tests that need gui interactions and should therefore not be run automatically.

"""
import pytest
from surepy.scripts.napari import napari_

pytestmark = pytest.mark.skip('GUI tests are skipped because they would need user interaction.')


def test_napari():
    napari_(file_type=2)
