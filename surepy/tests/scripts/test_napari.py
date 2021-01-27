"""

This module contains tests that need gui interactions and should therefore not be run automatically.

"""
import pytest
from surepy.scripts.script_napari import sc_napari

pytestmark = pytest.mark.skip('GUI tests are skipped because they need user interaction.')


def test_napari():
    sc_napari(file_type=2, bin_size=10, rescale='equal')
