"""

This module contains tests that need gui interactions and should therefore not be run automatically.

"""
import pytest
from surepy.scripts.script_rois import sc_draw_roi_napari

#pytestmark = pytest.mark.skip('GUI tests are skipped because they would need user interaction.')


def test_draw_roi_napari():
    sc_draw_roi_napari(file_type=2, bin_size=10)
