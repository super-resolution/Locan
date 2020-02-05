"""

This module contains tests that need gui interactions and should therefore not be run automatically.

"""
import pytest
from surepy.constants import ROOT_DIR
from surepy.gui.io import file_dialog
from surepy.constants import QT_BINDINGS

pytestmark = pytest.mark.skip('GUI tests are skipped because they would need user interaction.')


def test_file_dialog():
    print(QT_BINDINGS)
    result = file_dialog(directory=ROOT_DIR, message='Select single file')
    # print(result)
    assert (len(result) == 1)
