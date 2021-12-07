"""

This module contains tests that need gui interactions and should therefore not be run automatically.

"""
import pytest
from locan import ROOT_DIR, QT_BINDINGS
from locan.gui.io import file_dialog

pytestmark = pytest.mark.gui


def test_file_dialog():
    print(QT_BINDINGS)
    result = file_dialog(directory=ROOT_DIR, message='Select single file')
    # result = file_dialog()
    print(result)
    assert (len(result) == 1)
