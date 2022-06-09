"""

This module contains tests that need gui interactions and should therefore not be run automatically.

"""
import pytest

from locan import QT_BINDINGS, ROOT_DIR
from locan.gui.io import file_dialog

pytestmark = pytest.mark.gui


def test_file_dialog():
    print(QT_BINDINGS)
    result = file_dialog(directory=ROOT_DIR, message="Select single file")
    # result = file_dialog()
    print(result)
    assert len(result) == 1
