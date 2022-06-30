"""

This module contains tests that need gui interactions.

"""
import pytest

from locan import ROOT_DIR
from locan.dependencies import HAS_DEPENDENCY
from locan.gui.io import file_dialog


def test_file_dialog_no_qt():
    if not HAS_DEPENDENCY["qt"]:
        with pytest.raises(ImportError):
            file_dialog()


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["qt"], reason="requires qt_binding.")
def test_file_dialog():
    result = file_dialog(directory=ROOT_DIR, message="Select single file")
    # result = file_dialog()
    # print(result)
    assert len(result) == 1
