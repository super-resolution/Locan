"""

This module contains tests that need gui interactions.

"""
import pytest

from locan import ROOT_DIR
from locan.dependencies import HAS_DEPENDENCY
from locan.gui.io import file_dialog, set_file_path_dialog


def test_file_dialog_no_qt():
    if not HAS_DEPENDENCY["qt"]:
        with pytest.raises(ImportError):
            file_dialog()


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["qt"], reason="requires qt_binding.")
def test_file_dialog():
    file_path = ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_data.txt"
    result = file_dialog(directory=file_path, message="Select single file")
    # result = file_dialog()
    # print(result)
    assert len(result) == 1
    assert result == str(file_path).replace("\\", "/") or result[0] == str(
        file_path
    ).replace("\\", "/")


@pytest.mark.gui
@pytest.mark.skipif(not HAS_DEPENDENCY["qt"], reason="requires qt_binding.")
def test_set_file_path_dialog():
    file_path = ROOT_DIR / "tests/test_data/rapidSTORM_dstorm_data.txt"
    result = set_file_path_dialog(directory=file_path, message="Set file path...")
    # result = set_file_path_dialog()
    # print(result)
    assert isinstance(result, str)
