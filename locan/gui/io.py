"""

Functions for user interaction with paths and file names.

Note
-----
These functions start a QT app and might interfere with previously started apps.

"""
from __future__ import annotations

from locan.dependencies import HAS_DEPENDENCY, needs_package

if HAS_DEPENDENCY["qt"]:
    from qtpy.QtWidgets import QApplication, QFileDialog

from locan.configuration import QT_BINDING
from locan.dependencies import QtBindings

__all__: list[str] = ["file_dialog", "set_file_path_dialog"]


@needs_package("qt")
def file_dialog(
    directory: str | None = None,
    message: str = "Select a file...",
    filter: str = "Text files (*.txt);; All files (*)",
) -> str | list[str]:
    """
    Select file names in a ui dialog.

    Parameters
    ----------
    directory
        directory path to start dialog in. If None the current directory is used.
    message
        Hint what to do
    filter
        filter for file type

    Returns
    -------
    str | list[str]
        list with file names or empty list
    """
    if directory is None:
        directory_ = "./"
    else:
        directory_ = str(directory)

    if QT_BINDING == QtBindings.PYSIDE2:
        app = (
            QApplication.instance()
        )  # this is needed if the function is called twice in a row.
        if app is None:
            app = QApplication([])  # todo: [directory_] is not working - please fix!
    else:
        app = QApplication([directory_])

    fname = QFileDialog.getOpenFileNames(
        None,
        message,
        directory_,
        filter=filter
        # kwargs: parent, message, directory, filter
        # but kw_names are different for different qt_bindings
    )

    if isinstance(fname, tuple):
        return_value: str | list[str] = fname[0]
    else:
        return_value = str(fname)
    return return_value


@needs_package("qt")
def set_file_path_dialog(
    directory: str | None = None,
    message: str = "Set a file path...",
    filter: str = "All files (*)",
) -> str | list[str]:
    """
    Set file path (path/name.suffix) in a ui dialog.

    Parameters
    ----------
    directory
        directory path to start dialog in. If None the current directory is used.
    message
        Hint what to do
    filter
        filter for file type

    Returns
    -------
    str
        new file path
    """
    if directory is None:
        directory_ = "./"
    else:
        directory_ = str(directory)

    if QT_BINDING == QtBindings.PYSIDE2:
        app = (
            QApplication.instance()
        )  # this is needed if the function is called twice in a row.
        if app is None:
            app = QApplication([])  # todo: [directory_] is not working - please fix!
    else:
        app = QApplication([directory_])

    fname = QFileDialog.getSaveFileName(
        None,
        message,
        directory_,
        filter=filter
        # kwargs: parent, message, directory, filter
        # but kw_names are different for different qt_bindings
    )

    if isinstance(fname, tuple):
        return_value: str | list[str] = fname[0]
    else:
        return_value = str(fname)
    return return_value
