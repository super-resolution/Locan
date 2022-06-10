"""

Functions for user interaction with paths and file names.

"""
from qtpy.QtWidgets import QApplication, QFileDialog

from locan.configuration import QT_BINDING
from locan.dependencies import QtBindings

__all__ = ["file_dialog"]


def file_dialog(
    directory=None,
    message="Select a file...",
    filter="Text files (*.txt);; All files (*)",
):
    """
    Select file names in a ui dialog.

    Parameters
    ----------
    directory : str or None
        directory path to start dialog in. If None the current directory is used.

    message : str
        Hint what to do

    filter : str
        filter for file type

    Returns
    -------
    list of str
        list with file names or empty list
    """
    if QT_BINDING == QtBindings.NONE:
        raise ImportError("Function requires either PySide2 or PyQt5.")

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
        None, message, directory=directory_, filter=filter
    )

    if isinstance(fname, tuple):
        return fname[0]
    else:
        return str(fname)
