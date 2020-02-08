"""

Functions for user interaction with paths and file names.

"""
from surepy.constants import QT_BINDINGS, QtBindings

if QT_BINDINGS == QtBindings.PYSIDE2:
    from PySide2.QtWidgets import QApplication, QFileDialog
elif QT_BINDINGS == QtBindings.PYQT5:
    from PyQt5.QtWidgets import QApplication, QFileDialog

__all__ = ['file_dialog']


def file_dialog(directory=None, message='Select a file...', filter='Text files (*.txt);; All files (*)'):
    """
    Select file names in a ui dialog.

    Parameters
    ----------
    directory : str or None
        directory path (Default: None)

    message : str
        Hint what to do

    filter : str
        filter for file type

    Returns
    -------
    list of str
        list with file names or empty list
    """
    if QT_BINDINGS == QtBindings.NONE:
        raise ImportError('Function requires either PySide2 or PyQt5.')

    if directory is None:
        directory_ = './'
    else:
        directory_ = str(directory)

    if QT_BINDINGS == QtBindings.PYSIDE2:
        # app = QApplication([])  # todo: this is not working - please fix!
        app = QApplication.instance()  # this is needed if the function is called twice in a row.
        if app is None:
            app = QApplication([])  # todo: [directory_] is not working - please fix!
    elif QT_BINDINGS == QtBindings.PYQT5:
        app = QApplication([directory_])

    fname = QFileDialog.getOpenFileNames(None, message, directory=directory_, filter=filter)

    if isinstance(fname, tuple):
        return fname[0]
    else:
        return str(fname)
