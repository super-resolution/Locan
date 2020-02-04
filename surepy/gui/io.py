"""

Functions for user interaction with paths and file names.

"""
from surepy.constants import _has_pyside2, _has_pyqt5
if _has_pyqt5:
    from PyQt5.QtWidgets import QApplication, QFileDialog
elif _has_pyside2:
    from PySide2.QtGui import QApplication, QFileDialog


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
    if not (_has_pyside2 or _has_pyqt5):
        raise ImportError('Function requires either PySide2 or PyQt5.')

    if directory is None:
        directory_ = './'
    else:
        directory_ = str(directory)

    app = QApplication([directory])
    fname = QFileDialog.getOpenFileNames(None, message, directory=directory_, filter=filter)

    if isinstance(fname, tuple):
        return fname[0]
    else:
        return str(fname)
