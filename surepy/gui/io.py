'''

This module provides methods for user interaction with paths and file names.

'''

def file_dialog(dir=None, filter="Text files (*.txt);; All files (*)"):
    """
    Select a file via a dialog and return the file name.
    """
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog
    except ImportError:
        try:
            from PyQt4.QtGui import QApplication, QFileDialog
        except ImportError:
            from PySide.QtGui import QApplication, QFileDialog

    if dir is None:
        dir = './'

    app = QApplication([dir])
    fname = QFileDialog.getOpenFileName(None, "Select a file...",
                                        dir, filter=filter)

    if isinstance(fname, tuple):
        return fname[0]
    else:
        return str(fname)
