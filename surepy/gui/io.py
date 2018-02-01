'''

This module provides methods for user interaction with paths and file names.

'''

def file_dialog(directory=None, message='Select a file...', filter='Text files (*.txt);; All files (*)'):
    """
    Select file names in a ui dialog.

    Parameter
    ---------
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
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog
    except ImportError:
        try:
            from PyQt4.QtGui import QApplication, QFileDialog
        except ImportError:
            from PySide.QtGui import QApplication, QFileDialog

    if directory is None:
        dir = './'

    app = QApplication([directory])
    fname = QFileDialog.getOpenFileNames(None, message, directory, filter=filter)

    if isinstance(fname, tuple):
        return fname[0]
    else:
        return str(fname)
