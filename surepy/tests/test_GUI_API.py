import os
import napari
from surepy.constants import QtBindings, QT_BINDINGS


def test_QT_API():
    if QT_BINDINGS == QtBindings.PYSIDE2:
        assert os.environ['QT_API'] == 'pyside2'
    elif QT_BINDINGS == QtBindings.PYQT5:
        assert os.environ['QT_API'] == 'pyqt5'
