import os
import napari
from surepy.constants import QtBindings, QT_BINDINGS


def test_QT_API():
    assert os.environ['QT_API'] == QT_BINDINGS.value
