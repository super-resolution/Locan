import os

from locan.constants import QT_BINDINGS


def test_QT_API():
    if 'QT_API' in os.environ:
        assert os.environ['QT_API'] == QT_BINDINGS.value
