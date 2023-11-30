from pathlib import Path

from locan.configuration import (
    DATASETS_DIR,
    N_JOBS,
    QT_BINDING,
    RENDER_ENGINE,
)
from locan.constants import RenderEngine
from locan.dependencies import HAS_DEPENDENCY


def test_datasets_dir():
    # print(DATASETS_DIR)
    assert isinstance(DATASETS_DIR, Path)


def test_render_engine():
    assert isinstance(RENDER_ENGINE, RenderEngine)


def test_n_jobs():
    assert isinstance(N_JOBS, int)


def test_qt_bindings():
    assert "qt" in HAS_DEPENDENCY
    assert isinstance(QT_BINDING, str)

    if HAS_DEPENDENCY["qt"]:
        from qtpy import QT_VERSION

        assert QT_VERSION is not None


# tests for COLORMAP_DEFAULTS see test_colormap.py
