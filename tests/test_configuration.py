from locan import config
from locan.configuration import (
    QT_BINDING,
    RENDER_ENGINE,
)
from locan.constants import RenderEngine
from locan.dependencies import HAS_DEPENDENCY


def test_locan_config():
    # print(lc.config.model_dump_json())
    assert config.name == "locanConfiguration"
    assert config.cache_directory
    assert config.config_directory
    assert config.data_directory
    assert config.datasets_directory
    assert config.log_directory
    assert config.test_directory
    assert config.n_cpus


def test_render_engine():
    assert isinstance(RENDER_ENGINE, RenderEngine)


def test_qt_bindings():
    assert "qt" in HAS_DEPENDENCY
    assert isinstance(QT_BINDING, str)

    if HAS_DEPENDENCY["qt"]:
        from qtpy import QT_VERSION  # type: ignore [missing-import]

        assert QT_VERSION is not None


# tests for COLORMAP_DEFAULTS see test_colormap.py
