from locan import config
from locan.configuration import (
    QT_BINDING,
    Colormaps,
    Configuration,
    TqdmConfiguration,
)
from locan.constants import RenderEngine
from locan.dependencies import HAS_DEPENDENCY


class TestColormaps:

    def test_init(self):
        setting = Colormaps()
        assert setting.continuous == "cet_fire"

        setting.continuous = "viridis"
        assert setting.continuous == "viridis"

        setting = Colormaps(continuous="viridis")
        assert setting.continuous == "viridis"


class TestTqdmConfiguration:

    def test_init(self):
        setting = TqdmConfiguration()
        assert setting.leave is True
        assert setting.disable is False

        setting.leave = None
        assert setting.leave is None

        setting = TqdmConfiguration(leave=False)
        assert setting.leave is False


class TestConfiguration:

    def test_init(self):
        locan_config = Configuration()
        assert locan_config.name == "locanConfiguration"
        assert locan_config.cache_directory
        assert locan_config.config_directory
        assert locan_config.data_directory
        assert locan_config.datasets_directory
        assert locan_config.log_directory
        assert locan_config.test_directory
        assert locan_config.n_cpus
        assert locan_config.colormaps
        assert locan_config.render_engine
        assert locan_config.tqdm

    def test_conversion(self):
        locan_config = Configuration(
            n_cpus=-1,
            colormaps=dict(continuous="viridis"),
            render_engine="napari",
            tqdm=dict(leave=False),
        )
        # print(locan_config)
        assert locan_config.n_cpus == -1
        assert isinstance(locan_config.colormaps, Colormaps)
        assert locan_config.colormaps.continuous == "viridis"
        assert isinstance(locan_config.render_engine, RenderEngine)
        assert locan_config.render_engine == "napari"
        assert isinstance(locan_config.tqdm, TqdmConfiguration)
        assert locan_config.tqdm.leave is False


def test_locan_config():
    print(config)
    print(config.model_dump_json())
    assert config.name == "locanConfiguration"
    assert config.cache_directory
    assert config.config_directory
    assert config.data_directory
    assert config.datasets_directory
    assert config.log_directory
    assert config.test_directory
    assert config.n_cpus
    assert config.colormaps
    assert config.render_engine


def test_qt_bindings():
    assert "qt" in HAS_DEPENDENCY
    assert isinstance(QT_BINDING, str)

    if HAS_DEPENDENCY["qt"]:
        from qtpy import QT_VERSION  # type: ignore [missing-import]

        assert QT_VERSION is not None
