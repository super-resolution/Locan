import os
from pathlib import Path

import pytest

from locan.configuration import (
    CONFIGURATION_FILE_HIERARCHY,
    Configuration,
    ConfigurationBase,
    get_configuration_file_path,
)
from tests import TEST_DIR

# ruff: noqa: S108


def test_CONFIGURATION_FILE_HIERARCHY():
    assert isinstance(CONFIGURATION_FILE_HIERARCHY[0], Path)


def test_get_configuration_file_path():
    file_path = get_configuration_file_path()
    if file_path is not None:
        assert isinstance(file_path, Path)


class TestConfigurationBase:

    def test_init(self):
        class Configuration(ConfigurationBase):
            pass

        config = Configuration(
            name="SomeConfiguration",
            version="111",
        )

        assert config.name == "SomeConfiguration"
        assert config.version == "111"
        assert config.asdict() == {"name": "SomeConfiguration", "version": "111"}

    def test_from_toml(self):
        class Configuration(ConfigurationBase):
            pass

        path = TEST_DIR / "test_data" / "locan_config.toml"
        config = Configuration.from_toml(path)
        assert config.name == "TestConfiguration"


class TestConfiguration:

    def test_init(self):
        config = Configuration()
        # print(config.model_dump_json())
        # assert config.name == "locanConfiguration"
        # assert isinstance(config.cache_directory, Path)
        # assert config.cache_directory == Path.home() / ".cache" / "locan"
        # assert config.config_directory == Path.home() / ".config" / "locan"
        # assert config.data_directory == Path.home() / ".local/share" / "locan"
        # assert config.log_directory == Path.home() / ".local/state" / "locan"
        # assert config.test_directory == Path.cwd().resolve() / "tests/"

        assert config.name == "locanConfiguration"
        assert isinstance(config.cache_directory, Path)
        assert isinstance(config.config_directory, Path)
        assert isinstance(config.data_directory, Path)
        assert isinstance(config.log_directory, Path)
        assert isinstance(config.test_directory, Path)

        config = Configuration(
            name="OtherName",
            # str is not expected but will be converted to Path
            cache_directory=str(Path.home() / ".cache" / "other_cache"),
            config_directory=Path.home() / ".config" / "other_config",
            data_directory=Path.home() / ".local/share" / "other_data",
            log_directory=Path.home() / ".local/state" / "other_log",
            test_directory=TEST_DIR / "other_tests",
        )
        # print(config.model_dump_json())
        assert config.name == "OtherName"
        assert isinstance(config.cache_directory, Path)
        assert config.cache_directory == Path.home() / ".cache" / "other_cache"
        assert config.config_directory == Path.home() / ".config" / "other_config"
        assert config.data_directory == Path.home() / ".local/share" / "other_data"
        assert config.log_directory == Path.home() / ".local/state" / "other_log"
        assert config.test_directory == TEST_DIR / "other_tests"

    # noqa: S108
    def test_config_from_environment_variable(self):
        config = Configuration()

        os.environ["locan_NAME"] = "other_name"
        os.environ["locan_CACHE_DIRECTORY"] = "/tmp/.cache"
        os.environ["locan_CONFIG_DIRECTORY"] = "/tmp/.config"
        os.environ["locan_DATA_DIR"] = "/tmp/.data"
        os.environ["locan_LOG_DIR"] = "/tmp/.logs"
        os.environ["locan_TEST_DIR"] = "/tmp/tests"
        os.environ["locan_COLORMAPS_CONTINUOUS"] = "viridis"

        config.__init__()
        # print(config.model_dump_json())
        assert config.name == "other_name"
        assert config.cache_directory == Path("/tmp/.cache")
        assert config.config_directory == Path("/tmp/.config")
        assert config.data_directory == Path("/tmp/.data")
        assert config.log_directory == Path("/tmp/.logs")
        assert config.test_directory == Path("/tmp/tests")
        assert config.colormaps.continuous == "viridis"

        config.__init__(
            name="yet_another_name",
            cache_directory=Path.home() / ".cache" / "other_cache",
            config_directory=Path.home() / ".config" / "other_config",
            data_directory=Path.home() / ".local/share" / "other_data",
            log_directory=Path.home() / ".local/state" / "other_log",
            test_directory=TEST_DIR / "other_tests",
        )
        # print(config.model_dump_json())
        assert config.name == "yet_another_name"
        assert config.cache_directory == Path.home() / ".cache" / "other_cache"
        assert config.config_directory == Path.home() / ".config" / "other_config"
        assert config.data_directory == Path.home() / ".local/share" / "other_data"
        assert config.log_directory == Path.home() / ".local/state" / "other_log"
        assert config.test_directory == TEST_DIR / "other_tests"

        # cleanup
        del os.environ["locan_NAME"]
        del os.environ["locan_CACHE_DIRECTORY"]
        del os.environ["locan_CONFIG_DIRECTORY"]
        del os.environ["locan_DATA_DIR"]
        del os.environ["locan_LOG_DIR"]
        del os.environ["locan_TEST_DIR"]

    # noqa: S108
    def test_cache_from_environment_variable(self):
        config = Configuration()
        assert config.cache_directory == Path.home() / ".cache/locan"

        os.environ["XDG_CACHE_HOME"] = "/tmp/cdg_cache"
        config.__init__()
        assert config.cache_directory == Path("/tmp/cdg_cache")

        os.environ["locan_CACHE_DIR"] = "/tmp/env_cache"
        config.__init__()
        assert config.cache_directory == Path("/tmp/env_cache")

        os.environ["locan_CACHE_DIRECTORY"] = "/tmp/environment_cache"
        config.__init__()
        assert config.cache_directory == Path("/tmp/environment_cache")

        config.__init__(cache_directory="/tmp/init_cache")
        assert config.cache_directory == Path("/tmp/init_cache")

        # cleanup
        del os.environ["XDG_CACHE_HOME"]
        del os.environ["locan_CACHE_DIR"]
        del os.environ["locan_CACHE_DIRECTORY"]

    def test_init_with_toml_file(self):
        config = Configuration()
        # print(lc.config.model_config["toml_file"])
        toml_path = TEST_DIR / "test_data" / "locan_config.toml"
        config.model_config["toml_file"] = toml_path
        config.__init__()
        assert config.name == "TestConfiguration"
        assert config.data_directory == Path("some/path/to/data")

        # assert config.config_directory == Path.home() / ".config/locan"
        # assert config.cache_directory == Path.home() / ".cache/locan"
        # assert config.log_directory == Path.home() / ".local/state/locan"
        # assert config.test_directory == Path.cwd() / "tests/"

        assert isinstance(config.cache_directory, Path)
        assert isinstance(config.config_directory, Path)
        assert isinstance(config.log_directory, Path)
        assert isinstance(config.test_directory, Path)

    def test_from_toml(self):
        path = TEST_DIR / "test_data" / "locan_config.toml"
        config = Configuration.from_toml(path)
        assert config.name == "TestConfiguration"
        assert config.data_directory == Path("some/path/to/data")

    @pytest.mark.skip("Add table to pyproject.toml for testing.")
    def test_from_pyproject(self):
        # remember: change pyproject.toml for application that imports this package
        config = Configuration(n_cpus=4)
        print(config)
        print(get_configuration_file_path())
        print("CWD: ", Path.cwd())
        assert config.name == "PyprojectConfiguration"

    def test_to_toml(self, tmp_path):
        file_path = tmp_path / "locan_config.py"
        config = Configuration(
            name="TestConfiguration",
        )
        config.to_toml(path=file_path)

        new_config = Configuration.from_toml(file_path)
        assert config.name == new_config.name
        assert config.config_directory == new_config.config_directory
        assert config.data_directory == new_config.data_directory
