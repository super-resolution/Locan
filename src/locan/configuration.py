"""

Project configuration

Settings and configuration management used throughout the project.
Pydantic-settings is used for type checking and validation.
A configuration is instantiated upon locan startup from either (in order of precedence):
    * [tool.locan] settings in pyproject.toml,
    * a locan_config.toml file,
    * a .env file,
    * environment variables,
    * or default settings.

.. autosummary::
   :toctree: ./

    CONFIGURATION_FILE_HIERARCHY

"""

from __future__ import annotations

import logging
import platform
import tomllib
import warnings
from abc import ABC
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, Self

import tomli_w
from pydantic import AliasChoices, BaseModel, Field, field_serializer
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from locan.constants import RenderEngine
from locan.dependencies import QtBindings, _set_qt_binding

logger = logging.getLogger(__name__)

__all__: list[str] = ["Configuration", "get_configuration_file_path"]


def get_configuration_file_hierarchy() -> list[Path]:
    """
    Set up a list of file path in order of precedence for configuration lookup.
    The hierarchy slightly varies with computer platform.

    Returns
    -------
    list[Path]
    """
    if platform.system() == "Windows":
        return [
            # Windows - user specific
            Path.cwd().resolve() / "locan_config.toml",
            Path.home().resolve() / ".locan" / "locan_config.toml",
            Path.home().resolve() / ".locan" / ".config" / "locan_config.toml",
            Path.home().resolve() / ".config" / "locan_config.toml",
            Path.home().resolve() / ".config" / "locan" / "locan_config.toml",
            Path.home().resolve() / "AppData" / "Local" / "locan" / "locan_config.toml",
            Path.home().resolve()
            / "AppData"
            / "Roaming"
            / "locan"
            / "locan_config.toml",
            # Windows - global
            Path("C:") / "ProgramData" / "locan" / "locan_config.toml",
        ]
    elif platform.system() == "Linux":
        return [
            # Linux - user specific
            Path.cwd().resolve() / "locan_config.toml",
            Path.home().resolve() / ".locan" / "locan_config.toml",
            Path.home().resolve() / ".locan" / ".config" / "locan_config.toml",
            Path.home().resolve() / ".config" / "locan_config.toml",
            Path.home().resolve() / ".config" / "locan" / "locan_config.toml",
            # Linux - global
            Path("/etc") / "locan" / "locan_config.toml",
            Path("/etc") / ".config" / "locan" / "locan_config.toml",
        ]
    elif platform.system() == "Darwin":
        return [
            # MacOS - user specific
            Path.cwd().resolve() / "locan_config.toml",
            Path.home().resolve() / ".locan" / "locan_config.toml",
            Path.home().resolve() / ".locan" / ".config" / "locan_config.toml",
            Path.home().resolve() / ".config" / "locan_config.toml",
            Path.home().resolve() / ".config" / "locan" / "locan_config.toml",
            Path.home().resolve()
            / "Library"
            / "Application Support"
            / "locan_config.toml",
            Path.home().resolve()
            / "Library"
            / "Application Support"
            / "locan"
            / "locan_config.toml",
        ]
    else:
        return [
            Path.cwd().resolve() / "locan_config.toml",
            Path.home().resolve() / ".locan" / "locan_config.toml",
            Path.home().resolve() / ".locan" / ".config" / "locan_config.toml",
            Path.home().resolve() / ".config" / "locan_config.toml",
            Path.home().resolve() / ".config" / "locan" / "locan_config.toml",
        ]


#: Look up hierarchy for configuration files
CONFIGURATION_FILE_HIERARCHY = get_configuration_file_hierarchy()


def get_configuration_file_path() -> Path | None:
    """
    Return the configuration file found in the configuration file hierarchy.

    Returns
    -------
    Path | None
    """
    for path_ in get_configuration_file_hierarchy():
        if path_.is_file():
            return path_


class ConfigurationBase(BaseModel, ABC):
    """
    Abstract base class for configuration management with pydantic.
    """

    version: str
    name: str

    @classmethod
    def from_toml(cls, file_path: str | PathLike) -> Self:
        with open(file_path, "rb") as file:
            config_dict = tomllib.load(file)

        config = cls(**config_dict)
        return config

    def to_toml(self, path: str | PathLike) -> None:
        path = Path(path)
        if path.is_dir():
            config_file = path / "config.toml"
        else:
            config_file = path

        config_dict = self.model_dump(exclude_none=True)

        with open(config_file, mode="wb") as file:
            tomli_w.dump(config_dict, file)

        return None

    def asdict(self, **kwargs: Any) -> dict[str, Any]:
        return self.model_dump(**kwargs)


class Colormaps(BaseModel):
    """
    Preferred colormap names to be used for visualization.

    Attributes
    ----------
    continuous: str
        colormap for gray-scale images.
    continuous_reverse: str
        colormap for gray-scale images inverted.
    continuous_gray: str
        colormap for gray-scale images.
    continuous_gray_reverse: str
        colormap for gray-scale images inverted.
    diverging: str
        colormap for signed gray-scale images.
    categorical: str
        colormap for images representing categories.
    turbo: str
        colormap for images representing rainbow-like colors.
    """

    continuous: str = "cet_fire"
    continuous_reverse: str = "cet_fire_r"
    continuous_gray: str = "cet_gray"
    continuous_gray_reverse: str = "cet_gray_r"
    diverging: str = "cet_coolwarm"
    categorical: str = "cet_glasbey_dark"
    turbo: str = "turbo"


class TqdmConfiguration(BaseModel):
    """
    Parameters for tqdm to control progress bars.

    Attributes
    ----------
    leave: bool | None
        Leave tqdm progress bars after finishing the iteration.
    disable: bool | None
        Disable tqdm progress bars.
    """

    leave: bool | None = Field(default=True)
    disable: bool | None = Field(default=False)


class Configuration(BaseSettings, ConfigurationBase):
    """
    Locan configuration settings that are used throughout the project.

    Attributes
    ----------
    version: str
        Version number of configuration.
    name: str
        Name of configuration.
    cache_directory: Path
        Directory where locan cache is stored.
    config_directory: Path
        Directory where locan configuration file is stored.
    data_directory: Path
        Directory where project data is stored.
    datasets_directory: Path
        Directory where locan datasets are stored.
    log_directory: Path
        Directory where locan logs are stored.
    test_directory: Path
        Directory where locan tests are stored.
    n_cpus: int
        Maximum number of CPUs used in multiprocessing.
        If negative, then the number of jobs is set to the number of CPU cores.
        In scikit-learn methods, this setting is passed through to n_jobs.
    colormaps: Colormaps
        Preferred colormap names to be used for visualization.
    render_engine: RenderEngine
        Preferred engine to be used for rendering localization data as 2d or 3d images.
        Each engine represents a library to be used as backend for rendering and plotting.
    tqdm: TqdmConfiguration
        Parameters for tqdm to control progress bars.
    qt_binding: QtBindings
        Preferred QT binding
    """

    model_config = SettingsConfigDict(
        env_prefix="locan_",
        env_nested_delimiter="_",
        validate_by_alias=True,
        validate_by_name=True,
        toml_file=CONFIGURATION_FILE_HIERARCHY,
        pyproject_toml_table_header=("tool", "locan"),
        pyproject_toml_depth=5,
    )

    version: str = "1"
    name: str = "locanConfiguration"
    # pyrefly: ignore [no-matching-overload]
    cache_directory: Path = Field(
        default=Path.home().resolve() / ".cache" / "locan",
        alias=AliasChoices(
            "locan_CACHE_DIRECTORY",
            "locan_CACHE_DIR",
            "XDG_CACHE_HOME",
        ),
    )
    # pyrefly: ignore [no-matching-overload]
    config_directory: Path = Field(
        default=Path.home().resolve() / ".config" / "locan",
        alias=AliasChoices(
            "locan_CONFIG_DIRECTORY",
            "locan_CONFIG_DIR",
            "XDG_CONFIG_HOME",
        ),
    )
    # pyrefly: ignore [no-matching-overload]
    data_directory: Path = Field(
        default=Path.home().resolve() / ".local" / "share" / "locan",
        alias=AliasChoices(
            "locan_DATA_DIRECTORY",
            "locan_DATA_DIR",
            "XDG_DATA_HOME",
        ),
    )
    # pyrefly: ignore [no-matching-overload]
    datasets_directory: Path = Field(
        default=Path.home().resolve() / ".cache" / "locan_datasets",
        alias=AliasChoices(
            "locan_DATASETS_DIRECTORY",
            "locan_DATASETS_DIR",
        ),
    )
    # pyrefly: ignore [no-matching-overload]
    log_directory: Path = Field(
        default=Path.home().resolve() / ".local/state" / "locan",
        alias=AliasChoices(
            "locan_LOG_DIRECTORY",
            "locan_LOG_DIR",
            "XDG_STATE_HOME",
        ),
    )
    # pyrefly: ignore [no-matching-overload]
    test_directory: Path = Field(
        default=Path.cwd().resolve() / "tests/",
        alias=AliasChoices(
            "locan_TEST_DIRECTORY",
            "locan_TEST_DIR",
        ),
    )
    n_cpus: int = Field(default=1, alias="locan_N_CPUS")
    colormaps: Colormaps = Field(default=Colormaps(), alias="locan_COLORMAPS")
    render_engine: RenderEngine = Field(
        default=RenderEngine.MPL, alias="locan_RENDER_ENGINE"
    )
    tqdm: TqdmConfiguration = Field(default=TqdmConfiguration(), alias="locan_TQDM")
    # pyrefly: ignore [no-matching-overload]
    qt_binding: QtBindings = Field(
        default="",
        alias=AliasChoices(
            # "QT_API",
            # QT_API is checked in locan.dependencies._set_qt_binding
            "locan_QT_BINDING",
        ),
    )

    @field_serializer(
        "cache_directory",
        "config_directory",
        "data_directory",
        "datasets_directory",
        "log_directory",
        "test_directory",
        mode="plain",
    )
    def serialize_paths(self, value: str) -> str:
        return str(value)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_settings = TomlConfigSettingsSource(
            settings_cls,
        )
        pyproject_settings = PyprojectTomlConfigSettingsSource(
            settings_cls,
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            toml_settings,
            pyproject_settings,
        )


config: Configuration = Configuration()


# Preferred QT binding. At this point this variable cannot be set at
# runtime.
QT_BINDING: QtBindings | str = config.qt_binding

# Initialize pyqt; this also sets HAS_DEPENDENCY["qt"]
QT_BINDING = _set_qt_binding(QT_BINDING)
# pyrefly: ignore [bad-assignment]
config.qt_binding = QT_BINDING


def __getattr__(name):
    if name == "COLORMAP_DEFAULTS":
        warnings.warn(
            "locan.configuration.COLORMAP_DEFAULTS is deprecated; Use locan.config.colormaps instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        #: Mapping a type of colormap to its default colormap that is used throughout locan.
        #: See details in locan documentation on colormaps.
        COLORMAP_DEFAULTS: Mapping[str, str] = dict()

        COLORMAP_DEFAULTS = dict(
            CONTINUOUS="cet_fire",
            CONTINUOUS_REVERSE="cet_fire_r",
            CONTINUOUS_GRAY="cet_gray",
            CONTINUOUS_GRAY_REVERSE="cet_gray_r",
            DIVERGING="cet_coolwarm",
            CATEGORICAL="cet_glasbey_dark",
            TURBO="turbo",
        )

        return COLORMAP_DEFAULTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
