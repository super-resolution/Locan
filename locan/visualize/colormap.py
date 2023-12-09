"""

Provide colormaps for visualization.

This module provides convenience functions for using colormap definitions
from various visualization packages.

Default colormaps in locan are set through the
:py:data:`locan.configuration.COLORMAP_DEFAULTS` mapping.

Named colormaps are registered through the
:py:data:`locan.visualization.colormap.colormap_registry` mapping.

In locan :py:class:`locan.Colormap` serves as adapter class to provide an
interface for various visualization functions.
Instances of :py:class:`locan.Colormap` can be requested through the
:py:func:`locan.visualization.colormap.get_colormap` function
and contain references to matplotlib and napari colormap instances.

Examples
--------
>>> colormap = locan.get_colormap("viridis")
>>> assert isinstance(colormap.matplotlib, mcolors.Colormap)
>>> colormap.name
viridis

>>> colormap = locan.Colormap.from_matplotlib(colormap="viridis")
>>> assert isinstance(colormap.matplotlib, mcolors.Colormap)
>>> colormap.name
viridis

Variables
----------

.. autosummary::
   :toctree: ./

   colormap_registry

"""
from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from locan.configuration import COLORMAP_DEFAULTS
from locan.dependencies import HAS_DEPENDENCY, needs_package

if HAS_DEPENDENCY["colorcet"]:
    import colorcet

if HAS_DEPENDENCY["napari"]:
    import napari
    import vispy

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "Colormaps",
    "Colormap",
    "colormap_registry",
    "get_colormap",
]

MatplotlibColormap: TypeAlias = mcolors.Colormap
if HAS_DEPENDENCY["napari"]:
    NapariColormap: TypeAlias = napari.utils.Colormap
    VispyColormap: TypeAlias = vispy.color.Colormap
T_Colormap = TypeVar("T_Colormap", bound="Colormap")

Colormaps = Enum("Colormaps", COLORMAP_DEFAULTS, module=__name__)  # type: ignore[misc]
Colormaps.__doc__ = """
Preferred colormap types to be used for visualization.

Note
----
This enum is automatically generated from COLORMAP_DEFAULTS and should
not be modified.
"""


class Colormap:
    """
    Container class for colormaps.

    A locan Colormap can be instantiated from other colormaps and serves as
    adapter class.
    """

    def __init__(
        self, colormap: MatplotlibColormap | NapariColormap | VispyColormap
    ) -> None:
        self._name: str | None = None
        self._matplotlib: mcolors.Colormap | None = None
        self._napari: napari.utils.Colormap | None = None

        if isinstance(colormap, mcolors.Colormap):
            self._matplotlib = colormap
        elif isinstance(colormap, (napari.utils.Colormap, vispy.color.Colormap)):
            self._napari = colormap
        else:
            raise TypeError(
                "The colormap type is not supported. "
                "Try to use appropriate class methods for construction."
            )

    @property
    def name(self) -> str:
        if self._name is None:
            if self._matplotlib:
                self._name = self._matplotlib.name
            elif self._napari:
                self._name = self._napari.name
            else:
                raise ValueError("No colormap available")
        return self._name

    @property
    def matplotlib(self) -> mcolors.Colormap:
        if self._matplotlib is None:
            if self._napari is not None:
                name = self.napari.name
                colors = self.napari.colors
                controls = self.napari.controls
                self._matplotlib = mcolors.LinearSegmentedColormap.from_list(
                    name, list(zip(controls, colors))
                )
            else:
                raise ValueError("No colormap available")
        return self._matplotlib

    @property
    @needs_package("napari")
    def napari(self) -> napari.utils.Colormap:
        if self._napari is None:
            if self._matplotlib is not None:
                self._napari = (
                    napari.utils.colormaps.colormap_utils.vispy_or_mpl_colormap(
                        name=self.matplotlib.name
                    )
                )
            else:
                raise ValueError("No colormap available")
        return self._napari

    @classmethod
    def from_registry(cls: type[T_Colormap], colormap: str) -> Colormap:
        if colormap in colormap_registry:
            return colormap_registry[colormap]
        else:
            raise LookupError("The colormap is not in registry.")

    @classmethod
    def from_matplotlib(
        cls: type[T_Colormap], colormap: str | mcolors.Colormap
    ) -> T_Colormap:
        _matplotlib = plt.get_cmap(name=colormap)
        return cls(colormap=_matplotlib)

    @classmethod
    @needs_package("napari")
    def from_napari(
        cls: type[T_Colormap], colormap: str | dict[str, Any] | NapariColormap
    ) -> T_Colormap:
        if isinstance(colormap, str):
            _napari = napari.utils.colormaps.ensure_colormap(colormap=colormap)
        elif isinstance(colormap, napari.utils.Colormap):
            _napari = colormap
        elif isinstance(colormap, dict):
            _napari = napari.utils.Colormap(**colormap)
        else:
            raise TypeError("Cannot create napari.utils.Colormap from colormap input.")
        return cls(colormap=_napari)

    @classmethod
    @needs_package("colorcet")
    def from_colorcet(cls: type[T_Colormap], colormap: str) -> T_Colormap:
        if isinstance(colormap, str):
            if colormap.startswith("cet_"):
                _matplotlib = mcolors.Colormap(colormap)
            else:
                _matplotlib = colorcet.cm[colormap]
        else:
            raise TypeError("Cannot create Colormap from colormap input.")
        return cls(colormap=_matplotlib)

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> float | tuple[float, float, float, float] | npt.NDArray[np.float_]:
        if HAS_DEPENDENCY["matplotlib"] and self._matplotlib is not None:
            return_value = self._matplotlib(*args, **kwargs)
        elif HAS_DEPENDENCY["napari"] and self._napari is not None:
            return_value = self._napari.map(*args, **kwargs)
        else:
            raise NotImplementedError("There is no backend for colormaps available.")
        return return_value  # type: ignore


_colormap_registry_matplotlib: dict[str, Colormap] = {
    "viridis": Colormap.from_matplotlib("viridis"),
    "viridis_r": Colormap.from_matplotlib("viridis_r"),
    "gray": Colormap.from_matplotlib("gray"),
    "gray_r": Colormap.from_matplotlib("gray_r"),
    "turbo": Colormap.from_matplotlib("turbo"),
    "coolwarm": Colormap.from_matplotlib("coolwarm"),
    "tab20": Colormap.from_matplotlib("tab20"),
}

if HAS_DEPENDENCY["colorcet"]:
    _colormap_registry_colorcet: dict[str, Colormap] = {
        "cet_fire": Colormap.from_matplotlib("cet_fire"),
        "cet_fire_r": Colormap.from_matplotlib("cet_fire_r"),
        "cet_gray": Colormap.from_matplotlib("cet_gray"),
        "cet_gray_r": Colormap.from_matplotlib("cet_gray_r"),
        "turbo": Colormap.from_matplotlib("turbo"),
        "cet_coolwarm": Colormap.from_matplotlib("cet_coolwarm"),
        "cet_glasbey_dark": Colormap.from_matplotlib("cet_glasbey_dark"),
    }
else:
    _colormap_registry_colorcet = {}

#: A mapping of names onto Colormap instances.
colormap_registry: Mapping[str, Colormap] = (
    _colormap_registry_matplotlib | _colormap_registry_colorcet
)


ColormapType = Union[
    str,
    Colormaps,
    Colormap,
    mcolors.Colormap,
    "napari.utils.Colormap",
]


def get_colormap(colormap: ColormapType) -> Colormap:
    """
    Get a locan.Colormap instance from colormap searching string identifier through
    colormap_registry, matplotlib colormaps, napari_colormaps.

    Parameters
    ----------
    colormap
        Colormap request

    Returns
    -------
    Colormap
    """
    if isinstance(colormap, Colormap):
        return colormap
    elif isinstance(colormap, mcolors.Colormap):
        return Colormap.from_matplotlib(colormap=colormap)
    elif isinstance(colormap, Colormaps):
        return Colormap.from_registry(colormap.value)
    elif HAS_DEPENDENCY["napari"] and isinstance(colormap, napari.utils.Colormap):
        return Colormap.from_napari(colormap=colormap)
    elif isinstance(colormap, str):
        try:
            return Colormap.from_registry(colormap)
        except LookupError:
            pass
        try:
            return Colormap.from_matplotlib(colormap=colormap)
        except ValueError:
            pass
        if HAS_DEPENDENCY["napari"]:
            try:
                return Colormap.from_napari(colormap=colormap)
            except KeyError:
                pass
        raise TypeError(
            f"The colormap {colormap} is not available in either colormap_registry, "
            f"matplotlib or napari."
        )
    else:
        raise TypeError("No such colormap available.")
