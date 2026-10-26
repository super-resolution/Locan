"""

Visualize localization data.

This module provides functions to visualize localization data
with matplotlib or napari.

Submodules:
-----------

.. autosummary::
   :toctree: ./

   colormap
   render
   render_mpl
   render_napari
   transform

"""

from __future__ import annotations

import warnings
from typing import Any

from locan.visualize import colormap, render, render_mpl, render_napari, transform

# `Colormaps` is deprecated and provided lazily by `colormap.__getattr__`; a star
# import would trigger its DeprecationWarning on every `import locan`.
from .colormap import (
    Colormap as Colormap,
    colormap_registry as colormap_registry,
    get_colormap as get_colormap,
)
from .render import *
from .render_mpl import (
    apply_window as apply_window,
    render_2d_mpl as render_2d_mpl,
    render_2d_rgb_mpl as render_2d_rgb_mpl,
    render_2d_scatter_density as render_2d_scatter_density,
    scatter_2d_mpl as scatter_2d_mpl,
    scatter_3d_mpl as scatter_3d_mpl,
)
from .render_napari import (
    get_rois as get_rois,
    render_2d_napari as render_2d_napari,
    render_2d_napari_image as render_2d_napari_image,
    render_2d_rgb_napari as render_2d_rgb_napari,
    render_3d_napari as render_3d_napari,
    render_3d_napari_image as render_3d_napari_image,
    render_3d_rgb_napari as render_3d_rgb_napari,
    save_rois as save_rois,
    select_by_drawing_napari as select_by_drawing_napari,
)
from .transform import *

__all__: list[str] = []
__all__.extend(colormap.__all__)
__all__.extend(render.__all__)
__all__.extend(render_mpl.__all__)
__all__.extend(render_napari.__all__)
__all__.extend(transform.__all__)


def __getattr__(name: str) -> Any:
    # Forward deprecated names so the warning fires on use, not on import,
    # and is attributed to the caller rather than to this forwarder.
    if name == "Colormaps":
        warnings.warn(
            colormap.COLORMAPS_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return colormap.Colormaps
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
