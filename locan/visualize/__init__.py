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

from locan.visualize import colormap, render, render_mpl, render_napari, transform

from .colormap import *
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
