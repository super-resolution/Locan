"""

Render localization data.

This module provides convenience functions for rendering localization data.

"""
import logging

from locan.configuration import RENDER_ENGINE
from locan.constants import RenderEngine
from locan.dependencies import HAS_DEPENDENCY
from locan.visualize.napari.render2d import render_2d_napari
from locan.visualize.napari.render3d import render_3d_napari
from locan.visualize.render_mpl import render_2d_mpl, render_2d_scatter_density

__all__ = [
    "render_2d",
    "render_3d",
]

logger = logging.getLogger(__name__)


def render_2d(locdata, render_engine=RENDER_ENGINE, **kwargs):
    """
    Wrapper function to render localization data into a 2D image.
    For complete signatures see render_2d_mpl or corresponding functions.
    """
    if render_engine == RenderEngine.MPL:
        return render_2d_mpl(locdata, **kwargs)
    elif (
        HAS_DEPENDENCY["mpl_scatter_density"]
        and render_engine == RenderEngine.MPL_SCATTER_DENSITY
    ):
        return render_2d_scatter_density(locdata, **kwargs)
    elif HAS_DEPENDENCY["napari"] and render_engine == RenderEngine.NAPARI:
        return render_2d_napari(locdata, **kwargs)
    else:
        raise NotImplementedError(f"render_2d is not implemented for {render_engine}.")


def render_3d(locdata, render_engine=RENDER_ENGINE, **kwargs):
    """
    Wrapper function to render localization data into a 3D image.
    For complete signatures see render_3d_mpl or corresponding functions.
    """
    if HAS_DEPENDENCY["napari"] and render_engine == RenderEngine.NAPARI:
        return render_3d_napari(locdata, **kwargs)
    else:
        raise NotImplementedError(f"render_3d is not implemented for {render_engine}.")
