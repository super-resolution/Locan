"""

Configuration variables used throughout the project.

.. autosummary::
   :toctree: ./

   DATASETS_DIR
   RENDER_ENGINE
   N_JOBS
   COLORMAP_CONTINUOUS
   COLORMAP_DIVERGING
   COLORMAP_CATEGORICAL
   TQDM_LEAVE
   TQDM_DISABLE
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors

from locan.dependencies import HAS_DEPENDENCY

if HAS_DEPENDENCY["colorcet"]:
    from colorcet import m_fire, m_gray, m_coolwarm, m_glasbey_dark

from locan.constants import RenderEngine
from locan.dependencies import QtBindings, _set_qt_binding

__all__ = [
    "DATASETS_DIR",
    "RENDER_ENGINE",
    "N_JOBS",
    "TQDM_LEAVE",
    "TQDM_DISABLE",
    "COLORMAP_CONTINUOUS",
    "COLORMAP_DIVERGING",
    "COLORMAP_CATEGORICAL",
    "QT_BINDING",
]


#: Standard directory for example datasets.
DATASETS_DIR = Path.home() / "LocanDatasets"

#: Render engine.
RENDER_ENGINE = RenderEngine.MPL

#: The number of cores that are used in parallel for some algorithms.
#: Following the scikit convention: n_jobs is the number of parallel jobs to run.
#: If -1, then the number of jobs is set to the number of CPU cores.
N_JOBS: int = 1

#: Leave tqdm progress bars after finishing the iteration.
#: Flag to leave tqdm progress bars.
TQDM_LEAVE: bool = True

#: Disable tqdm progress bars.
#: Flag to disable all tqdm progress bars.
TQDM_DISABLE: bool = False

#: Default colormap for continuous scales. Default is `colorcet.m_fire` if installed or 'viridis'.
COLORMAP_CONTINUOUS: mcolors.Colormap | str = "viridis"

#: Default colormap for diverging scales. Default is `colorcet.m_coolwarm` if installed or 'coolwarm'.
COLORMAP_DIVERGING: mcolors.Colormap | str = "coolwarm"

#: Default colormap for categorical scales. Default is `colorcet.m_glasbey_dark` if installed or 'tab20'.
COLORMAP_CATEGORICAL: mcolors.Colormap | str = "tab20"

if HAS_DEPENDENCY["colorcet"]:
    COLORMAP_CONTINUOUS = m_fire
    COLORMAP_DIVERGING = m_coolwarm
    COLORMAP_CATEGORICAL = m_glasbey_dark

# Preferred QT binding. At this point this variable cannot be set at runtime.
QT_BINDING: QtBindings | str = ""

# Initialize pyqt
QT_BINDING = _set_qt_binding(QT_BINDING)
