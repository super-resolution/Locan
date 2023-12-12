"""

Configuration variables used throughout the project.

.. autosummary::
   :toctree: ./

   DATASETS_DIR
   RENDER_ENGINE
   N_JOBS
   COLORMAP_DEFAULTS
   TQDM_LEAVE
   TQDM_DISABLE
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from locan.constants import RenderEngine
from locan.dependencies import HAS_DEPENDENCY, QtBindings, _set_qt_binding

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "DATASETS_DIR",
    "RENDER_ENGINE",
    "N_JOBS",
    "TQDM_LEAVE",
    "TQDM_DISABLE",
    "COLORMAP_DEFAULTS",
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


#: Mapping a type of colormap to its default colormap that is used throughout locan.
#: See details in locan documentation on colormaps.
COLORMAP_DEFAULTS: Mapping[str, str] = dict()

if HAS_DEPENDENCY["colorcet"]:
    COLORMAP_DEFAULTS = dict(
        CONTINUOUS="cet_fire",
        CONTINUOUS_REVERSE="cet_fire_r",
        CONTINUOUS_GRAY="cet_gray",
        CONTINUOUS_GRAY_REVERSE="cet_gray_r",
        DIVERGING="cet_coolwarm",
        CATEGORICAL="cet_glasbey_dark",
        TURBO="turbo",
    )
else:
    COLORMAP_DEFAULTS = dict(
        CONTINUOUS="viridis",
        CONTINUOUS_REVERSE="viridis_r",
        CONTINUOUS_GRAY="gray",
        CONTINUOUS_GRAY_REVERSE="gray_r",
        DIVERGING="coolwarm",
        CATEGORICAL="tab20",
        TURBO="turbo",
    )

# Preferred QT binding. At this point this variable cannot be set at
# runtime.
QT_BINDING: QtBindings | str = ""

# Initialize pyqt
QT_BINDING = _set_qt_binding(QT_BINDING)
HAS_DEPENDENCY["qt"] = True if QT_BINDING else False
