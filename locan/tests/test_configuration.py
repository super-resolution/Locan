from pathlib import Path

import matplotlib.colors as mcolors

from locan.configuration import (
    DATASETS_DIR,
    RENDER_ENGINE,
    N_JOBS,
    COLORMAP_CONTINUOUS,
    COLORMAP_DIVERGING,
    COLORMAP_CATEGORICAL,
)
from locan.constants import RenderEngine


def test_datasets_dir():
    # print(DATASETS_DIR)
    assert isinstance(DATASETS_DIR, Path)


def test_render_enginge():
    assert isinstance(RENDER_ENGINE, RenderEngine)


def test_n_jobs():
    assert isinstance(N_JOBS, int)


def test_colormaps():
    colormaps = [COLORMAP_CONTINUOUS, COLORMAP_DIVERGING, COLORMAP_CATEGORICAL]
    for item in colormaps:
        assert isinstance(item, mcolors.Colormap) or isinstance(
            mcolors.Colormap(item), mcolors.Colormap
        )
