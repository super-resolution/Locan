from pathlib import Path

import matplotlib.colors as mcolors

from locan.configuration import (
    COLORMAP_CATEGORICAL,
    COLORMAP_CONTINUOUS,
    COLORMAP_DIVERGING,
    DATASETS_DIR,
    N_JOBS,
    QT_BINDING,
    RENDER_ENGINE,
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


def test_QT_BINDING():
    assert QT_BINDING
