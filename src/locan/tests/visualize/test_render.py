import matplotlib.pyplot as plt
import pytest

from locan import RenderEngine, render_2d, render_3d
from locan.dependencies import HAS_DEPENDENCY


@pytest.mark.gui
@pytest.mark.parametrize(
    "test_input, expected", list((member, 0) for member in list(RenderEngine))
)
def test_render_2d(locdata_blobs_2d, test_input, expected):
    if HAS_DEPENDENCY["napari"] and test_input == RenderEngine.NAPARI:
        render_2d(locdata_blobs_2d, render_engine=test_input)
        # napari.run()
    else:
        render_2d(locdata_blobs_2d, render_engine=test_input)
    # plt.show()

    plt.close("all")


@pytest.mark.gui
@pytest.mark.parametrize(
    "test_input, expected", list((member, 0) for member in list(RenderEngine))
)
def test_render_3d(locdata_blobs_3d, test_input, expected):
    if HAS_DEPENDENCY["napari"] and test_input == RenderEngine.NAPARI:
        render_3d(locdata_blobs_3d, render_engine=test_input)
        # napari.run()
    else:
        with pytest.raises(NotImplementedError):
            render_3d(locdata_blobs_3d, render_engine=test_input)
    # plt.show()

    plt.close("all")
