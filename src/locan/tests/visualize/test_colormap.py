import pytest
from matplotlib import colors as mcolors

from locan.configuration import COLORMAP_DEFAULTS
from locan.dependencies import HAS_DEPENDENCY
from locan.visualize import (
    Colormap,
    Colormaps,
    colormap_registry,
    get_colormap,
)

if HAS_DEPENDENCY["napari"]:
    import napari


def test_COLORMAP_DEFAULTS_and_Colormaps():
    assert "CONTINUOUS" in COLORMAP_DEFAULTS
    assert all(hasattr(Colormaps, key) for key in COLORMAP_DEFAULTS.keys())
    assert all(
        name in COLORMAP_DEFAULTS for name, value in Colormaps.__members__.items()
    )
    assert (
        COLORMAP_DEFAULTS[Colormaps.CONTINUOUS.name] == "cet_fire"
        if HAS_DEPENDENCY["colorcet"]
        else "viridis"
    )


def test_colormap_registry():
    assert all(value in colormap_registry for value in COLORMAP_DEFAULTS.values())


class TestColormap:
    def test_init(self):
        with pytest.raises(TypeError):
            Colormap()

        colormap = Colormap(colormap=mcolors.Colormap("viridis"))
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "viridis"

    @pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
    def test_init_with_napari(self):
        white_green_cmap = {
            "colors": [[1, 1, 1, 1], [0, 1, 0, 1]],
            "name": "white_to_green",
            "interpolation": "linear",
        }
        colormap = napari.utils.colormaps.Colormap(**white_green_cmap)
        colormap = Colormap(colormap=colormap)
        assert isinstance(colormap.napari, napari.utils.Colormap)
        assert colormap.name == "white_to_green"

    def test_from_matplotlib(self):
        colormap = Colormap.from_matplotlib(colormap="viridis")
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "viridis"

        colormap = Colormap.from_matplotlib(colormap=mcolors.Colormap("viridis"))
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "viridis"

    @pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
    def test_from_matplotlib_to_napari(self):
        colormap = Colormap.from_matplotlib(colormap="viridis")
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "viridis"
        assert isinstance(colormap.napari, napari.utils.Colormap)
        assert colormap.napari.name == "viridis"

    @pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
    def test_from_napari(self):
        white_green_cmap = {
            "colors": [[1, 1, 1, 1], [0, 1, 0, 1]],
            "name": "white_to_green",
            "interpolation": "linear",
        }
        colormap_ = napari.utils.colormaps.Colormap(**white_green_cmap)

        colormap = Colormap.from_napari(colormap=colormap_)
        assert isinstance(colormap.napari, napari.utils.Colormap)
        assert colormap.name == "white_to_green"
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "white_to_green"

        colormap = Colormap.from_napari(colormap=white_green_cmap)
        assert isinstance(colormap.napari, napari.utils.Colormap)
        assert colormap.name == "white_to_green"

        colormap = Colormap.from_napari(colormap="viridis")
        assert isinstance(colormap.napari, napari.utils.Colormap)
        assert colormap.name == "viridis"

    @pytest.mark.skipif(
        not HAS_DEPENDENCY["colorcet"], reason="Test requires colorcet."
    )
    def test_from_colorcet(self):
        colormap = Colormap.from_colorcet(colormap="cet_fire")
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "cet_fire"

        colormap = Colormap.from_colorcet(colormap="fire")
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "fire"

    def test_from_registry(self):
        colormap = Colormap.from_registry(colormap="viridis")
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "viridis"

        colormap = Colormap.from_registry(colormap=Colormaps.TURBO.value)
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "turbo"

        colormap = Colormap.from_registry(colormap=COLORMAP_DEFAULTS["TURBO"])
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "turbo"

        with pytest.raises(LookupError):
            Colormap.from_registry(colormap="not in registry")

    def test_call(self):
        colormap = Colormap.from_matplotlib(colormap="viridis")
        assert len(colormap(0.5)) == 4
        assert len(colormap([0, 0.5, 1])) == 3

    @pytest.mark.skipif(not HAS_DEPENDENCY["napari"], reason="Test requires napari.")
    def test_call_napari(self):
        colormap = Colormap.from_napari(colormap="viridis")
        assert colormap(0.5).shape == (1, 4)
        # assert colormap(0.5).tolist()[0] == [0.12814850801305894, 0.5651070037572249, 0.5508925046267017, 1.0]
        assert colormap([0, 0.5, 1]).shape == (3, 4)


def test_get_colormap():
    colormap = Colormap.from_matplotlib(colormap="viridis")
    colormap = get_colormap(colormap)
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "viridis"

    colormap = get_colormap("viridis")
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "viridis"

    colormap = get_colormap(Colormaps.TURBO)
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "turbo"

    colormap = get_colormap(Colormaps.TURBO.value)
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "turbo"

    with pytest.raises(TypeError):
        get_colormap("does not exist")
    with pytest.raises(TypeError):
        get_colormap([])
