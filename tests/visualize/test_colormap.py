import pytest
from matplotlib import colors as mcolors

from locan import config
from locan.dependencies import HAS_DEPENDENCY
from locan.visualize import (
    Colormap,
    colormap_registry,
    get_colormap,
)

if HAS_DEPENDENCY["napari"]:
    import napari  # type: ignore [missing-import]


def test_Colormaps():
    # todo: deprecate with locan.visualize.colormaps.Colormaps
    # Deprecated names are exercised on purpose; importing them must warn.
    with pytest.warns(DeprecationWarning, match="COLORMAP_DEFAULTS is deprecated"):
        from locan.configuration import COLORMAP_DEFAULTS
    with pytest.warns(DeprecationWarning, match="Colormaps is deprecated"):
        from locan.visualize import Colormaps

    assert Colormaps.continuous.value == "cet_fire"
    assert Colormaps.CONTINUOUS.value == "cet_fire"

    assert COLORMAP_DEFAULTS[Colormaps.CONTINUOUS_REVERSE.name.upper()] == "cet_fire_r"
    assert COLORMAP_DEFAULTS[Colormaps.CONTINUOUS_GRAY.name.upper()] == "cet_gray"
    assert (
        COLORMAP_DEFAULTS[Colormaps.CONTINUOUS_GRAY_REVERSE.name.upper()]
        == "cet_gray_r"
    )
    assert COLORMAP_DEFAULTS[Colormaps.DIVERGING.name.upper()] == "cet_coolwarm"
    assert COLORMAP_DEFAULTS[Colormaps.CATEGORICAL.name.upper()] == "cet_glasbey_dark"
    assert COLORMAP_DEFAULTS[Colormaps.TURBO.name.upper()] == "turbo"


def test_colormap_registry():
    assert all(value in colormap_registry for value in config.colormaps.model_dump().values())


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

        # from locan default colormaps that are not in napari
        colormap = colormap_registry["cet_fire"]
        assert isinstance(colormap.napari, napari.utils.Colormap)
        assert colormap.name == "cet_fire"

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

        with pytest.raises(LookupError):
            Colormap.from_napari(colormap="not in napari")

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
        colormap = Colormap.from_registry(colormap="cet_fire")
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "cet_fire"

        colormap = Colormap.from_registry(colormap="viridis")
        assert isinstance(colormap.matplotlib, mcolors.Colormap)
        assert colormap.name == "viridis"

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

    colormap = get_colormap(config.colormaps.turbo)
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "turbo"

    with pytest.raises(TypeError):
        get_colormap("does not exist")
    with pytest.raises(TypeError):
        get_colormap([])


def test_colormap_definitions():
    colormap = get_colormap("cet_fire")
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "cet_fire"

    colormap = get_colormap("cet_fire_r")
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "cet_fire_r"

    colormap = get_colormap("cet_gray")
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "cet_gray"

    colormap = get_colormap("cet_gray_r")
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "cet_gray_r"

    colormap = get_colormap("cet_glasbey_dark")
    assert isinstance(colormap.matplotlib, mcolors.Colormap)
    assert colormap.name == "cet_glasbey_dark"
